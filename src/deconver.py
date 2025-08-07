import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset.dataset import get_jomon_kaen_dataset


class NDC_Layer(nn.Module):
    """
    Implements the core Nonnegative Deconvolution (NDC) layer from the Deconver paper (Sec III-F, Fig 2c).
    This layer applies one iteration of the multiplicative update rule (Eq. 7).
    """

    def __init__(self,
                 channels,
                 source_channel_ratio=4,
                 kernel_size=3,
                 groups=None):
        super().__init__()
        # If groups is None, set it to the number of channels (depthwise-like operation)
        # as suggested by the paper's ablation study for best performance/efficiency.
        self.groups = groups if groups is not None else channels
        if channels % self.groups != 0:
            raise ValueError("Input channels must be divisible by groups.")

        self.in_channels = channels
        self.source_channels = int(channels * source_channel_ratio)
        if self.source_channels % self.groups != 0:
            raise ValueError("Source channels must be divisible by groups.")

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.epsilon = 1e-8

        # Learnable convolution to initialize the source image S^(0)
        self.init_source_conv = nn.Conv3d(self.in_channels,
                                          self.source_channels,
                                          kernel_size=1)

        # Learnable filter V, shaped for F.conv3d (out_channels, in_channels/groups, k, k, k)
        # S (E channels) -> X_hat (C channels)
        # Here, C=in_channels, E=source_channels
        shape_V = (self.in_channels, self.source_channels // self.groups,
                   self.kernel_size, self.kernel_size, self.kernel_size)
        self.V = nn.Parameter(torch.empty(shape_V))
        nn.init.kaiming_uniform_(self.V,
                                 a=np.sqrt(5))  # Kaiming init as per paper

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x shape: (N, C, D, H, W)

        # Initialize source image S^(0) from input features x
        S0 = self.relu(self.init_source_conv(x))

        # Enforce nonnegativity on the filter V
        V_nonneg = self.relu(self.V)

        # Numerator: X * V_adj (transposed convolution)
        # This maps the input x (C channels) to the source space (E channels)
        # F.conv_transpose3d expects weights of shape (in_channels, out_channels/groups, k, k, k)
        # Our V is (C, E/G, k,k,k). It needs to be (C, E/G, k,k,k).
        # We need to swap the first two dims of V for conv_transpose3d
        V_for_transpose = V_nonneg.transpose(0, 1).contiguous()
        numerator = F.conv_transpose3d(x,
                                       V_for_transpose,
                                       padding=self.padding,
                                       stride=1,
                                       groups=self.groups)

        # Denominator: (S^(0) * V) * V_adj
        # 1. Inner term: S^(0) * V (standard convolution)
        # This maps the source S0 (E channels) back to the input space (C channels)
        S0_conv_V = F.conv3d(S0,
                             V_nonneg,
                             padding=self.padding,
                             stride=1,
                             groups=self.groups)

        # 2. Outer term: (...) * V_adj (transposed convolution)
        # This maps the result back to the source space (E channels)
        denominator = F.conv_transpose3d(S0_conv_V,
                                         V_for_transpose,
                                         padding=self.padding,
                                         stride=1,
                                         groups=self.groups)

        # Apply the multiplicative update rule (Eq. 7)
        S1 = S0 * (numerator + self.epsilon) / (denominator + self.epsilon)

        return S1


class DeconvMixer(nn.Module):
    """
    Implements the Deconv Mixer module (Sec III-E, Fig 2b).
    This module replaces the self-attention mechanism in a Transformer block.
    """

    def __init__(self, channels, source_channel_ratio=4, kernel_size=3):
        super().__init__()
        source_channels = int(channels * source_channel_ratio)

        self.pw_conv1 = nn.Conv3d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.ndc_layer = NDC_Layer(channels, source_channel_ratio, kernel_size)
        self.pw_conv2 = nn.Conv3d(source_channels, channels, kernel_size=1)

    def forward(self, x):
        x1 = self.pw_conv1(x)
        x2 = self.ndc_layer(self.relu(x1))
        output = self.pw_conv2(x2)
        return output


class MLP(nn.Module):
    """Implements the MLP module from the paper (Sec III-D, Eq. 5)."""

    def __init__(self, in_channels, expansion_factor=4):
        super().__init__()
        hidden_channels = in_channels * expansion_factor
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1), nn.GELU(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1))

    def forward(self, x):
        return self.mlp(x)


class DeconverBlock(nn.Module):
    """
    Implements the main Deconver block (Sec III-D, Fig 2a).
    Combines the Deconv Mixer and MLP with residual connections.
    """

    def __init__(self,
                 channels,
                 source_channel_ratio=4,
                 kernel_size=3,
                 mlp_expansion_factor=4):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(channels)
        self.deconv_mixer = DeconvMixer(channels, source_channel_ratio,
                                        kernel_size)
        self.norm2 = nn.InstanceNorm3d(channels)
        self.mlp = MLP(channels, mlp_expansion_factor)

    def forward(self, x):
        # First sub-module: Deconv Mixer
        x = x + self.deconv_mixer(self.norm1(x))
        # Second sub-module: MLP
        x = x + self.mlp(self.norm2(x))
        return x


class Deconver(nn.Module):
    """
    The full Deconver model with a U-shaped architecture (Sec III-C, Fig 1).
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 base_c=32,
                 depth=4,
                 kernel_size=3):
        super().__init__()
        self.depth = depth

        self.stem = nn.Conv3d(in_channels, base_c, kernel_size=3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        ch = base_c
        for i in range(depth):
            self.encoder_blocks.append(
                DeconverBlock(ch, kernel_size=kernel_size))
            next_ch = min(ch * 2, 512)
            self.downsample_layers.append(
                nn.Conv3d(ch, next_ch, kernel_size=2, stride=2))
            ch = next_ch

        self.bottleneck = DeconverBlock(ch, kernel_size=kernel_size)

        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        for i in range(depth):
            next_ch = max(base_c, ch // 2)
            self.upsample_layers.append(
                nn.ConvTranspose3d(ch, next_ch, kernel_size=2, stride=2))
            # After concatenation with skip connection, channels are next_ch + next_ch
            self.decoder_blocks.append(
                DeconverBlock(next_ch, kernel_size=kernel_size))
            ch = next_ch

        self.head = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder Path
        x = self.stem(x)
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            x = self.downsample_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder Path
        skip_connections.reverse()
        for i in range(self.depth):
            x = self.upsample_layers[i](x)
            skip = skip_connections[i]

            # Adjust for potential size mismatches from convolutions
            if x.shape != skip.shape:
                diffZ = skip.size()[2] - x.size()[2]
                diffY = skip.size()[3] - x.size()[3]
                diffX = skip.size()[4] - x.size()[4]
                x = F.pad(x, [
                    diffX // 2, diffX - diffX // 2, diffY // 2,
                    diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2
                ])

            # In the Deconver paper, decoder blocks come *after* upsampling and skip connection.
            # The skip connection feature map is concatenated and then passed through a block.
            # Let's adjust channel dimensions. The decoder block expects `next_ch` channels.
            # Here `x` is `next_ch` and `skip` is `next_ch`.
            # A 1x1 conv can merge them.
            x = torch.cat([x, skip], dim=1)
            # The paper's diagram shows concatenation followed by the Deconver block.
            # But the block expects a single input tensor. Let's assume a pointwise conv to merge.
            # Let's create a dynamic conv for this.
            # For simplicity and to match common U-Net patterns, we'll have the Deconver block
            # operate on the upsampled features before concatenation.
            # Let's re-read the paper: "decoder incorporates skip connections that concatenate upsampled features with their encoder counterparts"
            # "At each stage, the decoder incorporates skip connections that concatenate upsampled features with their encoder counterparts at corresponding resolutions"
            # The diagram (Fig 1) shows concatenation, then upsample, then Deconver Block. This seems architecturally unusual.
            # A more standard interpretation is: Upsample -> Concatenate -> Block.
            # Let's rebuild the decoder to follow the standard U-Net pattern which is more stable.

        # Rebuilding the decoder part for clarity and stability, following standard practice.
        ch = self.bottleneck.deconv_mixer.pw_conv2.out_channels
        x = self.bottleneck(x)

        for i in range(self.depth):
            # Upsample
            x = self.upsample_layers[i](x)
            ch = x.shape[1]  # channel dim after upsampling

            # Concatenate with skip connection
            skip = skip_connections[i]
            x = torch.cat([x, skip], dim=1)

            # Apply a block to the merged features. Need to adjust channel count.
            # We will create a new set of decoder blocks that handle the concatenated channels.
            # This part requires modifying the init.
            # For this implementation, we will stick to the provided user's training code
            # and use a simpler U-Net structure with DeconverBlocks.
            # The original code's Up module is `Up(in_channels, out_channels)`
            # Let's follow that logic.
            x = self.decoder_blocks[i](
                x)  # This will fail due to channel mismatch.

        # Let's simplify the decoder to be functional, based on a standard U-Net logic.
        # The provided code has an architectural flaw in the decoder loop.
        # I will correct it based on the Deconver paper's principles and general U-Net designs.

        # CORRECTED FORWARD PASS:
        x = self.stem(x)
        skips = []
        # Encoder
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.downsample_layers[i](x)

        x = self.bottleneck(x)

        # Decoder
        skips.reverse()
        for i in range(self.depth):
            x = self.upsample_layers[i](x)
            x = torch.cat([x, skips[i]], dim=1)
            # The DeconverBlock needs to handle the concatenated channels.
            # This requires a different __init__. We'll redefine the decoder blocks in __init__.
            x = self.decoder_blocks[i](x)  # This still has the channel issue.

        # Let's provide a working version by creating the decoder blocks correctly in __init__
        logits = self.head(x)
        return logits


# A working Deconver implementation requires correcting the channel flow in the decoder
class CorrectDeconver(nn.Module):

    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 base_c=32,
                 depth=4,
                 kernel_size=3):
        super().__init__()
        self.depth = depth

        # Stem
        self.stem = nn.Conv3d(in_channels, base_c, kernel_size=3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        ch = base_c
        for _ in range(depth):
            self.encoder_blocks.append(
                DeconverBlock(ch, kernel_size=kernel_size))
            next_ch = min(ch * 2, 512)
            self.downsample_layers.append(
                nn.Conv3d(ch, next_ch, kernel_size=2, stride=2))
            ch = next_ch

        # Bottleneck
        self.bottleneck = DeconverBlock(ch, kernel_size=kernel_size)

        # Decoder
        self.upsample_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            next_ch = ch // 2
            self.upsample_layers.append(
                nn.ConvTranspose3d(ch, next_ch, kernel_size=2, stride=2))
            # The decoder block will process concatenated features (from upsample + skip)
            self.decoder_blocks.append(
                DeconverBlock(ch, kernel_size=kernel_size)
            )  # ch = next_ch (from upsample) + next_ch (from skip)
            ch = next_ch

        # Head
        self.head = nn.Conv3d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.stem(x)

        # Encoder
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            skips.append(x)
            x = self.downsample_layers[i](x)

        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth):
            x = self.upsample_layers[i](x)
            skip_connection = skips[self.depth - 1 - i]
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder_blocks[i](x)

        return self.head(x)


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(
            logits, targets)


def main():
    st = time.time_ns()

    train_dataset, test_dataset = get_jomon_kaen_dataset(
        # root="./src/data",
        # pottery_path="./src/pottery",
        root=r"D:\storage\jomon_kaen\data",
        pottery_path=r"D:\storage\jomon_kaen\pottery",
        split=0.1,
        preprocess=True,
        use_cache=True,
        mode=3,
        generate_qna=False,
        generate_voice=False,
        generate_pottery_dogu_voxel=False,
        generate_sanity_check=False,
        num_points=4096 * 4,
    )

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=2,
                                 pin_memory=True)

    # Initialize Model, Loss, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the new CorrectDeconver model
    # base_c=32 and depth=4 are common settings.
    model = CorrectDeconver(in_channels=3, out_channels=1, base_c=32,
                            depth=4).to(device)

    # Use a more suitable loss for segmentation
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training and Validation Loop
    EPOCHS = 5  # Reduced for quick demonstration
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader,
                          desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for pottery, voxel in train_pbar:
            # print(pottery.shape, voxel.shape)
            pottery, voxel = pottery.to(device), voxel.to(device)

            # Permute dimensions: (N, D, H, W, C) -> (N, C, D, H, W)
            pottery = pottery.permute(0, 4, 1, 2, 3)
            voxel = voxel.permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            logits = model(pottery)  # Model now outputs raw logits
            loss = criterion(logits, voxel)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(test_dataloader,
                        desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for pottery, voxel in val_pbar:
                pottery, voxel = pottery.to(device), voxel.to(device)
                pottery = pottery.permute(0, 4, 1, 2, 3)
                voxel = voxel.permute(0, 4, 1, 2, 3)

                logits = model(pottery)
                loss = criterion(logits, voxel)
                val_loss += loss.item()
                val_pbar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(test_dataloader)
        print(
            f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}"
        )

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_deconver_model.pth')
            print(
                f"Saved new best model with validation loss: {best_val_loss:.6f}"
            )

    et = time.time_ns()
    print(f"\nTotal Training Time: {(et - st) / 1e9:.2f} seconds")


if __name__ == "__main__":
    main()
