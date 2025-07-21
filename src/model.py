import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Assuming the dataset script is in dataset/dataset.py
from dataset.dataset import get_jomon_kaen_dataset


class DoubleConv3D(nn.Module):
    """(Convolution => [Batch Norm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='trilinear',
                                  align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels,
                                     in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2
        ])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Deconver(nn.Module):

    def __init__(self, n_channels_in=3, n_channels_out=1, bilinear=True):
        super(Deconver, self).__init__()
        self.inc = DoubleConv3D(n_channels_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


def main():
    st = time.time_ns()

    # --- 1. Load Datasets ---
    # Note: For a real run, you might remove pottery_ids to use all data
    train_dataset, test_dataset = get_jomon_kaen_dataset(
        root="./src/data",
        pottery_path="./src/pottery",
        split=0.25,
        preprocess=True,
        use_cache=True,
        pottery_ids=["IN0017"],
        mode=3,
        generate_qna=False,
        generate_voice=False,
        generate_pottery_dogu_voxel=False,
        generate_sanity_check=False)

    # --- 2. Create DataLoaders ---
    # Warning: 3D data is large. Start with batch_size=1 and increase if memory allows.
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

    # --- 3. Initialize Model, Loss, and Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = Deconver(n_channels_in=3, n_channels_out=1).to(device)
    criterion = nn.MSELoss(
    )  # Mean Squared Error is good for intensity regression
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- 4. Training and Validation Loop ---
    EPOCHS = 20
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_dataloader,
                          desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for pottery, voxel in train_pbar:
            pottery, voxel = pottery.to(device), voxel.to(device)

            # Permute dimensions: (N, D, H, W, C) -> (N, C, D, H, W)
            pottery = pottery.permute(0, 4, 1, 2, 3)
            voxel = voxel.permute(0, 4, 1, 2, 3)

            optimizer.zero_grad()
            outputs = model(pottery)
            loss = criterion(outputs, voxel)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(test_dataloader,
                        desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for pottery, voxel in val_pbar:
                pottery, voxel = pottery.to(device), voxel.to(device)
                pottery = pottery.permute(0, 4, 1, 2, 3)
                voxel = voxel.permute(0, 4, 1, 2, 3)

                outputs = model(pottery)
                loss = criterion(outputs, voxel)
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
