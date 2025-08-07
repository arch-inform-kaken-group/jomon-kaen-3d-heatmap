import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# The get_jomon_kaen_dataset function is expected to be in the dataset/dataset.py file.
# Make sure the path is correct for your project structure.
from dataset.dataset import get_jomon_kaen_dataset


# 1. MODEL DEFINITION (The nn.Module remains the same)
class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x

class PointNetRegressor(nn.Module):
    def __init__(self, num_points=8192):
        super(PointNetRegressor, self).__init__()
        self.num_points = num_points
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(6, 64, 1); self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1); self.bn2 = nn.BatchNorm1d(64)
        self.feature_tnet = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1); self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1); self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1); self.bn5 = nn.BatchNorm1d(1024)
        self.reg_conv1 = nn.Conv1d(1088, 512, 1); self.reg_bn1 = nn.BatchNorm1d(512)
        self.reg_conv2 = nn.Conv1d(512, 256, 1); self.reg_bn2 = nn.BatchNorm1d(256)
        self.reg_conv3 = nn.Conv1d(256, 128, 1); self.reg_bn3 = nn.BatchNorm1d(128)
        self.reg_conv4 = nn.Conv1d(128, 1, 1)
        # self.conv5 = nn.Conv1d(128, 512, 1); self.bn5 = nn.BatchNorm1d(512)
        # self.reg_conv1 = nn.Conv1d(576, 256, 1); self.reg_bn1 = nn.BatchNorm1d(256)
        # self.reg_conv2 = nn.Conv1d(256, 128, 1); self.reg_bn2 = nn.BatchNorm1d(128)
        # self.reg_conv3 = nn.Conv1d(128, 64, 1); self.reg_bn3 = nn.BatchNorm1d(64)
        # self.reg_conv4 = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        xyz = x[:, :3, :]
        rgb = x[:, 3:, :]
        input_transform = self.input_tnet(xyz)
        xyz_transformed = torch.bmm(xyz.transpose(1, 2), input_transform).transpose(1, 2)
        x_transformed = torch.cat([xyz_transformed, rgb], dim=1)
        x = F.relu(self.bn1(self.conv1(x_transformed)))
        x = F.relu(self.bn2(self.conv2(x)))
        feature_transform = self.feature_tnet(x)
        x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)
        point_features = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature_repeated = global_feature.repeat(1, 1, self.num_points)
        concat_features = torch.cat([point_features, global_feature_repeated], dim=1)
        x = F.relu(self.reg_bn1(self.reg_conv1(concat_features)))
        x = F.relu(self.reg_bn2(self.reg_conv2(x)))
        x = F.relu(self.reg_bn3(self.reg_conv3(x)))
        x = self.reg_conv4(x)
        return x

# 2. PYTORCH LIGHTNING DATAMODULE
class JomonKaenDataModule(pl.LightningDataModule):
    def __init__(self, data_root, pottery_path, batch_size=8, num_workers=4, num_points=8192):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        common_params = {
            "root": self.hparams.data_root,
            "pottery_path": self.hparams.pottery_path,
            "num_points": self.hparams.num_points,
            "preprocess": True, "target_voxel_resolution": 512, "mode": 3,
        }
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = get_jomon_kaen_dataset(
                split=0.1,
                use_cache=True,
                **common_params
            )
        if stage == 'predict':
            _ , self.predict_dataset = get_jomon_kaen_dataset(
                split=0.1,
                use_cache=True,
                **common_params
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size * 2, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

# 3. PYTORCH LIGHTNING MODULE (MODIFIED)
class PointNetLightningModule(pl.LightningModule):
    def __init__(self, num_points=4096, learning_rate=1e-3, lr_final=1e-5, accuracy_threshold=0.05, acc_thresh_final=0.01, sparsity_weight=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetRegressor(num_points=self.hparams.num_points)
        self.criterion = nn.MSELoss()
        # Store initial threshold for decay calculation, as hparams.accuracy_threshold will be modified
        self.initial_accuracy_threshold = self.hparams.accuracy_threshold

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        """
        Called at the beginning of each training epoch.
        We use this hook to decay the accuracy_threshold and log it.
        The learning rate is handled by the scheduler.
        """
        # Calculate the new accuracy threshold based on a linear decay.
        progress = self.current_epoch / self.trainer.max_epochs
        # new_thresh = self.initial_accuracy_threshold - (self.initial_accuracy_threshold - self.hparams.acc_thresh_final) * progress
        
        # Update the hyperparameter, ensuring it doesn't fall below the final value as a safeguard.
        # self.hparams.accuracy_threshold = max(self.hparams.acc_thresh_final, new_thresh)
        
        # Log the current values to see them in the progress bar and logs
        self.log('acc_thresh', self.hparams.accuracy_threshold, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # The LR is logged automatically by the LearningRateMonitor callback

    # def _shared_step(self, batch, batch_idx):
    #     inputs, targets = batch
    #     inputs = inputs.permute(0, 2, 1) # B, C, N
    #     outputs = self(inputs)           # B, 1, N
    #     outputs = outputs.permute(0, 2, 1) # B, N, 1
        
    #     loss = self.criterion(outputs, targets)
        
    #     with torch.no_grad():
    #         # Accuracy is calculated based on the *current* threshold, which decays over epochs
    #         correct = torch.sum(torch.abs(outputs - targets) < self.hparams.accuracy_threshold)
    #         accuracy = correct / targets.numel()
            
    #     return loss, accuracy

    def _shared_step(self, batch, batch_idx):
        """
        A shared step for training, validation, and testing.
        Includes L1 sparsity regularization.
        """
        # 1. Unpack batch and perform the forward pass
        inputs, targets = batch
        inputs = inputs.permute(0, 2, 1)      # B, C, N
        outputs = self(inputs)                # B, 1, N
        outputs = outputs.permute(0, 2, 1)    # B, N, 1

        # 2. Calculate the primary reconstruction loss (MSE)
        mse_loss = self.criterion(outputs, targets)

        # 3. Calculate the L1 sparsity penalty on the outputs
        # We use .mean() to make it independent of batch size.
        l1_penalty = torch.mean(torch.abs(outputs))

        # 4. Calculate the total loss by adding the weighted penalty
        # self.hparams.sparsity_weight is the lambda (λ) hyperparameter
        total_loss = mse_loss + self.hparams.sparsity_weight * l1_penalty
        
        # 5. Calculate accuracy for monitoring (within no_grad context)
        with torch.no_grad():
            # Accuracy is calculated based on the *current* threshold
            correct = torch.sum(torch.abs(outputs - targets) < self.hparams.accuracy_threshold)
            accuracy = correct / targets.numel()

            # Best Practice: Log individual loss components for easier debugging
            self.log('mse_loss', mse_loss)
            self.log('weighted_l1_penalty', self.hparams.sparsity_weight * l1_penalty)
            
        # 6. Return the combined loss and the accuracy
        return total_loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and a learning rate scheduler that will decay the LR.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        # CosineAnnealingLR will smoothly decrease the LR from the initial value
        # down to `lr_final` over the course of `max_epochs`.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr_final
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # The scheduler step is called after each epoch
                "frequency": 1,
            },
        }

# 4. MAIN EXECUTION BLOCK (MODIFIED)
if __name__ == '__main__':
    # --- Configuration ---
    NUM_POINTS = 4096 * 8
    BATCH_SIZE = 8
    MAX_EPOCHS = 200
    NUM_GPUS = torch.cuda.device_count()
    # Use half of the available CPU cores for data loading to avoid system overload
    NUM_WORKERS = int(os.cpu_count() / 2) if os.cpu_count() else 0
    OUTPUT_DIR = "outputs"
    
    # --- Hyperparameters for decay ---
    LEARNING_RATE_INITIAL = 1e-3
    LEARNING_RATE_FINAL = 1e-5    # The LR will decay to this value
    ACC_THRESH_INITIAL = 0.01
    ACC_THRESH_FINAL = 0.01       # The accuracy threshold will decay to this value

    torch.set_float32_matmul_precision('high')
    print(f"Found {NUM_GPUS} GPUs and {os.cpu_count()} CPUs. Using {NUM_WORKERS} workers.")

    # --- Initialize Data and Model Modules ---
    datamodule = JomonKaenDataModule(
        data_root=r"D:\storage\jomon_kaen\data", # Adjusted to your original path
        pottery_path=r"D:\storage\jomon_kaen\pottery", # Adjusted to your original path
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_points=NUM_POINTS
    )

    model = PointNetLightningModule(
        num_points=NUM_POINTS,
        learning_rate=LEARNING_RATE_INITIAL,
        lr_final=LEARNING_RATE_FINAL,
        accuracy_threshold=ACC_THRESH_INITIAL,
        acc_thresh_final=ACC_THRESH_FINAL
    )

    # --- Configure Callbacks ---
    # 1. Saves the single best model based on validation loss
    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath='checkpoints/', 
        filename='pointnet-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )
    # 2. Saves a checkpoint every 10 epochs, keeping all of them
    periodic_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        dirpath='checkpoints/',
        filename='pointnet-epoch={epoch:02d}-{val_loss:.4f}',
        save_top_k=-1, # -1 means save all models that meet the criteria
        save_on_train_epoch_end=True
    )
    # 3. Stops training if validation loss doesn't improve for a number of epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=20, # Increased patience as LR scheduling might cause temporary plateaus
        verbose=True, 
        mode='min'
    )
    # 4. Logs the learning rate to the logger (e.g., TensorBoard)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # --- Initialize Trainer ---
    trainer = pl.Trainer(
        accelerator='gpu' if NUM_GPUS > 0 else 'cpu',
        devices=NUM_GPUS if NUM_GPUS > 0 else 1,
        precision="16-mixed" if NUM_GPUS > 0 else "32-true",
        max_epochs=MAX_EPOCHS,
        callbacks=[best_checkpoint_callback, periodic_checkpoint_callback, lr_monitor_callback],
        log_every_n_steps=10
    )

    # --- Start Training ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule=datamodule)
    print("--- Training Finished ---")

    # --- INFERENCE SECTION ---
    print("\n--- Starting Inference ---")
    
    # Use the path from the callback that saves the single best model
    best_model_path = best_checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Error: Could not find best model checkpoint. Was training run?")
    else:
        print(f"Loading best model from: {best_model_path}")
        model = PointNetLightningModule.load_from_checkpoint(best_model_path)
        
        datamodule.setup('predict')
        predict_loader = datamodule.predict_dataloader()
        
        output_dir_pred = os.path.join(OUTPUT_DIR, "predictions")
        output_dir_gt = os.path.join(OUTPUT_DIR, "ground_truth")
        os.makedirs(output_dir_pred, exist_ok=True)
        os.makedirs(output_dir_gt, exist_ok=True)
        print(f"Prediction outputs will be saved to: {output_dir_pred}")
        print(f"Ground truth outputs will be saved to: {output_dir_gt}")
        
        cmap = plt.get_cmap('jet')
        
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')
        
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(predict_loader):
                print(f"Processing batch {i+1}/{len(predict_loader)}...")
                
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                
                outputs = model(inputs.permute(0, 2, 1))
                predicted_intensities = outputs.permute(0, 2, 1).cpu().numpy()
                
                for j in range(predicted_intensities.shape[0]):
                    xyz = inputs[j, :, :3].cpu().numpy()
                    
                    # --- Process and Save Prediction ---
                    pred_intensities = predicted_intensities[j]
                    # Add a small epsilon to prevent division by zero if all intensities are the same
                    norm_pred = (pred_intensities - np.min(pred_intensities)) / (np.max(pred_intensities) - np.min(pred_intensities) + 1e-8)
                    colors_pred = cmap(norm_pred.squeeze())[:, :3]
                    
                    pcd_pred = o3d.geometry.PointCloud()
                    pcd_pred.points = o3d.utility.Vector3dVector(xyz)
                    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)
                    
                    pred_filename = os.path.join(output_dir_pred, f"prediction_batch{i}_item{j}.ply")
                    o3d.io.write_point_cloud(pred_filename, pcd_pred)
                    
                    # --- Process and Save Ground Truth ---
                    gt_intensities = targets[j].cpu().numpy()
                    norm_gt = (gt_intensities - np.min(gt_intensities)) / (np.max(gt_intensities) - np.min(gt_intensities) + 1e-8)
                    colors_gt = cmap(norm_gt.squeeze())[:, :3]

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(xyz)
                    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

                    gt_filename = os.path.join(output_dir_gt, f"groundtruth_batch{i}_item{j}.ply")
                    o3d.io.write_point_cloud(gt_filename, pcd_gt)

            print(f"\nSaved predictions and ground truths for {len(predict_loader.dataset)} items.")
        print("\nInference complete.")
