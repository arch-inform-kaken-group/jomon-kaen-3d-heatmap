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
import torchinfo

from dataset.dataset import get_jomon_kaen_dataset


# 1. MODEL DEFINITION (MODIFIED FOR MULTI-TASK LEARNING)
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
        identity = torch.eye(self.k,
                             device=x.device).view(1, self.k * self.k).repeat(
                                 batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetRegressor(nn.Module):
    # Updated to accept in_channels and out_channels for multi-task
    def __init__(self, num_points=8192, in_channels=6, out_channels=6):
        super(PointNetRegressor, self).__init__()
        self.num_points = num_points
        self.input_tnet = TNet(k=3)
        # Input channels is 6 (XYZ + RGB)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_tnet = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
        self.reg_conv1 = nn.Conv1d(1088, 512, 1)
        self.reg_bn1 = nn.BatchNorm1d(512)
        self.reg_conv2 = nn.Conv1d(512, 256, 1)
        self.reg_bn2 = nn.BatchNorm1d(256)
        self.reg_conv3 = nn.Conv1d(256, 128, 1)
        self.reg_bn3 = nn.BatchNorm1d(128)
        # Output channels is 6 (1 for Heatmap + 5 for Emotions)
        self.reg_conv4 = nn.Conv1d(128, out_channels, 1)

    def forward(self, x):
        xyz = x[:, :3, :]
        rgb_features = x[:, 3:, :]
        input_transform = self.input_tnet(xyz)
        xyz_transformed = torch.bmm(xyz.transpose(1, 2),
                                    input_transform).transpose(1, 2)
        x_transformed = torch.cat([xyz_transformed, rgb_features], dim=1)
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
        concat_features = torch.cat([point_features, global_feature_repeated],
                                    dim=1)
        x = F.relu(self.reg_bn1(self.reg_conv1(concat_features)))
        x = F.relu(self.reg_bn2(self.reg_conv2(x)))
        x = F.relu(self.reg_bn3(self.reg_conv3(x)))
        x = self.reg_conv4(x)
        return x


# 2. PYTORCH LIGHTNING DATAMODULE (Set to mode 1)
class JomonKaenDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root,
                 pottery_path,
                 batch_size=8,
                 num_workers=4,
                 num_points=8192):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        common_params = {
            "root": self.hparams.data_root,
            "pottery_path": self.hparams.pottery_path,
            "num_points": self.hparams.num_points,
            "preprocess": True,
            "target_voxel_resolution": 512,
            "mode": 1,
            "limit": 1000,
            "min_emotion_count": 2,
            "min_qa_size": 10,
        }
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = get_jomon_kaen_dataset(
                test_groups=["G17"], use_cache=True, **common_params)
        if stage == 'predict':
            _, self.predict_dataset = get_jomon_kaen_dataset(
                test_groups=["G17"], use_cache=True, **common_params)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size * 2,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True)


# 3. PYTORCH LIGHTNING MODULE (MODIFIED with Hybrid Loss)
class PointNetLightningModule(pl.LightningModule):

    def __init__(self, num_points=4096, learning_rate=1e-3, lr_final=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetRegressor(num_points=self.hparams.num_points,
                                       in_channels=6,
                                       out_channels=6)
        # Define separate loss functions for each task
        self.heatmap_criterion = nn.MSELoss()
        self.emotion_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        inputs, targets_combined = batch
        inputs = inputs.permute(0, 2, 1)  # B, 6, N
        targets_combined = targets_combined.permute(0, 2, 1)  # B, 6, N

        # Get combined model output (B, 6, N)
        outputs_combined = self(inputs)

        # Split predictions and targets for each task
        heatmap_pred = outputs_combined[:, 0:1, :]
        emotion_logits_pred = outputs_combined[:, 1:, :]

        heatmap_target = targets_combined[:, 0:1, :]
        emotion_target = targets_combined[:, 1:, :]

        # Calculate separate losses
        loss_heatmap = self.heatmap_criterion(heatmap_pred, heatmap_target)
        loss_emotion = self.emotion_criterion(emotion_logits_pred,
                                              emotion_target)

        # Combine losses (can be weighted, but simple sum is a good start)
        total_loss = loss_heatmap + loss_emotion

        with torch.no_grad():
            # Calculate accuracy for emotion classification
            emotion_probs = torch.sigmoid(emotion_logits_pred)
            emotion_preds = torch.round(emotion_probs)
            emotion_correct = torch.sum(emotion_preds == emotion_target)
            emotion_acc = emotion_correct / emotion_target.numel()
            # Calculate Mean Absolute Error for heatmap regression
            heatmap_mae = F.l1_loss(heatmap_pred, heatmap_target)

        # Log individual losses and metrics for better monitoring
        self.log(f'{self.current_stage}_heatmap_loss',
                 loss_heatmap,
                 on_step=False,
                 on_epoch=True)
        self.log(f'{self.current_stage}_emotion_loss',
                 loss_emotion,
                 on_step=False,
                 on_epoch=True)
        self.log(f'{self.current_stage}_emotion_acc',
                 emotion_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.log(f'{self.current_stage}_heatmap_mae',
                 heatmap_mae,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        self.current_stage = 'train'
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.current_stage = 'val'
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr_final)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


# 4. MAIN EXECUTION BLOCK (MODIFIED)
if __name__ == '__main__':
    NUM_POINTS, BATCH_SIZE, MAX_EPOCHS = 4096 * 4, 4, 200
    NUM_GPUS = torch.cuda.device_count()
    NUM_WORKERS = int(os.cpu_count() / 4) if os.cpu_count() else 0
    OUTPUT_DIR = "outputs_multitask"
    LEARNING_RATE_INITIAL, LEARNING_RATE_FINAL = 1e-3, 1e-3

    torch.set_float32_matmul_precision('high')
    datamodule = JomonKaenDataModule(
        data_root=r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan",
        pottery_path=r"D:\storage\jomon_kaen\pottery",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_points=NUM_POINTS)

    model = PointNetLightningModule(num_points=NUM_POINTS,
                                    learning_rate=LEARNING_RATE_INITIAL,
                                    lr_final=LEARNING_RATE_FINAL)

    torchinfo.summary(model, input_size=(BATCH_SIZE, 6, NUM_POINTS))

    trainer = pl.Trainer(accelerator='gpu',
                         devices=NUM_GPUS,
                         precision="16-mixed",
                         max_epochs=MAX_EPOCHS,
                         callbacks=[
                             ModelCheckpoint(monitor='val_loss',
                                             dirpath='checkpoints_multitask/',
                                             filename='best-{epoch:02d}',
                                             save_top_k=1,
                                             mode='min'),
                             LearningRateMonitor(logging_interval='epoch')
                         ])

    print("\nStarting Multi-Task Training (Mode 1)")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished")

    # INFERENCE SECTION (MODIFIED for multi-task output)
    print("\nStarting Multi-Task Inference")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print("Error: Could not find best model checkpoint.")
    else:
        print(f"Loading best model from: {best_model_path}")
        model = PointNetLightningModule.load_from_checkpoint(best_model_path)
        datamodule.setup('predict')
        predict_loader = datamodule.predict_dataloader()

        output_dir_pred = os.path.join(OUTPUT_DIR, "predictions")
        output_dir_gt = os.path.join(OUTPUT_DIR, "ground_truth")
        EMOTION_LABELS = [
            "interesting", "beautiful", "strange", "creepy", "nothing"
        ]

        # Create subdirectories for heatmap and each emotion
        os.makedirs(os.path.join(output_dir_pred, "heatmap"), exist_ok=True)
        os.makedirs(os.path.join(output_dir_gt, "heatmap"), exist_ok=True)
        for label in EMOTION_LABELS:
            os.makedirs(os.path.join(output_dir_pred, label), exist_ok=True)
            os.makedirs(os.path.join(output_dir_gt, label), exist_ok=True)

        cmap = plt.get_cmap('jet')
        model.eval()
        if torch.cuda.is_available(): model.to('cuda')

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(predict_loader):
                print(f"Processing batch {i+1}/{len(predict_loader)}")
                if torch.cuda.is_available(): inputs = inputs.to('cuda')

                outputs = model(inputs.permute(0, 2, 1)).cpu()  # B, 6, N

                for j in range(
                        outputs.shape[0]):  # Loop through items in batch
                    xyz = inputs[j, :, :3].cpu().numpy()  # N, 3

                    # --- Process and Save Heatmap ---
                    pred_hm = outputs[j, 0, :].numpy()
                    gt_hm = targets[j, :, 0].numpy()
                    norm_pred_hm = (pred_hm - pred_hm.min()) / (
                        pred_hm.max() - pred_hm.min() + 1e-8)
                    norm_gt_hm = (gt_hm - gt_hm.min()) / (gt_hm.max() -
                                                          gt_hm.min() + 1e-8)

                    pcd_pred_hm = o3d.geometry.PointCloud()
                    pcd_pred_hm.points = o3d.utility.Vector3dVector(xyz)
                    pcd_pred_hm.colors = o3d.utility.Vector3dVector(
                        cmap(norm_pred_hm)[:, :3])
                    o3d.io.write_point_cloud(
                        os.path.join(output_dir_pred, "heatmap",
                                     f"b{i}_i{j}.ply"), pcd_pred_hm)

                    pcd_gt_hm = o3d.geometry.PointCloud()
                    pcd_gt_hm.points = o3d.utility.Vector3dVector(xyz)
                    pcd_gt_hm.colors = o3d.utility.Vector3dVector(
                        cmap(norm_gt_hm)[:, :3])
                    o3d.io.write_point_cloud(
                        os.path.join(output_dir_gt, "heatmap",
                                     f"b{i}_i{j}.ply"), pcd_gt_hm)

                    # --- Process and Save Emotions ---
                    pred_emotions = torch.sigmoid(
                        outputs[j, 1:, :]).numpy()  # 5, N
                    gt_emotions = targets[j, :, 1:].numpy().T  # 5, N

                    for k in range(len(EMOTION_LABELS)):
                        emotion_label = EMOTION_LABELS[k]
                        colors_pred = cmap(pred_emotions[k, :])[:, :3]
                        colors_gt = np.full((xyz.shape[0], 3), [0.8, 0.8, 0.8])
                        colors_gt[gt_emotions[k, :] == 1] = [1.0, 1.0, 0.0]

                        pcd_pred_e = o3d.geometry.PointCloud()
                        pcd_pred_e.points = o3d.utility.Vector3dVector(xyz)
                        pcd_pred_e.colors = o3d.utility.Vector3dVector(
                            colors_pred)
                        o3d.io.write_point_cloud(
                            os.path.join(output_dir_pred, emotion_label,
                                         f"b{i}_i{j}.ply"), pcd_pred_e)

                        pcd_gt_e = o3d.geometry.PointCloud()
                        pcd_gt_e.points = o3d.utility.Vector3dVector(xyz)
                        pcd_gt_e.colors = o3d.utility.Vector3dVector(colors_gt)
                        o3d.io.write_point_cloud(
                            os.path.join(output_dir_gt, emotion_label,
                                         f"b{i}_i{j}.ply"), pcd_gt_e)

        print("\nInference complete.")
