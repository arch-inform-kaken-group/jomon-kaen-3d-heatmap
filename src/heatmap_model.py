import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

# Pottery & Dogu assigned numbers
ASSIGNED_NUMBERS_DICT = {
    'AS0001': '1',
    'FH0008': '2',
    'IN0003': '3',
    'IN0008': '4',
    'IN0009': '5',
    'IN0017': '6',
    'IN0081': '7',
    'IN0104': '8',
    'IN0135': '9',
    'IN0148': '10',
    'IN0220': '11',
    'IN0228': '12',
    'IN0232': '13',
    'IN0239': '14',
    'IN0277': '15',
    'MY0001': '16',
    'MY0002': '17',
    'MY0004': '18',
    'MY0006': '19',
    'MY0007': '20',
    'ND0001': '21',
    'NM0001': '22',
    'NM0002': '23',
    'NM0009': '24',
    'NM0010': '25',
    'NM0014': '26',
    'NM0015': '27',
    'NM0017': '28',
    'NM0041': '29',
    'NM0049': '30',
    'NM0066': '31',
    'NM0070': '32',
    'NM0072': '33',
    'NM0073': '34',
    'NM0079': '35',
    'NM0080': '36',
    'NM0099': '37',
    'NM0106': '38',
    'NM0133': '39',
    'NM0135': '40',
    'NM0144': '41',
    'NM0154': '42',
    'NM0156': '43',
    'NM0159': '44',
    'NM0168': '45',
    'NM0173': '46',
    'NM0175': '47',
    'NM0189': '48',
    'NM0191': '49',
    'NM0206': '50',
    'SB0002': '51',
    'SB0004': '52',
    'SI0001': '53',
    'SJ0503': '54',
    'SJ0504': '55',
    'SK0001': '56',
    'SK0002': '57',
    'SK0003': '58',
    'SK0004': '59',
    'SK0005': '60',
    'SK0013': '61',
    'SS0001': '62',
    'TJ0004': '63',
    'TJ0005': '64',
    'TJ0010': '65',
    'TK0002': '66',
    'TK0048': '67',
    'TK0057': '68',
    'UD0001': '69',
    'UD0003': '70',
    'UD0005': '71',
    'UD0006': '72',
    'UD0011': '73',
    'UD0013': '74',
    'UD0014': '75',
    'UD0016': '76',
    'UD0023': '77',
    'UD0302': '78',
    'UD0304': '79',
    'UD0308': '80',
    'UD0318': '81',
    'UD0322': '82',
    'UD0411': '83',
    'UD0412': '84',
    'UK0001': '85',
}


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
        identity = torch.eye(self.k,
                             device=x.device).view(1, self.k * self.k).repeat(
                                 batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetRegressor(nn.Module):

    def __init__(self, num_points=8192, num_outputs=26):
        super(PointNetRegressor, self).__init__()
        self.num_points = num_points
        self.input_tnet = TNet(k=3)
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_tnet = TNet(k=64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)
        # self.reg_conv1 = nn.Conv1d(1088, 512, 1)
        # self.reg_bn1 = nn.BatchNorm1d(512)
        # self.reg_conv2 = nn.Conv1d(512, 256, 1)
        # self.reg_bn2 = nn.BatchNorm1d(256)
        # self.reg_conv3 = nn.Conv1d(256, 128, 1)
        # self.reg_bn3 = nn.BatchNorm1d(128)
        # self.reg_conv4 = nn.Conv1d(128, 1, 1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout1d(0.4)
        self.dense3 = nn.Linear(64, num_outputs)

    def forward(self, x):
        xyz = x[:, :3, :]
        rgb = x[:, 3:, :]
        input_transform = self.input_tnet(xyz)
        xyz_transformed = torch.bmm(xyz.transpose(1, 2),
                                    input_transform).transpose(1, 2)
        x_transformed = torch.cat([xyz_transformed, rgb], dim=1)
        x = F.relu(self.bn1(self.conv1(x_transformed)))
        x = F.relu(self.bn2(self.conv2(x)))
        feature_transform = self.feature_tnet(x)
        x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)
        point_features = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = self.flatten(x.max(dim=2)[0])
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.sigmoid(self.dense3(self.dropout(x)))

        return x


class PreprocessJomonKaenDataset(Dataset):

    def __init__(
        self,
        data,
        pottery_path,
        target_voxel_resolution=512,
        num_points=4096,
    ):
        super(PreprocessJomonKaenDataset, self).__init__()

        self.data = data
        self.target_voxel_resolution = target_voxel_resolution
        self.num_points = num_points

        self.headers = [
            'CODE', 'HAS_FLAME_LIKE_DECORATION', 'HAS_CROWN_LIKE_DECORATION',
            'HAS_HANDLES', 'HAS_CORD_MARKED_PATTERN', 'HAS_NAIL_ENGRAVING',
            'HAS_SPIRAL_PATTERN', 'HAS_FLAT_BASE', 'NUMBER_OF_PERTRUSIONS_0.0',
            'NUMBER_OF_PERTRUSIONS_1.0', 'NUMBER_OF_PERTRUSIONS_2.0',
            'NUMBER_OF_PERTRUSIONS_3.0', 'NUMBER_OF_PERTRUSIONS_4.0',
            'NUMBER_OF_PERTRUSIONS_6.0', 'NUMBER_OF_PERTRUSIONS_8.0',
            'SHAPE_TYPE_NAN', 'SHAPE_TYPE_三仏生式', 'SHAPE_TYPE_三十稲場式',
            'SHAPE_TYPE_千石原式', 'SHAPE_TYPE_南三十稲場式', 'SHAPE_TYPE_大木7b式',
            'SHAPE_TYPE_大木8a式', 'SHAPE_TYPE_新保・新崎式', 'SHAPE_TYPE_朝日式',
            'SHAPE_TYPE_栃倉式', 'SHAPE_TYPE_沖ノ原式', 'SHAPE_TYPE_馬高式'
        ]

        self.labels = pd.read_csv(pottery_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 1. Load the Open3D point cloud objects from their paths
        data_paths = self.data[index]
        # Assuming 'voxel_filename' is a variable holding the key for the heatmap file
        # and that self.data is a list of dictionaries.
        pottery_file = str(data_paths['processed_pottery_path'])
        pottery_name = os.path.basename(pottery_file)

        pottery_pcd = o3d.io.read_point_cloud(pottery_file)

        # Process Pottery Point Cloud
        pottery_points = np.asarray(pottery_pcd.points, dtype=np.float32)
        if pottery_pcd.has_colors():
            pottery_colors = np.asarray(pottery_pcd.colors, dtype=np.float32)
        else:
            # If no colors exist, use a neutral gray (0.5)
            pottery_colors = np.full_like(pottery_points,
                                          0.5,
                                          dtype=np.float32)

        # Sample a Fixed Number of Points
        # Since both clouds share the same structure, we can sample indices once.
        num_available_points = len(pottery_points)

        # Handle cases where one of the point clouds might be empty
        if num_available_points == 0:
            # Return zero tensors to prevent errors.
            pottery_xyz_rgb = torch.zeros((self.num_points, 6),
                                          dtype=torch.float32)
            return pottery_xyz_rgb

        # To be safe, ensure both arrays have the same length before sampling
        min_points = num_available_points

        # Choose indices to sample. Use replacement if not enough points are available.
        replace = num_available_points < self.num_points
        sample_indices = np.random.choice(min_points,
                                          self.num_points,
                                          replace=replace)

        # Use the same indices to sample from both the pottery and heatmap data
        sampled_pottery_points = pottery_points[sample_indices]
        sampled_pottery_colors = pottery_colors[sample_indices]

        # Combine and Convert to Tensors
        # Input tensor: concatenate XYZ and RGB features
        pottery_xyz_rgb = np.hstack(
            (sampled_pottery_points, sampled_pottery_colors))

        # Convert final numpy arrays to PyTorch tensors
        pottery_tensor = torch.from_numpy(pottery_xyz_rgb)
        target_tensor = torch.from_numpy(
            self.labels[self.labels['CODE'] == pottery_name][
                self.headers[1:]].values.astype(np.float32)).squeeze()

        return pottery_tensor, target_tensor


def get_pottery_id_list():
    return [f"{pid}({num})" for pid, num in ASSIGNED_NUMBERS_DICT.items()]


# 2. PYTORCH LIGHTNING DATAMODULE
class JomonKaenDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_root,
                 pottery_path,
                 batch_size=8,
                 num_workers=4,
                 num_points=8192):
        super().__init__()
        self.save_hyperparameters()

    def get_jomon_kaen_data(self, data_root, pottery_path, num_points,
                            test_groups):
        data = []
        for p in os.listdir(pottery_path):
            data_paths = {}
            data_paths['processed_pottery_path'] = f"{pottery_path}/{p}"
            data_paths['POTTERY'] = p
            data.append(data_paths)

        train_data = []
        test_data = []
        if len(test_groups) > 0:
            for data_paths in data:
                if (data_paths['POTTERY'] in test_groups):
                    test_data.append(data_paths)
                else:
                    train_data.append(data_paths)

        train_dataset = PreprocessJomonKaenDataset(
            data=train_data,
            pottery_path=data_root,
            num_points=num_points,
        )
        test_dataset = PreprocessJomonKaenDataset(
            data=test_data,
            pottery_path=data_root,
            num_points=num_points,
        )

        return train_dataset, test_dataset

    def setup(self, stage=None):
        common_params = {
            "data_root": self.hparams.data_root,
            "pottery_path": self.hparams.pottery_path,
            "num_points": self.hparams.num_points,
        }
        if stage == 'fit' or stage is None:
            self.train_dataset, self.val_dataset = self.get_jomon_kaen_data(
                test_groups=[
                    "IN0009(5).ply", "NM0049(30).ply", "UD0005(71).ply",
                    "ND0001(21).ply"
                ],
                **common_params)
        if stage == 'predict':
            _, self.predict_dataset = self.get_jomon_kaen_data(test_groups=[
                "IN0009(5).ply", "NM0049(30).ply", "UD0005(71).ply",
                "ND0001(21).ply"
            ],
                                                               **common_params)

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


# 3. PYTORCH LIGHTNING MODULE (MODIFIED)
class PointNetLightningModule(pl.LightningModule):

    def __init__(self,
                 num_points=4096,
                 learning_rate=1e-3,
                 lr_final=1e-5,
                 accuracy_threshold=0.05,
                 acc_thresh_final=0.01,
                 sparsity_weight=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetRegressor(num_points=self.hparams.num_points)
        self.criterion = nn.CrossEntropyLoss()
        # Store initial threshold for decay calculation, as hparams.accuracy_threshold will be modified
        self.initial_accuracy_threshold = self.hparams.accuracy_threshold

    def forward(self, x):
        outputs = self.model(x)
        return outputs

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
        self.log('acc_thresh',
                 self.hparams.accuracy_threshold,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        # The LR is logged automatically by the LearningRateMonitor callback

    def _shared_step(self, batch, batch_idx):
        """
        A shared step for training, validation, and testing.
        Includes L1 sparsity regularization.
        """
        # # 1. Unpack batch and perform the forward pass
        inputs, targets = batch
        inputs = inputs.permute(0, 2, 1)  # Change shape to (B, 6, N)
        outputs = self.model(inputs)
        print(outputs.shape, targets.shape)

        loss = self.criterion(outputs, targets)

        accuracy = (torch.argmax(outputs, dim=1, keepdim=True) == targets).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        print(batch)
        loss, accuracy = self._shared_step(batch, batch_idx)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_acc',
                 accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc',
                 accuracy,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and a learning rate scheduler that will decay the LR.
        """
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.learning_rate)

        # CosineAnnealingLR will smoothly decrease the LR from the initial value
        # down to `lr_final` over the course of `max_epochs`.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.lr_final)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":
                "epoch",  # The scheduler step is called after each epoch
                "frequency": 1,
            },
        }


# 4. MAIN EXECUTION BLOCK (MODIFIED)
if __name__ == '__main__':
    # Configuration
    NUM_POINTS = 4096
    BATCH_SIZE = 4
    MAX_EPOCHS = 200
    NUM_GPUS = torch.cuda.device_count()
    # Use half of the available CPU cores for data loading to avoid system overload
    NUM_WORKERS = int(os.cpu_count() / 2) if os.cpu_count() else 0
    OUTPUT_DIR = "outputs"

    # Hyperparameters for decay
    LEARNING_RATE_INITIAL = 1e-3
    LEARNING_RATE_FINAL = 1e-5  # The LR will decay to this value
    ACC_THRESH_INITIAL = 0.01
    ACC_THRESH_FINAL = 0.01  # The accuracy threshold will decay to this value

    torch.set_float32_matmul_precision('high')
    print(
        f"Found {NUM_GPUS} GPUs and {os.cpu_count()} CPUs. Using {NUM_WORKERS} workers."
    )

    # Initialize Data and Model Modules
    datamodule = JomonKaenDataModule(
        # data_root=r"D:\storage\jomon_kaen\data_my",
        data_root=r"src\jomon_kaen_dataset\DS_Labels_Cleaned.csv",
        pottery_path=r"src/jomon_kaen_dataset/processed/voxel_pottery",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        num_points=NUM_POINTS)

    model = PointNetLightningModule(num_points=NUM_POINTS,
                                    learning_rate=LEARNING_RATE_INITIAL,
                                    lr_final=LEARNING_RATE_FINAL,
                                    accuracy_threshold=ACC_THRESH_INITIAL,
                                    acc_thresh_final=ACC_THRESH_FINAL)

    import torchinfo
    torchinfo.summary(model)

    # Configure Callbacks
    # 1. Saves the single best model based on validation loss
    best_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='pointnet-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min')
    # 2. Saves a checkpoint every 10 epochs, keeping all of them
    periodic_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10,
        dirpath='checkpoints/',
        filename='pointnet-epoch={epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,  # -1 means save all models that meet the criteria
        save_on_train_epoch_end=True)
    # 3. Stops training if validation loss doesn't improve for a number of epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=
        20,  # Increased patience as LR scheduling might cause temporary plateaus
        verbose=True,
        mode='min')
    # 4. Logs the learning rate to the logger (e.g., TensorBoard)
    lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')

    # Initialize Trainer
    trainer = pl.Trainer(accelerator='gpu' if NUM_GPUS > 0 else 'cpu',
                         devices=NUM_GPUS if NUM_GPUS > 0 else 1,
                         precision="16-mixed" if NUM_GPUS > 0 else "32-true",
                         max_epochs=MAX_EPOCHS,
                         callbacks=[
                             best_checkpoint_callback,
                             periodic_checkpoint_callback, lr_monitor_callback
                         ],
                         log_every_n_steps=10)

    # Start Training
    print("\nStarting Training")
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished")

    # INFERENCE SECTION
    print("\nStarting Inference")

    # Use the path from the callback that saves the single best model
    best_model_path = best_checkpoint_callback.best_model_path
    if not best_model_path or not os.path.exists(best_model_path):
        print(f"Error: Could not find best model checkpoint.")
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
                print(f"Processing batch {i+1}/{len(predict_loader)}")

                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')

                outputs = model(inputs.permute(0, 2, 1))
                predicted_intensities = outputs.permute(0, 2, 1).cpu().numpy()

                for j in range(predicted_intensities.shape[0]):
                    xyz = inputs[j, :, :3].cpu().numpy()

                    # Process and Save Prediction
                    pred_intensities = predicted_intensities[j]
                    # Add a small epsilon to prevent division by zero if all intensities are the same
                    norm_pred = (pred_intensities - np.min(pred_intensities)
                                 ) / (np.max(pred_intensities) -
                                      np.min(pred_intensities) + 1e-8)
                    colors_pred = cmap(norm_pred.squeeze())[:, :3]

                    pcd_pred = o3d.geometry.PointCloud()
                    pcd_pred.points = o3d.utility.Vector3dVector(xyz)
                    pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

                    pred_filename = os.path.join(
                        output_dir_pred, f"prediction_batch{i}_item{j}.ply")
                    o3d.io.write_point_cloud(pred_filename, pcd_pred)

                    # Process and Save Ground Truth
                    gt_intensities = targets[j].cpu().numpy()
                    norm_gt = (gt_intensities - np.min(gt_intensities)) / (
                        np.max(gt_intensities) - np.min(gt_intensities) + 1e-8)
                    colors_gt = cmap(norm_gt.squeeze())[:, :3]

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(xyz)
                    pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

                    gt_filename = os.path.join(
                        output_dir_gt, f"groundtruth_batch{i}_item{j}.ply")
                    o3d.io.write_point_cloud(gt_filename, pcd_gt)

            print(
                f"\nSaved predictions and ground truths for {len(predict_loader.dataset)} items."
            )
        print("\nInference complete.")
