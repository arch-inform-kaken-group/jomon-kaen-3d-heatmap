import os
import torch
import open3d as o3d
from dataset.utils import filter_data_on_condition
from torch.utils.data import DataLoader
from full_model_v4 import (
    ExtendedVoxelDataModule,
    EMOTION_ORDER,
    DEFAULT_QNA_ANSWER_COLOR_MAP,
    voxel_grid_to_point_cloud
)

# Config
RAW_DATA_DIR = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
MESH_DIR = r"D:\storage\jomon_kaen\pottery"
VOXEL_RESOLUTION = 80
MAX_COMMENT_LEN = 64
BATCH_SIZE = 1
NUM_WORKERS = 0

# Load data
all_data_paths, _ = filter_data_on_condition(
    root=RAW_DATA_DIR,
    pottery_path=MESH_DIR,
    preprocess=True,
    use_cache=True,
    mode=0,
    target_voxel_resolution=VOXEL_RESOLUTION,
    min_emotion_count=1,
    min_qa_size=1
)
print(f"Loaded {len(all_data_paths)} samples.")

# Setup datamodule
datamodule = ExtendedVoxelDataModule(
    all_data_paths=all_data_paths,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    voxel_resolution=VOXEL_RESOLUTION,
    max_comment_len=MAX_COMMENT_LEN,
)
datamodule.setup(stage='fit')

# Token decoder
def decode_tokens(token_tensor, tokenizer):
    words = []
    for idx in token_tensor.cpu().numpy():
        if idx == 2:  # <eos>
            break
        if idx in [0, 1]:  # <pad>, <sos>
            continue
        word = tokenizer.idx_to_word.get(idx, '<unk>')
        words.append(word)
    return ' '.join(words)

# Start inspection
N = 2
loader = DataLoader(datamodule.train_dataset, batch_size=1, shuffle=False)
tokenizer = datamodule.tokenizer

for i, (pottery_voxel, (emo_voxels, heatmap_voxel, comment_tokens)) in enumerate(loader):
    if i >= N:
        break

    print(f"Sample {i}")
    print(f"Pottery shape: {pottery_voxel.shape}")
    print(f"Emotion shape: {emo_voxels.shape}")
    print(f"Heatmap shape: {heatmap_voxel.shape}")
    decoded_comment = decode_tokens(comment_tokens.squeeze(), tokenizer)
    print(f"Comment: \"{decoded_comment}\"")

    # Visualize input pottery
    print("Showing input pottery")
    pcd_input = voxel_grid_to_point_cloud(pottery_voxel[0], reference_pcd_path=None)
    o3d.visualization.draw_geometries([pcd_input], window_name="Input Pottery")

    # Visualize each active emotion
    for j, emo_name in enumerate(EMOTION_ORDER):
        if emo_voxels[0, j].sum().item() > 0:
            color_rgb = DEFAULT_QNA_ANSWER_COLOR_MAP[emo_name]['rgb']
            print(f"Showing emotion: {emo_name}")
            pcd_emo = voxel_grid_to_point_cloud(
                emo_voxels[0, j],
                intensity_threshold=-1.0,  # show all voxels inside pottery
                mask_tensor=(pottery_voxel[0].sum(dim=0) > 0).float(),
                fixed_color_rgb=color_rgb
            )
            o3d.visualization.draw_geometries([pcd_emo], window_name=f"Emotion: {emo_name}")

    # Visualize heatmap
    print("Showing eye gaze heatmap")
    pcd_heat = voxel_grid_to_point_cloud(
        heatmap_voxel[0],
        intensity_threshold=-1.0,
        mask_tensor=(pottery_voxel[0].sum(dim=0) > 0).float(),
        colormap_name='jet'
    )
    o3d.visualization.draw_geometries([pcd_heat], window_name="Eye Gaze Heatmap")