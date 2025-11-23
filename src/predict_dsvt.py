import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dataset.utils import filter_data_on_condition, DEFAULT_QNA_ANSWER_COLOR_MAP
from dataset.voxel_dataset import ExtendedVoxelDataset
from model.helper import voxel_grid_to_point_cloud
from dsvt_3dcnn_gpt_long import DSVTFullModel, SimpleTokenizer

# --- Config matching training ---
EMOTION_ORDER = ["面白い・気になる形だ", "美しい・芸術的だ", "不思議・意味不明", "不気味・不安・怖い", "何も感じない"]
VOXEL_RESOLUTION = 80
MAX_COMMENT_LEN = 150
RAW_DATA_DIR = r"D:\storage\jomon_kaen\jomon_kaen_dataset\japan"
MESH_DIR = r"D:\storage\jomon_kaen\pottery"

# --- Tokenizer builder (same as before) ---
def build_tokenizer_from_data(all_data_paths, max_comment_len=MAX_COMMENT_LEN):
    all_comments = []
    for item in all_data_paths:
        comment_path = item.get('TRANSCRIPT', '')
        if os.path.exists(comment_path):
            try:
                with open(comment_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:
                        all_comments.append(text)
            except Exception as e:
                print(f"WARNING: Failed to read {comment_path}: {e}")
    tokenizer = SimpleTokenizer(max_len=max_comment_len)
    tokenizer.build_vocab(all_comments)
    return tokenizer

# --- Main Prediction Function ---
def predict_and_save_by_pottery_id(
    model_checkpoint_path: str,
    raw_data_dir: str,
    pottery_mesh_dir: str,
    save_root_dir: str,
    voxel_resolution: int = VOXEL_RESOLUTION,
    max_comment_len: int = MAX_COMMENT_LEN,
    emotion_order = EMOTION_ORDER,
    batch_size: int = 4,
    num_workers: int = 2
):
    os.makedirs(save_root_dir, exist_ok=True)

    # 1. Load data paths
    print("Filtering data paths...")
    all_data_paths, _ = filter_data_on_condition(
        root=raw_data_dir,
        pottery_path=pottery_mesh_dir,
        preprocess=True,
        use_cache=True,
        mode=0,
        target_voxel_resolution=voxel_resolution,
        min_emotion_count=1,
        min_qa_size=1,
        limit=10000,
        generate_report=False,
        generate_sanity_check=False,
        generate_fixation=False,
    )

    if 0 == len(all_data_paths):
        raise ValueError("No valid data found after filtering!")

    print(f"Found {len(all_data_paths)} samples.")

    # 2. Build tokenizer
    tokenizer = build_tokenizer_from_data(all_data_paths, max_comment_len)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # 3. Dataset & DataLoader
    dataset = ExtendedVoxelDataset(
        all_data_paths,
        voxel_resolution=voxel_resolution,
        tokenizer=tokenizer,
        augment_color_p=0.0,
        jitter_voxel_p=0.0,
        emotion_order=emotion_order
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 4. Load model
    print("Loading model checkpoint...")
    model = DSVTFullModel(vocab_size=tokenizer.vocab_size, max_comment_len=max_comment_len)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model loaded successfully.")

    # 5. Inference loop
    total_samples = len(all_data_paths)
    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            inputs, (emotion_labels, heatmaps, tokens) = batch
            inputs = inputs.to(device, non_blocking=True)
            emotion_labels = emotion_labels.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            # Forward pass (in eval mode → captioner uses autoregressive decoding)
            emo_pred, gaze_pred, cap_pred = model(inputs, tokens=None)  # tokens=None triggers generation

            # Process each sample in batch
            for i in range(inputs.size(0)):
                if global_idx >= total_samples:
                    break

                item = all_data_paths[global_idx]
                pottery_id = item['ID']
                group = item['GROUP']
                session = item['SESSION_ID']
                sample_prefix = f"{group}_{session}"
                save_dir = os.path.join(save_root_dir, pottery_id)
                os.makedirs(save_dir, exist_ok=True)

                ref_pottery_path = item.get('processed_pottery_path', None)
                input_mask = inputs[i]

                # --- Save input point cloud ---
                input_pcd = voxel_grid_to_point_cloud(
                    inputs[i],
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=None
                )
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_input.ply"), input_pcd)

                # --- Save emotion predictions & GT ---
                emo_probs = torch.sigmoid(emo_pred[i])  # [5, D, H, W]
                gt_emotions = emotion_labels[i]         # [5, D, H, W]

                for e_idx, emo_name in enumerate(emotion_order):
                    color = DEFAULT_QNA_ANSWER_COLOR_MAP[emo_name]['rgb']
                    name = DEFAULT_QNA_ANSWER_COLOR_MAP[emo_name]['name']

                    pred_pcd = voxel_grid_to_point_cloud(
                        emo_probs[e_idx],
                        intensity_threshold=-1,
                        reference_pcd_path=ref_pottery_path,
                        mask_tensor=input_mask,
                        fixed_color_rgb=color
                    )
                    gt_pcd = voxel_grid_to_point_cloud(
                        gt_emotions[e_idx].float(),
                        intensity_threshold=-1,
                        reference_pcd_path=ref_pottery_path,
                        mask_tensor=input_mask,
                        fixed_color_rgb=color
                    )
                    o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_{name}_PRED.ply"), pred_pcd)
                    o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_{name}_GT.ply"), gt_pcd)

                # --- Save heatmap (gaze) ---
                heatmap_pred_pcd = voxel_grid_to_point_cloud(
                    torch.sigmoid(gaze_pred[i, 0]),  # squeeze channel
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=input_mask,
                    colormap_name='jet'
                )
                heatmap_gt_pcd = voxel_grid_to_point_cloud(
                    heatmaps[i, 0].float(),
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=input_mask,
                    colormap_name='jet'
                )
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_heatmap_PRED.ply"), heatmap_pred_pcd)
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_heatmap_GT.ply"), heatmap_gt_pcd)

                # --- Decode caption ---
                pred_ids = torch.argmax(cap_pred[i], dim=-1).cpu().numpy()  # [max_len]
                gt_ids = tokens[i].cpu().numpy()

                def decode(ids):
                    words = []
                    for idx in ids:
                        if idx == 2:  # <eos>
                            break
                        if idx not in [0, 1]:  # skip <pad>, <sos>
                            words.append(tokenizer.idx_to_word.get(idx, "<unk>"))
                    return " ".join(words)

                pred_text = decode(pred_ids)
                gt_text = decode(gt_ids)

                with open(os.path.join(save_dir, f"{sample_prefix}_caption.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Ground Truth: {gt_text}\n")
                    f.write(f"Prediction:   {pred_text}\n")

                global_idx += 1

    print(f"All predictions saved to: {save_root_dir}")


# --- Run ---
if __name__ == "__main__":
    # Ensure environment settings
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # set to 0 for inference
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    predict_and_save_by_pottery_id(
        model_checkpoint_path=r"C:\Users\User\Desktop\Python\jomon-kaen-3d-heatmap\lightning_logs\version_0\checkpoints\epoch=899-step=122400.ckpt",
        raw_data_dir=RAW_DATA_DIR,
        pottery_mesh_dir=MESH_DIR,
        save_root_dir=r"D:\storage\jomon_kaen\prediction_dsvt_9",
        voxel_resolution=VOXEL_RESOLUTION,
        max_comment_len=MAX_COMMENT_LEN,
        emotion_order=EMOTION_ORDER,
        batch_size=4,
        num_workers=4
    )