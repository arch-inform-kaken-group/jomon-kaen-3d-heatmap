import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dataset.utils import filter_data_on_condition, DEFAULT_QNA_ANSWER_COLOR_MAP
from dataset.voxel_dataset import ExtendedVoxelDataset
from full_model_GPT_multi_token_split_heads import (
    ExtendedVoxelDataModule,
    MeaningMakingLightningModule,
    MeaningMakingModel,
    SimpleTokenizer
)
from model.config import *
from model.helper import voxel_grid_to_point_cloud

# --- Reuse tokenizer logic ---
def build_tokenizer_from_data(all_data_paths, max_comment_len=50):
    all_comments = []
    for item in all_data_paths:
        comment_path = item.get('TRANSCRIPT', '')
        if os.path.exists(comment_path):
            try:
                with open(comment_path, 'r', encoding='utf-8') as f:
                    all_comments.append(f.read().strip())
            except Exception as e:
                print(f"WARNING: Failed to read {comment_path}")
    tokenizer = SimpleTokenizer(max_len=max_comment_len)
    tokenizer.build_vocab(all_comments)
    return tokenizer

# --- Main prediction function ---
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

    # 1. Load all data paths
    print(" Filtering data paths...")
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

    if len(all_data_paths) == 0:
        raise ValueError("No data found after filtering!")

    print(f" Found {len(all_data_paths)} samples.")

    # 2. Build tokenizer
    tokenizer = build_tokenizer_from_data(all_data_paths, max_comment_len)

    # 3. Create dataloader
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

    # 4. Dummy datamodule for vocab size (only needed for model init)
    dummy_datamodule = ExtendedVoxelDataModule(
        all_data_paths=[],
        batch_size=1,
        num_workers=0,
        voxel_resolution=voxel_resolution,
        max_comment_len=max_comment_len,
        test_groups=[],
    )
    dummy_datamodule.tokenizer = tokenizer

    # 5. Load model correctly
    print(" Loading model checkpoint...")
    lightning_module = MeaningMakingLightningModule.load_from_checkpoint(
        model_checkpoint_path,
        model=MeaningMakingModel(
            num_emotions=len(EMOTION_ORDER),
            vocab_size=dummy_datamodule.tokenizer.vocab_size,
            max_comment_len=MAX_COMMENT_LEN,
            conv_dims=CONV_DIMS,
            resolution=VOXEL_RESOLUTION,
            num_experts=NUM_EXPERTS
        ),
        strict=True
    )
    model = lightning_module.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(" Model loaded.")

    # 6. Run inference with progress bar
    total_samples = len(all_data_paths)
    print(f" Running inference on {total_samples} samples...")

    with torch.no_grad():
        global_idx = 0
        for batch in tqdm(dataloader, desc="Inference", total=len(dataloader)):
            inputs, (emotion_labels, heatmaps, tokens) = batch
            inputs = inputs.to(device)
            tokens = tokens.to(device)
            emotion_labels = emotion_labels.to(device)
            heatmaps = heatmaps.to(device)

            # Get all expert outputs
            all_outputs, _ = model(inputs, target_tokens=None, return_all_experts=True)

            # Select best expert by dummy loss (or use expert 0 if you prefer)
            branch_losses = []
            for emo_pred, heat_pred, tok_pred, _ in all_outputs:
                loss_emo = torch.mean(torch.sigmoid(emo_pred))
                loss_heat = torch.mean(torch.sigmoid(heat_pred))
                branch_losses.append(loss_emo + loss_heat)
            best_idx = torch.argmin(torch.tensor(branch_losses)).item()
            emotion_preds, heatmap_preds, token_preds, _ = all_outputs[best_idx]

            # Save each sample
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

                # --- Save input ---
                input_pcd = voxel_grid_to_point_cloud(
                    inputs[i],
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=None
                )
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_input.ply"), input_pcd)

                # --- Save emotions ---
                preds_prob = torch.sigmoid(emotion_preds[i])
                labels = emotion_labels[i]
                for e_idx, emo_name in enumerate(emotion_order):
                    color = DEFAULT_QNA_ANSWER_COLOR_MAP[emo_name]['rgb']
                    name = DEFAULT_QNA_ANSWER_COLOR_MAP[emo_name]['name']
                    pred_pcd = voxel_grid_to_point_cloud(
                        preds_prob[e_idx],
                        intensity_threshold=-1,
                        reference_pcd_path=ref_pottery_path,
                        mask_tensor=input_mask,
                        fixed_color_rgb=color
                    )
                    gt_pcd = voxel_grid_to_point_cloud(
                        labels[e_idx].float(),
                        intensity_threshold=-1,
                        reference_pcd_path=ref_pottery_path,
                        mask_tensor=input_mask,
                        fixed_color_rgb=color
                    )
                    o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_{name}_PRED.ply"), pred_pcd)
                    o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_{name}_GT.ply"), gt_pcd)

                # --- Save heatmap ---
                heatmap_pred_pcd = voxel_grid_to_point_cloud(
                    torch.sigmoid(heatmap_preds[i]),
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=input_mask,
                    colormap_name='jet'
                )
                heatmap_gt_pcd = voxel_grid_to_point_cloud(
                    heatmaps[i].float(),
                    intensity_threshold=-1,
                    reference_pcd_path=ref_pottery_path,
                    mask_tensor=input_mask,
                    colormap_name='jet'
                )
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_heatmap_PRED.ply"), heatmap_pred_pcd)
                o3d.io.write_point_cloud(os.path.join(save_dir, f"{sample_prefix}_heatmap_GT.ply"), heatmap_gt_pcd)

                # --- Save caption ---
                pred_ids = torch.argmax(token_preds[i], dim=-1).cpu().numpy()
                gt_ids = tokens[i].cpu().numpy()

                def decode(ids):
                    words = []
                    for idx in ids:
                        if idx == 2: break
                        if idx not in [0, 1]:
                            w = tokenizer.idx_to_word.get(idx, "<unk>")
                            words.append(w)
                    return " ".join(words)

                pred_text = decode(pred_ids)
                gt_text = decode(gt_ids)

                with open(os.path.join(save_dir, f"{sample_prefix}_caption.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Ground Truth: {gt_text}\n")
                    f.write(f"Prediction:   {pred_text}\n")

                global_idx += 1

    print(f" All predictions saved to: {save_root_dir}")

# --- Run ---
if __name__ == "__main__":
    predict_and_save_by_pottery_id(
        model_checkpoint_path=r"D:\storage\jomon_kaen\validation_predictions_efficient_fixed_13\lightning_logs\version_0\checkpoints\last.ckpt",
        raw_data_dir=RAW_DATA_DIR,
        pottery_mesh_dir=MESH_DIR,
        save_root_dir=r"D:\storage\jomon_kaen\prediction_13",
        voxel_resolution=VOXEL_RESOLUTION,
        max_comment_len=MAX_COMMENT_LEN,
        emotion_order=EMOTION_ORDER,
        batch_size=4,
        num_workers=4
    )