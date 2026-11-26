import argparse
import json
import os
from typing import Dict
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import JEPAConfig
from dataset.metadata import CANONICAL_EMOTIONS
from dataset.point_jepa_dataset import JomonJEPADataModule
from train import JomonJEPALightningModule
from dataset.utils import DEFAULT_QNA_ANSWER_COLOR_MAP


def compute_gaze_cc(pred: torch.Tensor, target: torch.Tensor) -> float:
    vp = pred - pred.mean()
    vt = target - target.mean()
    denom = torch.sqrt((vp ** 2).sum() * (vt ** 2).sum()) + 1e-6
    return float(((vp * vt).sum() / denom).item())


def compute_emotion_f1(pred_probs: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred_bin = (pred_probs > thresh).float()
    tp = (pred_bin * target).sum(dim=0)
    fp = (pred_bin * (1.0 - target)).sum(dim=0)
    fn = ((1.0 - pred_bin) * target).sum(dim=0)
    f1_per_class = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return float(f1_per_class.mean().item())


def build_concept_bank(model, dataloader, device) -> Dict:
    if not hasattr(model, "concept_encoder"):
        return {"embeddings": torch.empty(0), "texts": []}
    texts = []
    for batch in dataloader:
        for t in batch["text"]:
            if t and t not in texts:
                texts.append(t)
    if not texts:
        return {"embeddings": torch.empty(0), "texts": []}
    embs = model.concept_encoder.encode_texts(texts, device=device)
    return {"embeddings": embs, "texts": texts}


import glob
import trimesh
from scipy.spatial import cKDTree

def _load_dense_reference_points(vessel_id: str, pottery_mesh_dir: str, fallback_ply: str, target_dense_pts: int = 100000) -> np.ndarray:
    """
    Loads the original high-res pottery mesh from pottery_mesh_dir (sampling 100k surface points)
    for dense visualization; falls back to processed_pottery_path if raw mesh is not found.
    """
    base_id = vessel_id.split("(")[0].strip()
    candidates = []
    if pottery_mesh_dir and os.path.isdir(pottery_mesh_dir):
        for ext in ("obj", "ply", "glb", "stl"):
            candidates.extend(glob.glob(os.path.join(pottery_mesh_dir, f"*{base_id}*.{ext}")))

    if candidates:
        mesh = trimesh.load(candidates[0], force="mesh", process=False)
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces) > 0:
            pts, _ = trimesh.sample.sample_surface(mesh, target_dense_pts)
            return np.asarray(pts, dtype=np.float32)
        elif hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            return np.asarray(mesh.vertices, dtype=np.float32)

    if fallback_ply and os.path.exists(fallback_ply):
        ref_pcd = o3d.io.read_point_cloud(fallback_ply)
        return np.asarray(ref_pcd.points, dtype=np.float32)
    return None


def save_point_predictions_ply(
    coord_norm: np.ndarray,
    gaze_pred: np.ndarray,
    emo_pred: np.ndarray,
    save_dir: str,
    prefix: str,
    ref_pcd_path: str = None,
    vessel_id: str = "",
    pottery_mesh_dir: str = "",
    dense_points: int = 100000,
):
    os.makedirs(save_dir, exist_ok=True)

    full_pts = _load_dense_reference_points(
        vessel_id=vessel_id,
        pottery_mesh_dir=pottery_mesh_dir,
        fallback_ply=ref_pcd_path,
        target_dense_pts=dense_points,
    )

    if full_pts is not None and len(full_pts) > 0:
        centroid = full_pts.mean(axis=0, keepdims=True)
        full_norm = full_pts - centroid
        scale = np.max(np.linalg.norm(full_norm, axis=1)) + 1e-6
        full_norm = full_norm / scale

        tree = cKDTree(coord_norm)
        _, nn_idx = tree.query(full_norm, k=4)  # 4-NN smooth surface interpolation

        export_pts = full_pts
        if gaze_pred is not None:
            gaze_pred = gaze_pred[nn_idx].mean(axis=1)
        if emo_pred is not None:
            emo_pred = emo_pred[nn_idx].mean(axis=1)
    else:
        export_pts = coord_norm

    if gaze_pred is not None:
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
        g_norm = gaze_pred.squeeze() / (gaze_pred.max() + 1e-6)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(export_pts))
        pcd.colors = o3d.utility.Vector3dVector(cmap(g_norm)[:, :3])
        o3d.io.write_point_cloud(os.path.join(save_dir, f"{prefix}_gaze_PRED.ply"), pcd)

    if emo_pred is not None:
        for c_idx, meta in enumerate(CANONICAL_EMOTIONS):
            rgb = np.array(DEFAULT_QNA_ANSWER_COLOR_MAP[meta["jp"]]["rgb"], dtype=np.float64) / 255.0
            raw_p = emo_pred[:, c_idx]
            intensity = np.clip((raw_p - 0.25) / 0.75, 0.0, 1.0)[:, None]
            colors = 0.12 * (1.0 - intensity) + rgb[None, :] * intensity
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(export_pts))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(
                os.path.join(save_dir, f"{prefix}_emotion_{meta['color']}_PRED.ply"), pcd
            )

def evaluate_at_temperature(
    model,
    val_loader,
    device: torch.device,
    temperature: float,
    num_samples: int = 3,
    concept_bank: Dict = None,
    export_dir: str = None,
    pottery_mesh_dir: str = None,
    dense_points: int = 100000,
) -> Dict[str, float]:
    gaze_ccs, emo_f1s, text_sims = [], [], []

    for b_idx, batch in enumerate(val_loader):
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        B = batch_dev["coord"].shape[0]

        sample_preds = [
            model.predict(
                batch_dev,
                temperature=temperature,
                stochastic=(temperature > 0.05),
                concept_bank=concept_bank,
            )
            for _ in range(num_samples)
        ]

        for i in range(B):
            if "gaze" in sample_preds[0]:
                g_stack = torch.stack([sp["gaze"][i] for sp in sample_preds], dim=0).mean(dim=0)
                gaze_ccs.append(compute_gaze_cc(g_stack, batch_dev["gaze"][i]))

            if "emotion_probs" in sample_preds[0]:
                e_stack = torch.stack([sp["emotion_probs"][i] for sp in sample_preds], dim=0).mean(dim=0)
                emo_f1s.append(compute_emotion_f1(e_stack, batch_dev["emotion"][i]))

            if "text" in sample_preds[0] and hasattr(model, "concept_encoder"):
                gt_text = batch_dev["text"][i]
                cand_texts = [sp["text"][i] for sp in sample_preds]
                gt_emb = model.concept_encoder.encode_texts([gt_text], device=device)
                cand_embs = model.concept_encoder.encode_texts(cand_texts, device=device)
                sims = F.cosine_similarity(gt_emb, cand_embs, dim=-1)
                best_sample_idx = int(sims.argmax().item())
                text_sims.append(float(sims[best_sample_idx].item()))
            else:
                best_sample_idx = 0

            if export_dir is not None and b_idx < 5:
                prefix = f"{batch_dev['vessel_id'][i]}_{batch_dev['group'][i]}_{batch_dev['session_id'][i]}"
                g_out = sample_preds[best_sample_idx]["gaze"][i].cpu().numpy() if "gaze" in sample_preds[0] else None
                e_out = sample_preds[best_sample_idx]["emotion"][i].cpu().numpy() if "emotion" in sample_preds[0] else None
                save_point_predictions_ply(
                    batch_dev["coord"][i].cpu().numpy(),
                    g_out,
                    e_out,
                    export_dir,
                    prefix,
                    ref_pcd_path=batch_dev["ref_pcd_path"][i],
                    vessel_id=batch_dev["vessel_id"][i],
                    pottery_mesh_dir=pottery_mesh_dir,
                    dense_points=dense_points,
                )
                if "text" in sample_preds[0]:
                    with open(os.path.join(export_dir, f"{prefix}_text.txt"), "w", encoding="utf-8") as f:
                        f.write(f"GT:   {batch_dev['text'][i]}\n")
                        f.write(f"PRED: {sample_preds[best_sample_idx]['text'][i]}\n")

    gaze_score = float(np.mean(gaze_ccs)) if gaze_ccs else 0.0
    emo_score = float(np.mean(emo_f1s)) if emo_f1s else 0.0
    text_score = float(np.mean(text_sims)) if text_sims else 0.0
    composite = gaze_score + emo_score + text_score

    return {
        "temperature": temperature,
        "gaze_cc": gaze_score,
        "emotion_f1": emo_score,
        "text_concept_sim": text_score,
        "composite_score": composite,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate & Temperature Search for Specialized 3D-JEPA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_mode", type=str, default=None, choices=["wsl_test", "dgx_full"])
    parser.add_argument("--eval_country", type=str, default=None, choices=["japan", "malaysia", "both"])
    parser.add_argument("--eval_artifact_type", type=str, default=None, choices=["pottery", "dogu", "both"])
    parser.add_argument("--export_dir", type=str, default=None)
    parser.add_argument(
        "--dense_points",
        type=int,
        default=100000,
        help="Number of surface points sampled from raw meshes for dense .ply heatmap exports.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = JEPAConfig(**ckpt["hyper_parameters"])
    lit_model = JomonJEPALightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device, cfg=cfg
    )

    if args.run_mode is not None:
        cfg.run_mode = args.run_mode
        cfg.apply_preset()
    if args.eval_country is not None:
        cfg.country = args.eval_country
    if args.eval_artifact_type is not None:
        cfg.artifact_type = args.eval_artifact_type

    export_dir = args.export_dir or os.path.join(
        cfg.get_experiment_dir(),
        f"eval_{cfg.country}_{cfg.artifact_type}",
    )

    cfg.num_workers = 0
    datamodule = JomonJEPADataModule(cfg)
    datamodule.setup()

    model = lit_model.model.to(device).eval()
    concept_bank = None
    if cfg.text_decoder_type == "next_concept":
        concept_bank = build_concept_bank(model, datamodule.train_dataloader(), device)

    print(f"\n=== Temperature Search [{cfg.get_experiment_tag()} | {cfg.country}/{cfg.artifact_type}] ===")
    results = []
    for temp in tqdm(cfg.temperature_search_grid, desc="Temperature Grid"):
        metrics = evaluate_at_temperature(
            model=model,
            val_loader=datamodule.val_dataloader(),
            device=device,
            temperature=temp,
            num_samples=cfg.num_eval_samples_per_item,
            concept_bank=concept_bank,
            export_dir=None,
            pottery_mesh_dir=cfg.pottery_mesh_dir,
            dense_points=args.dense_points
        )
        results.append(metrics)
        print(
            f"T={temp:.2f} | Composite={metrics['composite_score']:.4f} | "
            f"Gaze CC={metrics['gaze_cc']:.4f} | Emotion F1={metrics['emotion_f1']:.4f} | "
            f"Text Sim={metrics['text_concept_sim']:.4f}"
        )

    best_result = max(results, key=lambda r: r["composite_score"])
    best_temp = best_result["temperature"]
    print(f"\n>>> Best Temperature: T = {best_temp} (Composite = {best_result['composite_score']:.4f})")

    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, "best_temperature.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best_result, "all_trials": results}, f, indent=2)

    evaluate_at_temperature(
        model=model,
        val_loader=datamodule.val_dataloader(),
        device=device,
        temperature=best_temp,
        num_samples=cfg.num_eval_samples_per_item,
        concept_bank=concept_bank,
        export_dir=export_dir,
    )
    print(f"Saved optimal predictions to {export_dir}")


from utils.crash_logger import run_with_crash_logger

if __name__ == "__main__":
    run_with_crash_logger(
        main,
        log_dir="./runs_jomon_jepa/crash_logs",
        script_name="eval",
    )