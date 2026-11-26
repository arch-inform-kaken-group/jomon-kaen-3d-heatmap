from dataclasses import dataclass, field
from typing import List, Literal, Optional
import os

EMOTION_ORDER_JP = [
    "面白い・気になる形だ",
    "美しい・芸術的だ",
    "不思議・意味不明",
    "不気味・不安・怖い",
    "何も感じない",
]

EMOTION_ORDER_EN = [
    "Interesting and attentional shape",
    "Beautiful and artistic",
    "Strange and incomprehensible",
    "Creepy / unsettling / scary",
    "Feel nothing",
]

CANONICAL_EMOTION_COLORS = ["cyan", "green", "yellow", "red", "grey"]

# Canonical ordering for clean, deterministic folder names
MODALITY_ORDER = ["object", "gaze", "emotion", "text"]


@dataclass
class JEPAConfig:
    # Run Mode Preset: "wsl_test" or "dgx_full"
    run_mode: Literal["wsl_test", "dgx_full"] = "wsl_test"

    # Data Paths (by country)
    data_roots: dict = field(default_factory=lambda: {
        # "japan": "~/mnt/d/storage/jomon_kaen/jomon_kaen_dataset/japan",
        # "malaysia": "~/mnt/d/storage/jomon_kaen/jomon_kaen_dataset/malaysia",
        "japan": "~/Desktop/jomonkaen/jomon-kaen-3d-heatmap/src/jomon_kaen_dataset/japan",
        "malaysia": "~/Desktop/jomonkaen/jomon-kaen-3d-heatmap/src/jomon_kaen_dataset/malaysia",
    })
    # pottery_mesh_dir: str = "~/mnt/d/storage/jomon_kaen/pottery"
    pottery_mesh_dir: str = "~/Desktop/jomonkaen/jomon-kaen-3d-heatmap/src/pottery"
    save_dir: str = "./runs_jomon_jepa"

    # Country & Artifact Type Filtering
    country: Literal["japan", "malaysia", "both", "all"] = "japan"
    artifact_type: Literal["pottery", "dogu", "both", "all"] = "both"
    custom_artifact_type_json: Optional[str] = None

    # Vessel-disjoint validation IDs
    val_vessel_ids: List[str] = field(default_factory=lambda: [
        "NM0239(89)", "UD0028(93)", "NM0002(23)", "NM0175(47)",
        "NM0010(25)", "NM0001(22)", "SJ0504(55)", "UD0308(80)"
    ])
    val_split_ratio: float = 0.15

    # Specialized Input & Output Modalities
    # Only parameters strictly required for these modalities are instantiated.
    input_modalities: List[str] = field(default_factory=lambda: ["object"])
    output_modalities: List[str] = field(default_factory=lambda: ["gaze", "emotion", "text"])

    # Point Cloud Geometry Settings
    num_points: int = 4096
    voxel_resolution_preprocess: int = 80
    use_normals: bool = True

    # 3D Backbone (PointTransformerV3 / Sonata / Concerto)
    backbone_type: Literal["ptv3_sonata", "ptv3_concerto", "ptv3_utonia", "pointnet_fallback"] = "ptv3_sonata"
    ptv3_pretrained_repo: str = "facebook/sonata"
    freeze_backbone_epochs: int = 5
    latent_dim: int = 256
    num_jepa_queries: int = 32

    # Text Decoder Architecture Selection
    text_decoder_type: Literal["slm", "next_concept"] = "next_concept"
    slm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    concept_encoder_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    concept_dim: int = 768
    max_concepts_per_transcript: int = 6
    max_text_len: int = 128
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # Stochastic Sampling & Temperature Search
    default_temperature: float = 0.7
    top_p: float = 0.9
    latent_noise_scale: float = 0.15
    temperature_search_grid: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 1.0, 1.3]
    )
    num_eval_samples_per_item: int = 3

    # Training Hyperparameters
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 100
    lr: float = 2e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 0.01
    precision: str = "16-mixed"
    devices: int = 1
    limit_samples: Optional[int] = None

    # Loss Weights
    w_jepa: float = 1.0
    w_vicreg: float = 0.25
    w_gaze: float = 1.5
    w_emotion: float = 1.5
    w_text: float = 1.0

    def normalize_filters(self):
        """Normalizes 'all' -> 'both', expands '~' paths, and sorts modality lists canonically."""
        if self.country == "all":
            self.country = "both"
        if self.artifact_type == "all":
            self.artifact_type = "both"
        self.data_roots = {
            k: os.path.expanduser(v) for k, v in self.data_roots.items()
        }
        self.pottery_mesh_dir = os.path.expanduser(self.pottery_mesh_dir)
        self.save_dir = os.path.expanduser(self.save_dir)
        self.input_modalities = [m for m in MODALITY_ORDER if m in set(self.input_modalities)]
        self.output_modalities = [m for m in MODALITY_ORDER if m in set(self.output_modalities)]

    def get_experiment_tag(self) -> str:
        self.normalize_filters()
        in_str = "-".join(self.input_modalities)
        out_str = "-".join(self.output_modalities)
        dec_tag = f"__{self.text_decoder_type}" if "text" in self.output_modalities else ""
        return f"in_{in_str}__out_{out_str}{dec_tag}"

    def get_experiment_dir(self) -> str:
        """
        Builds structured path:
          {save_dir}/{country}/{pottery|dogu|both}/in_{inputs}__out_{outputs}[__{decoder}]
        """
        self.normalize_filters()
        return os.path.join(
            self.save_dir,
            self.country,
            self.artifact_type,
            self.get_experiment_tag(),
        )

    def apply_preset(self):
        """Applies hardware & runtime presets for WSL quick test vs DGX full training."""
        self.normalize_filters()
        if self.run_mode == "wsl_test":
            self.max_epochs = 2
            self.batch_size = 2
            self.num_workers = 2
            self.num_points = 2048
            self.limit_samples = 16
            self.devices = 1
            if self.text_decoder_type == "slm":
                self.slm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        elif self.run_mode == "dgx_full":
            self.max_epochs = 120
            self.batch_size = 16
            self.num_workers = 8
            self.num_points = 16384
            self.precision = "bf16-mixed"
            self.limit_samples = None