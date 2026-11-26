import argparse
import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from config import JEPAConfig
from dataset.point_jepa_dataset import JomonJEPADataModule
from models.multimodal_jepa import build_multimodal_jepa
from losses import MultimodalJEPACriterion


class JomonJEPALightningModule(pl.LightningModule):
    def __init__(self, cfg: JEPAConfig = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = JEPAConfig(**kwargs)
        elif isinstance(cfg, dict):
            cfg = JEPAConfig(**cfg)
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = build_multimodal_jepa(cfg)
        self.criterion = MultimodalJEPACriterion(cfg)

    def on_train_epoch_start(self):
        trainable = self.current_epoch >= self.cfg.freeze_backbone_epochs
        self.model.object_encoder.set_backbone_trainable(trainable)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.update_ema_teacher(momentum=0.99)

    def _step(self, batch, stage: str):
        outputs = self.model(batch)
        losses = self.criterion(outputs, batch)
        for k, v in losses.items():
            self.log(
                f"{stage}_{k}",
                v,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=(k in ("total_loss", "gaze_loss", "emotion_loss", "text_loss")),
                batch_size=batch["coord"].shape[0],
            )
        return losses["total_loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        backbone_params = list(self.model.object_encoder.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if id(p) not in backbone_ids and p.requires_grad]

        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in backbone_params if p.requires_grad], "lr": self.cfg.backbone_lr},
                {"params": other_params, "lr": self.cfg.lr},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.cfg.max_epochs), eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Specialized Multimodal 3D-JEPA")
    parser.add_argument("--run_mode", type=str, default="wsl_test", choices=["wsl_test", "dgx_full"])
    parser.add_argument("--country", type=str, default="japan", choices=["japan", "malaysia", "both", "all"])
    parser.add_argument("--artifact_type", type=str, default="both", choices=["pottery", "dogu", "both", "all"])
    parser.add_argument("--inputs", nargs="+", default=["object"], help="e.g. object gaze")
    parser.add_argument("--outputs", nargs="+", default=["gaze", "emotion", "text"], help="e.g. emotion text")
    parser.add_argument("--text_decoder", type=str, default="next_concept", choices=["slm", "next_concept"])
    parser.add_argument(
        "--backbone",
        type=str,
        default="ptv3_sonata",
        choices=["ptv3_sonata", "ptv3_concerto", "ptv3_utonia", "pointnet_fallback"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./runs_jomon_jepa")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(42)

    cfg = JEPAConfig(
        run_mode=args.run_mode,
        country=args.country,
        artifact_type=args.artifact_type,
        input_modalities=args.inputs,
        output_modalities=args.outputs,
        text_decoder_type=args.text_decoder,
        backbone_type=args.backbone,
        save_dir=args.save_dir,
    )
    cfg.apply_preset()

    if args.epochs is not None:
        cfg.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.devices is not None:
        cfg.devices = args.devices

    # Structured directory: {save_dir}/{country}/{pottery|dogu|both}/in_{inputs}__out_{outputs}[__{decoder}]
    exp_dir = cfg.get_experiment_dir()
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(cfg), f, indent=2)

    print(f"[Experiment Dir] {exp_dir}")

    datamodule = JomonJEPADataModule(cfg)
    model = JomonJEPALightningModule(cfg)

    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{val_total_loss:.4f}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    loggers = [
        CSVLogger(save_dir=log_dir, name="csv"),
        TensorBoardLogger(save_dir=log_dir, name="tb"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices if torch.cuda.is_available() else 1,
        strategy="ddp" if (torch.cuda.is_available() and cfg.devices > 1) else "auto",
        precision=cfg.precision if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt_callback, LearningRateMonitor(logging_interval="epoch")],
        logger=loggers,
        log_every_n_steps=2,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)
    print(f"Training complete.\n  Best checkpoint: {ckpt_callback.best_model_path}\n  Logs saved to:   {log_dir}")


if __name__ == "__main__":
    main()