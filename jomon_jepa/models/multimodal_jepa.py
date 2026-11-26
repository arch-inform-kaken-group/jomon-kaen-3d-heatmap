from typing import Dict, Optional
import torch
import torch.nn as nn

from models.encoders import (
    PretrainedPTv3ObjectEncoder,
    SpatialAffectEncoder,
    FrozenConceptEncoder,
)
from models.jepa import Multimodal3DJEPAPredictor
from models.spatial_heads import PointSpatialAffectDecoder
from models.text_decoders import SLMPerceiverTextDecoder, NextConceptLCMDecoder


VALID_MODALITIES = {"object", "gaze", "emotion", "text"}


class Multimodal3DJEPAModel(nn.Module):
    """
    Pruned Multimodal 3D-JEPA Model:
    Instantiates ONLY the encoders, JEPA query parameters, EMA teachers, and output heads
    strictly required by `input_modalities` and `output_modalities`.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        cfg.normalize_filters()
        self.input_modalities = set(cfg.input_modalities)
        self.output_modalities = set(cfg.output_modalities)

        if not self.input_modalities.issubset(VALID_MODALITIES):
            raise ValueError(f"Invalid input_modalities: {self.input_modalities - VALID_MODALITIES}")
        if not self.output_modalities.issubset(VALID_MODALITIES - {"object"}) or len(self.output_modalities) == 0:
            raise ValueError(f"Invalid output_modalities: {self.output_modalities}")
        if "object" not in self.input_modalities:
            raise ValueError("'object' must always be included in input_modalities as the 3D anchor.")

        self.has_spatial_input = bool({"gaze", "emotion"} & self.input_modalities)
        self.has_text_input = "text" in self.input_modalities
        self.predict_spatial = bool({"gaze", "emotion"} & self.output_modalities)
        self.predict_text = "text" in self.output_modalities

        # 1. Core 3D Object Encoder (PTv3 Sonata / Concerto)
        self.object_encoder = PretrainedPTv3ObjectEncoder(
            backbone_type=cfg.backbone_type, latent_dim=cfg.latent_dim
        )

        # 2. Spatial Affect Encoders (only allocated if gaze/emotion is an input or output)
        if self.has_spatial_input:
            self.spatial_encoder = SpatialAffectEncoder(latent_dim=cfg.latent_dim)
        if self.predict_spatial:
            self.spatial_teacher = SpatialAffectEncoder(latent_dim=cfg.latent_dim)
            if self.has_spatial_input:
                self.spatial_teacher.load_state_dict(self.spatial_encoder.state_dict())
            for p in self.spatial_teacher.parameters():
                p.requires_grad = False

        # 3. Frozen Concept Encoder (only allocated if text is an input or output)
        if self.has_text_input or self.predict_text:
            self.concept_encoder = FrozenConceptEncoder(
                model_name=cfg.concept_encoder_name,
                concept_dim=cfg.concept_dim,
                latent_dim=cfg.latent_dim,
            )

        # 4. Pruned Cross-Modal 3D-JEPA Predictor
        self.jepa = Multimodal3DJEPAPredictor(
            latent_dim=cfg.latent_dim,
            num_queries=cfg.num_jepa_queries,
            num_layers=4,
            num_heads=8,
            has_spatial_input=self.has_spatial_input,
            has_text_input=self.has_text_input,
            predict_spatial=self.predict_spatial,
            predict_concept=self.predict_text,
        )

        # 5. Output Heads (only allocated for active output_modalities)
        if self.predict_spatial:
            self.spatial_decoder = PointSpatialAffectDecoder(latent_dim=cfg.latent_dim)

        if self.predict_text:
            if cfg.text_decoder_type == "slm":
                self.text_decoder = SLMPerceiverTextDecoder(
                    latent_dim=cfg.latent_dim,
                    model_name=cfg.slm_model_name,
                    use_lora=cfg.use_lora,
                    lora_r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    max_text_len=cfg.max_text_len,
                )
            elif cfg.text_decoder_type == "next_concept":
                self.text_decoder = NextConceptLCMDecoder(
                    latent_dim=cfg.latent_dim,
                    concept_dim=self.concept_encoder.concept_dim,
                    max_concepts=cfg.max_concepts_per_transcript,
                )
            else:
                raise ValueError(f"Unknown text_decoder_type: {cfg.text_decoder_type}")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[ModelBuilt] {cfg.get_experiment_tag()} | "
            f"Trainable Params: {trainable_params / 1e6:.2f}M / Total: {total_params / 1e6:.2f}M"
        )

    @torch.no_grad()
    def update_ema_teacher(self, momentum: float = 0.99):
        if self.has_spatial_input and self.predict_spatial:
            for p_s, p_t in zip(self.spatial_encoder.parameters(), self.spatial_teacher.parameters()):
                p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    def _encode_inputs(self, batch: Dict):
        coord = batch["coord"]
        obj_feats = self.object_encoder(coord, batch["color"], batch["normal"])

        cond_spatial = None
        if self.has_spatial_input:
            gaze_in = batch["gaze"] if "gaze" in self.input_modalities else None
            emo_in = batch["emotion"] if "emotion" in self.input_modalities else None
            cond_spatial = self.spatial_encoder(coord, gaze=gaze_in, emotion=emo_in)

        cond_text = None
        if self.has_text_input:
            text_embs = self.concept_encoder.encode_texts(batch["text"], device=coord.device)
            cond_text = self.concept_encoder.to_latent(text_embs)

        return obj_feats, cond_spatial, cond_text

    def _gather_text_conditioning_tokens(self, jepa_out: Dict) -> torch.Tensor:
        tokens = []
        if "pred_spatial_tokens" in jepa_out:
            tokens.append(jepa_out["pred_spatial_tokens"])
        if "pred_concept_tokens" in jepa_out:
            tokens.append(jepa_out["pred_concept_tokens"])
        return torch.cat(tokens, dim=1)

    def forward(self, batch: Dict) -> Dict:
        coord = batch["coord"]
        device = coord.device
        obj_feats, cond_spatial, cond_text = self._encode_inputs(batch)

        jepa_out = self.jepa(
            obj_point_feats=obj_feats,
            coord=coord,
            cond_spatial_feats=cond_spatial,
            cond_text_latent=cond_text,
            temperature=self.cfg.default_temperature,
            noise_scale=self.cfg.latent_noise_scale,
            stochastic=self.training,
        )

        outputs = {"jepa": jepa_out}

        if self.predict_spatial:
            with torch.no_grad():
                target_spatial_pts = self.spatial_teacher(
                    coord, gaze=batch["gaze"], emotion=batch["emotion"]
                )
                outputs["target_global_spatial"] = target_spatial_pts.mean(dim=1)

            gaze_logits, emotion_logits = self.spatial_decoder(
                obj_feats, jepa_out["pred_spatial_tokens"]
            )
            if "gaze" in self.output_modalities:
                outputs["gaze_logits"] = gaze_logits
            if "emotion" in self.output_modalities:
                outputs["emotion_logits"] = emotion_logits

        if self.predict_text:
            target_text_embs = self.concept_encoder.encode_texts(batch["text"], device=device)
            outputs["target_global_concept"] = self.concept_encoder.to_latent(target_text_embs)

            combined_tokens = self._gather_text_conditioning_tokens(jepa_out)
            if self.cfg.text_decoder_type == "slm":
                outputs["text_loss"] = self.text_decoder(
                    combined_tokens, batch["text"], batch["country"], batch["artifact_type"]
                )
            else:
                concept_embs, concept_mask = self.concept_encoder.encode_concept_sequences(
                    batch["concepts"], max_len=self.cfg.max_concepts_per_transcript, device=device
                )
                lcm_out = self.text_decoder(
                    combined_tokens, concept_embs, concept_mask, batch["text"]
                )
                outputs["text_loss"] = lcm_out["loss"]
                outputs["pred_concepts"] = lcm_out["pred_concepts"]

        return outputs

    @torch.no_grad()
    def predict(
        self,
        batch: Dict,
        temperature: float = 0.7,
        stochastic: bool = True,
        concept_bank: Optional[Dict] = None,
    ) -> Dict:
        self.eval()
        coord = batch["coord"]
        obj_feats, cond_spatial, cond_text = self._encode_inputs(batch)

        jepa_out = self.jepa(
            obj_point_feats=obj_feats,
            coord=coord,
            cond_spatial_feats=cond_spatial,
            cond_text_latent=cond_text,
            temperature=temperature,
            noise_scale=self.cfg.latent_noise_scale,
            stochastic=stochastic,
        )

        preds = {}
        if self.predict_spatial:
            gaze_logits, emotion_logits = self.spatial_decoder(
                obj_feats, jepa_out["pred_spatial_tokens"]
            )
            gaze_probs, emo_out = self.spatial_decoder.sample_outputs(
                gaze_logits, emotion_logits, temperature=temperature, stochastic=stochastic
            )
            if "gaze" in self.output_modalities:
                preds["gaze"] = gaze_probs
            if "emotion" in self.output_modalities:
                preds["emotion"] = emo_out
                preds["emotion_probs"] = torch.sigmoid(emotion_logits / max(temperature, 1e-3))

        if self.predict_text:
            combined_tokens = self._gather_text_conditioning_tokens(jepa_out)
            if self.cfg.text_decoder_type == "slm":
                preds["text"] = self.text_decoder.generate(
                    combined_tokens,
                    batch["country"],
                    batch["artifact_type"],
                    temperature=temperature,
                    top_p=self.cfg.top_p,
                )
            else:
                lcm_gen = self.text_decoder.generate(
                    combined_tokens, temperature=temperature, concept_bank=concept_bank
                )
                preds["text"] = lcm_gen["texts"]
                preds["pred_concepts"] = lcm_gen["pred_concepts"]

        return preds


def build_multimodal_jepa(cfg) -> Multimodal3DJEPAModel:
    return Multimodal3DJEPAModel(cfg)