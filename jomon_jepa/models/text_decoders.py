from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_latents: int = 16, num_heads: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, out_dim) * 0.02)
        self.in_proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        kv = self.in_proj(x)
        q = self.latents.expand(B, -1, -1)
        out, _ = self.attn(self.norm(q), kv, kv)
        q = q + out
        q = q + self.ffn(self.norm(q))
        return q


class SLMPerceiverTextDecoder(nn.Module):
    """
    Lightweight Causal SLM conditioned on JEPA spatial & concept tokens via Perceiver Resampler.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        max_text_len: int = 128,
    ):
        super().__init__()
        self.max_text_len = max_text_len
        self.model_name = model_name
        self.lm = None
        self.tokenizer = None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.lm = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            if use_lora:
                try:
                    from peft import LoraConfig, get_peft_model

                    lora_cfg = LoraConfig(
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        target_modules=["q_proj", "v_proj"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    self.lm = get_peft_model(self.lm, lora_cfg)
                except Exception as peft_err:
                    print(f"[SLM] PEFT LoRA not applied ({peft_err}); freezing LM weights.")
                    for p in self.lm.parameters():
                        p.requires_grad = False
            else:
                for p in self.lm.parameters():
                    p.requires_grad = False

            lm_hidden = self.lm.config.hidden_size
        except Exception as e:
            print(f"[SLM] Could not load HuggingFace SLM '{model_name}' ({e}). Using compact GRU fallback.")
            lm_hidden = 256
            self.fallback_vocab = 4096
            self.fallback_emb = nn.Embedding(self.fallback_vocab, lm_hidden)
            self.fallback_gru = nn.GRU(lm_hidden, lm_hidden, num_layers=2, batch_first=True)
            self.fallback_head = nn.Linear(lm_hidden, self.fallback_vocab)

        self.resampler = PerceiverResampler(in_dim=latent_dim, out_dim=lm_hidden, num_latents=16)

    def _build_prompts(self, countries: List[str], artifact_types: List[str]) -> List[str]:
        prompts = []
        for c, a in zip(countries, artifact_types):
            if c == "japan":
                prompts.append(f"遺物種別: {a}。視線と感情の反応に基づく観察コメント: ")
            else:
                prompts.append(f"Artifact: {a}. Observation comment based on gaze and emotion: ")
        return prompts

    def forward(
        self,
        jepa_tokens: torch.Tensor,
        texts: List[str],
        countries: List[str],
        artifact_types: List[str],
    ) -> torch.Tensor:
        prefix_embeds = self.resampler(jepa_tokens)  # [B, 16, H]
        device = prefix_embeds.device

        if self.lm is None:
            # Offline fallback language loss
            B = jepa_tokens.shape[0]
            ids = torch.zeros((B, 32), dtype=torch.long, device=device)
            for b, t in enumerate(texts):
                chars = [max(1, ord(ch) % self.fallback_vocab) for ch in (t or " ")[:31]]
                ids[b, : len(chars)] = torch.tensor(chars, device=device)
            h0 = prefix_embeds.mean(dim=1).unsqueeze(0).repeat(2, 1, 1)
            out, _ = self.fallback_gru(self.fallback_emb(ids[:, :-1]), h0)
            logits = self.fallback_head(out)
            return F.cross_entropy(logits.reshape(-1, self.fallback_vocab), ids[:, 1:].reshape(-1), ignore_index=0)

        prompts = self._build_prompts(countries, artifact_types)
        full_texts = [p + (t if t else "特になし") + self.tokenizer.eos_token for p, t in zip(prompts, texts)]
        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        tok_embeds = self.lm.get_input_embeddings()(input_ids)

        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        prefix_mask = torch.ones((inputs_embeds.shape[0], prefix_embeds.shape[1]), device=device, dtype=attn_mask.dtype)
        full_mask = torch.cat([prefix_mask, attn_mask], dim=1)

        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        prefix_labels = torch.full(
            (inputs_embeds.shape[0], prefix_embeds.shape[1]), -100, device=device, dtype=labels.dtype
        )
        full_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=full_labels)
        if prefix_embeds.shape[0] > 1:
            prefix_std = prefix_embeds.float().std(dim=0).mean()
            diversity_penalty = F.relu(0.25 - prefix_std)
            return outputs.loss + 0.5 * diversity_penalty
            
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        jepa_tokens: torch.Tensor,
        countries: List[str],
        artifact_types: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[str]:
        prefix_embeds = self.resampler(jepa_tokens)
        device = prefix_embeds.device
        B = jepa_tokens.shape[0]

        if self.lm is None:
            return ["[Fallback SLM sample output]" for _ in range(B)]

        prompts = self._build_prompts(countries, artifact_types)
        enc = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        tok_embeds = self.lm.get_input_embeddings()(enc["input_ids"])
        prefix_embeds = prefix_embeds.to(tok_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        prefix_mask = torch.ones((B, prefix_embeds.shape[1]), device=device, dtype=enc["attention_mask"].dtype)
        full_mask = torch.cat([prefix_mask, enc["attention_mask"]], dim=1)

        do_sample = temperature > 0.05
        gen_ids = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            max_new_tokens=64,
            do_sample=do_sample,
            temperature=max(temperature, 1e-3),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


class NextConceptLCMDecoder(nn.Module):
    """
    Large Concept Model (LCM) / Next-Concept Predictor (Requirement 6):
      1. Predicts a sequence of sentence-level concept embeddings [B, S, concept_dim]
         autoregressively from 3D JEPA context tokens using MSE + Cosine + Diffusion noise.
      2. Decodes predicted concept embeddings back into human-readable text via a lightweight
         concept-conditioned text decoder (or nearest-concept bank during evaluation).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        concept_dim: int = 768,
        max_concepts: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.concept_dim = concept_dim
        self.max_concepts = max_concepts

        self.bos_concept = nn.Parameter(torch.randn(1, 1, concept_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_concepts + 1, latent_dim) * 0.02)
        self.concept_in_proj = nn.Linear(concept_dim, latent_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.concept_transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.concept_out_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, concept_dim),
        )
        self.stop_head = nn.Linear(latent_dim, 1)

        # Lightweight character/subword decoder to render predicted concept vectors into text
        self.vocab_size = 4096
        self.char_emb = nn.Embedding(self.vocab_size, 256)
        self.concept_to_gru = nn.Linear(concept_dim, 256)
        self.text_gru = nn.GRU(256, 256, num_layers=2, batch_first=True)
        self.text_out = nn.Linear(256, self.vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        jepa_tokens: torch.Tensor,
        target_concepts: torch.Tensor,
        concept_mask: torch.Tensor,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            jepa_tokens:     [B, Q, latent_dim]
            target_concepts: [B, S, concept_dim]
            concept_mask:    [B, S]
        """
        B, S, _ = target_concepts.shape
        device = jepa_tokens.device

        # Diffusion-style Gaussian noise injection during training (as in Two-Tower Diffusion LCM)
        noisy_targets = target_concepts + torch.randn_like(target_concepts) * 0.05
        bos = self.bos_concept.expand(B, 1, -1)
        shifted_concepts = torch.cat([bos, noisy_targets[:, :-1]], dim=1)  # [B, S, concept_dim]

        tgt = self.concept_in_proj(shifted_concepts) + self.pos_embed[:, :S]
        causal_mask = self._causal_mask(S, device)

        h = self.concept_transformer(tgt=tgt, memory=jepa_tokens, tgt_mask=causal_mask)
        pred_concepts = self.concept_out_proj(h)  # [B, S, concept_dim]
        stop_logits = self.stop_head(h).squeeze(-1)  # [B, S]

        # Concept prediction loss (MSE + 1 - CosineSimilarity)
        valid = concept_mask.unsqueeze(-1)
        mse_loss = ((pred_concepts - target_concepts) ** 2 * valid).sum() / (valid.sum() * self.concept_dim + 1e-6)
        cos_sim = F.cosine_similarity(pred_concepts, target_concepts, dim=-1)
        cos_loss = ((1.0 - cos_sim) * concept_mask).sum() / (concept_mask.sum() + 1e-6)

        # Stop-concept binary loss
        stop_targets = 1.0 - concept_mask
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)

        # Train concept-to-text renderer on pooled ground-truth concepts
        pooled_concept = (target_concepts * valid).sum(dim=1) / (valid.sum(dim=1) + 1e-6)
        h0 = torch.tanh(self.concept_to_gru(pooled_concept)).unsqueeze(0).repeat(2, 1, 1)
        max_len = 64
        ids = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for b, t in enumerate(texts):
            # Byte-level UTF-8 encoding handles both Japanese & English losslessly
            raw_bytes = [min(self.vocab_size - 2, int(byte) + 2) for byte in (t or " ").encode("utf-8")[: max_len - 2]]
            seq = [1] + raw_bytes + [self.vocab_size - 1]
            ids[b, : len(seq)] = torch.tensor(seq, device=device)

        gru_out, _ = self.text_gru(self.char_emb(ids[:, :-1]), h0)
        text_logits = self.text_out(gru_out)
        render_loss = F.cross_entropy(
            text_logits.reshape(-1, self.vocab_size), ids[:, 1:].reshape(-1), ignore_index=0
        )

        return {
            "loss": mse_loss + cos_loss + 0.2 * stop_loss + 0.5 * render_loss,
            "pred_concepts": pred_concepts,
        }

    @torch.no_grad()
    def generate(
        self,
        jepa_tokens: torch.Tensor,
        temperature: float = 0.7,
        concept_bank: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, object]:
        """
        Autoregressively generates concept vectors with temperature-scaled Gaussian sampling,
        then renders text (and optionally retrieves nearest verbalized concepts from concept_bank).
        """
        B = jepa_tokens.shape[0]
        device = jepa_tokens.device
        seq = self.bos_concept.expand(B, 1, -1)
        generated_concepts = []

        for step in range(self.max_concepts):
            S = seq.shape[1]
            tgt = self.concept_in_proj(seq) + self.pos_embed[:, :S]
            causal_mask = self._causal_mask(S, device)
            h = self.concept_transformer(tgt=tgt, memory=jepa_tokens, tgt_mask=causal_mask)
            next_c = self.concept_out_proj(h[:, -1:, :])
            if temperature > 0.05:
                next_c = next_c + torch.randn_like(next_c) * (0.1 * temperature)
            generated_concepts.append(next_c)
            seq = torch.cat([seq, next_c], dim=1)

        pred_concepts = torch.cat(generated_concepts, dim=1)  # [B, max_concepts, concept_dim]
        pooled_concept = pred_concepts.mean(dim=1)

        # Render via byte-level GRU decoder with temperature sampling
        h = torch.tanh(self.concept_to_gru(pooled_concept)).unsqueeze(0).repeat(2, 1, 1)
        cur_tok = torch.ones((B, 1), dtype=torch.long, device=device)
        byte_lists = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(64):
            out, h = self.text_gru(self.char_emb(cur_tok), h)
            logits = self.text_out(out[:, -1, :]) / max(temperature, 1e-3)
            # Restrict sampling strictly to valid UTF-8 byte tokens [2..257] and EOS [vocab_size - 1]
            logits[:, 258 : self.vocab_size - 1] = -float("inf")
            logits[:, 0:2] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = (
                torch.multinomial(probs, num_samples=1)
                if temperature > 0.05
                else logits.argmax(dim=-1, keepdim=True)
            )
            cur_tok = next_tok
            for b in range(B):
                if done[b]:
                    continue
                tid = int(next_tok[b, 0].item())
                if tid == self.vocab_size - 1 or tid == 0:
                    done[b] = True
                elif 2 <= tid < 258:
                    byte_lists[b].append(tid - 2)

        rendered_texts = [bytes(bl).decode("utf-8", errors="ignore") for bl in byte_lists]

        # If a concept bank of training sentences is provided, also retrieve nearest semantic concept
        retrieved_texts = []
        if concept_bank is not None and len(concept_bank["embeddings"]) > 0:
            bank_embs = F.normalize(concept_bank["embeddings"].to(device), dim=-1)
            q_embs = F.normalize(pooled_concept, dim=-1)
            sims = q_embs @ bank_embs.T
            best_idx = sims.argmax(dim=-1).tolist()
            retrieved_texts = [concept_bank["texts"][i] for i in best_idx]

        return {
            "pred_concepts": pred_concepts,
            "texts": retrieved_texts if retrieved_texts else rendered_texts,
            "rendered_texts": rendered_texts,
        }