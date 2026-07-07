"""Testa MTP nativa di Qwen3.6 come drafter per l'auto-speculativa.

Il checkpoint originale Qwen (non le conversioni mlx-community, che la
scartano) contiene una testa multi-token-prediction stile DeepSeek:
UN layer transformer (attention gated + MoE) che, dato l'hidden post-norm
della posizione t e l'embedding del token t+1, predice il token t+2.
Come drafter costa 1/40 di un forward pieno.

Pesi: mtp_head.safetensors (bf16, ~1,7 GB, scaricato via range-request dagli
shard originali). Struttura riusata da mlx_lm (DecoderLayer full-attention),
quindi la semantica di attention/RoPE/MoE è quella ufficiale; l'unico
adattamento è il rename HF->mlx_lm degli esperti (gate_up fuso da splittare).

La testa ha la sua KV cache, allineata posizione-per-posizione alla prompt
cache principale: va avanzata anche sui token accettati (un forward batched
da 1 layer per ciclo) e riavvolta in lockstep sui reject.
"""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class MTPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        from mlx_lm.models.qwen3_5 import DecoderLayer
        d = args.hidden_size
        eps = args.rms_norm_eps
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.pre_fc_norm_hidden = nn.RMSNorm(d, eps=eps)
        self.pre_fc_norm_embedding = nn.RMSNorm(d, eps=eps)
        self.norm = nn.RMSNorm(d, eps=eps)
        # layer_idx scelto perché (idx+1) % full_attention_interval == 0
        # -> variante full-attention (la testa MTP non è DeltaNet)
        self.layers = [DecoderLayer(args, args.full_attention_interval - 1)]

    def __call__(self, hidden, emb, cache):
        """hidden, emb: [1, T, D] -> hidden post-norm della testa [1, T, D]."""
        from mlx_lm.models.base import create_attention_mask
        h = self.pre_fc_norm_hidden(hidden)
        e = self.pre_fc_norm_embedding(emb)
        x = self.fc(mx.concatenate([e, h], axis=-1))  # ordine: embedding, hidden
        mask = create_attention_mask(x, cache[0]) if x.shape[1] > 1 else None
        x = self.layers[0](x, mask=mask, cache=cache[0])
        return self.norm(x)


def load_mtp_head(path, lang_model) -> MTPHead:
    """Carica mtp_head.safetensors e lo mappa sulla struttura mlx_lm."""
    head = MTPHead(lang_model.args)
    w = {k.removeprefix("mtp."): v for k, v in mx.load(str(Path(path))).items()}
    # rename HF -> mlx_lm (stesso split di Model.sanitize per i layer MoE)
    gu = w.pop("layers.0.mlp.experts.gate_up_proj")  # [E, 2I, D] fuso
    mid = gu.shape[-2] // 2
    w["layers.0.mlp.switch_mlp.gate_proj.weight"] = gu[..., :mid, :]
    w["layers.0.mlp.switch_mlp.up_proj.weight"] = gu[..., mid:, :]
    w["layers.0.mlp.switch_mlp.down_proj.weight"] = \
        w.pop("layers.0.mlp.experts.down_proj")
    # il checkpoint HF salva le RMSNorm zero-centered (peso effettivo = 1+w);
    # mlx_lm fa lo stesso +1.0 in conversione (qwen3_next.Model.sanitize).
    # NB: "norm" ovunque nel nome, non solo suffisso — pre_fc_norm_hidden e
    # pre_fc_norm_embedding finiscono in "hidden/embedding.weight"!
    w = {k: (v + 1.0 if "norm" in k and v.ndim == 1 else v)
         for k, v in w.items()}
    head.load_weights(list(w.items()), strict=True)
    return head


def make_mtp_cache(model):
    """KV cache dedicata della testa (1 layer full-attention)."""
    from mlx_lm.models.cache import KVCache
    return [KVCache()]
