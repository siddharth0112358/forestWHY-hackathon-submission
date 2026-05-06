"""I-JEPA Sentinel-2 encoder + 6 differential JEPA panels.

Vendored from the upstream forestWHY training repo. Inference path only —
no training/EMA/HDF5 code. Self-contained: imports only torch, timm,
numpy, PIL, matplotlib, scikit-learn (PCA panel).

The encoder is a ViT-Large/8 over 13-channel Sentinel-2 tiles (64×64). Weights
are pre-trained with I-JEPA on global Sentinel-2; the loader auto-downloads
from `Siddharth63/forestWHY-JEPA-vitl` if no local checkpoint is provided.

Public API:
    load_jepa_encoder(ckpt_path=None, device="auto", hf_repo=..., hf_filename=...) -> S2Encoder
    make_jepa_panels(before_13band, after_13band, encoder, device, size=128) -> dict[str, PIL.Image]
    BAND_IDX, S2_MEAN, S2_STD               (constants)
    to_rgb_image, colorise                  (panel helpers, re-exported by spectral.py)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.models.vision_transformer import Block

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel-2 constants
# ─────────────────────────────────────────────────────────────────────────────

BAND_IDX: dict[str, int] = {
    "B1": 0, "B2": 1, "B3": 2,  "B4": 3, "B5": 4,
    "B6": 5, "B7": 6, "B8": 7,  "B8A": 8, "B9": 9,
    "B10": 10, "B11": 11, "B12": 12,
}

S2_MEAN = np.array([
    0.0334, 0.0428, 0.0614, 0.0590, 0.0902, 0.1820,
    0.2218, 0.2430, 0.2507, 0.2648, 0.1568, 0.1093, 0.0764,
], dtype=np.float32)

S2_STD = np.array([
    0.0554, 0.0572, 0.0582, 0.0671, 0.0778, 0.1160,
    0.1271, 0.1458, 0.1491, 0.1500, 0.1118, 0.0933, 0.0730,
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers (used by spectral.py + panel functions)
# ─────────────────────────────────────────────────────────────────────────────

def patch_to_band_dict(patch: np.ndarray) -> Dict[str, np.ndarray]:
    patch = patch.astype(np.float32)
    patch[patch < -1e10] = 0.0
    return {name: np.clip(patch[idx], 0.0, 1.0) for name, idx in BAND_IDX.items()}


def to_rgb_image(r: np.ndarray, g: np.ndarray, b: np.ndarray, size: int = 128) -> Image.Image:
    stack = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    p2, p98 = np.percentile(stack, 2), np.percentile(stack, 98)
    stack = np.clip((stack - p2) / (p98 - p2 + 1e-6), 0, 1)
    img = Image.fromarray((stack * 255).astype(np.uint8))
    return img.resize((size, size), Image.BILINEAR) if size != stack.shape[0] else img


def colorise(
    arr: np.ndarray,
    cmap: str = "RdYlGn",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    size: int = 128,
) -> Image.Image:
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-6), 0, 1)
    rgb = (matplotlib.colormaps.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    return img.resize((size, size), Image.BILINEAR) if size != arr.shape[0] else img


# ─────────────────────────────────────────────────────────────────────────────
# Architecture (vendored from pretrain_ijepa_s2_fixed.py)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed13(nn.Module):
    """13-channel patch embedding for Sentinel-2."""

    def __init__(self, img_size: int = 64, patch_size: int = 8, embed_dim: int = 1024):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(13, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)


class S2Encoder(nn.Module):
    """ViT-Large/8 Sentinel-2 encoder. Default: embed=1024, depth=24, heads=16."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        in_chans: int = 13,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed13(img_size, patch_size, embed_dim)
        N = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, embed_dim))
        self.embed_dim = embed_dim
        self.num_patches = N
        self.patch_size = patch_size
        self.grid_size = self.patch_embed.grid_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        tok = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        if mask is not None:
            tok = tok[:, mask, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tok = torch.cat([cls, tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def encode_cls(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 0]

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, 1:]

    @torch.no_grad()
    def get_all_layer_attentions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: (depth, num_heads, N+1, N+1) softmaxed attention for every block.

        Single forward pass, attention extracted from each block's qkv projection.
        Expects batch size 1; the panel functions only ever score one tile.
        """
        B = x.shape[0]
        if B != 1:
            raise ValueError("get_all_layer_attentions expects batch=1")

        tok = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tok = torch.cat([cls, tok], dim=1)

        all_attns = []
        for blk in self.blocks:
            B_, N_, D_ = tok.shape
            qkv = blk.attn.qkv(tok)
            qkv = qkv.reshape(B_, N_, 3, blk.attn.num_heads, D_ // blk.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)         # (3, B, H, N, head_dim)
            q, k, _v = qkv.unbind(0)
            scale = (D_ // blk.attn.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)              # (B, H, N+1, N+1)
            all_attns.append(attn[0])                # (H, N+1, N+1)
            tok = blk(tok)

        return torch.stack(all_attns, dim=0)         # (depth, H, N+1, N+1)


# ─────────────────────────────────────────────────────────────────────────────
# Loader — local path -> $JEPA_CKPT -> HF Hub auto-download
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "backends") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jepa_encoder(
    ckpt_path: Optional[str] = None,
    device: str = "auto",
    hf_repo: str = "Siddharth63/forestWHY-JEPA-vitl",
    hf_filename: str = "s2_ijepa_gee_vitl_full_encoder_final.pt",
) -> S2Encoder:
    """Load the I-JEPA Sentinel-2 ViT-L/8 encoder.

    Resolution order:
      1. `ckpt_path` argument
      2. `JEPA_CKPT` env var
      3. HuggingFace Hub: `hf_hub_download(hf_repo, hf_filename)`
         (overridable via `JEPA_HF_REPO` and `JEPA_HF_FILENAME` env vars)

    Returns an `S2Encoder` in eval mode, on the resolved device, parameters frozen.
    Raises `RuntimeError` (not `sys.exit`) so callers can recover.
    """
    device = _resolve_device(device)

    # 1. argument
    resolved: Optional[str] = ckpt_path

    # 2. env var
    if not resolved:
        env_path = os.environ.get("JEPA_CKPT", "").strip()
        if env_path:
            resolved = env_path

    # 3. HF Hub
    if not resolved:
        repo = os.environ.get("JEPA_HF_REPO", hf_repo)
        filename = os.environ.get("JEPA_HF_FILENAME", hf_filename)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub not installed; pip install huggingface_hub") from exc
        try:
            log.info("Downloading JEPA encoder %s/%s from HF Hub ...", repo, filename)
            resolved = hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download JEPA encoder from HF Hub ({repo}/{filename}). "
                f"Set JEPA_CKPT to a local .pt path or run `huggingface-cli login` "
                f"if the repo is private. Original error: {exc}"
            ) from exc

    if not Path(resolved).exists():
        raise RuntimeError(f"JEPA checkpoint not found at: {resolved}")

    log.info("Loading JEPA encoder from %s on %s ...", resolved, device)
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    embed_dim = cfg.get("embed_dim", ckpt.get("embed_dim", 1024) if isinstance(ckpt, dict) else 1024)
    depth = cfg.get("depth", ckpt.get("depth", 24) if isinstance(ckpt, dict) else 24)
    num_heads = cfg.get("num_heads", ckpt.get("num_heads", 16) if isinstance(ckpt, dict) else 16)

    encoder = S2Encoder(embed_dim=embed_dim, depth=depth, num_heads=num_heads).to(device)

    state = ckpt["encoder"] if isinstance(ckpt, dict) and "encoder" in ckpt else ckpt
    encoder.load_state_dict(state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Sanity check on a dummy patch
    with torch.no_grad():
        dummy = torch.zeros(1, 13, 64, 64, device=device)
        attns = encoder.get_all_layer_attentions(dummy)
        patches = encoder.encode_patches(dummy)
        N = encoder.num_patches
        if tuple(attns.shape) != (depth, num_heads, N + 1, N + 1):
            raise RuntimeError(
                f"get_all_layer_attentions shape mismatch: got {tuple(attns.shape)}, "
                f"expected ({depth}, {num_heads}, {N + 1}, {N + 1})"
            )
        if tuple(patches.shape) != (1, N, embed_dim):
            raise RuntimeError(
                f"encode_patches shape mismatch: got {tuple(patches.shape)}, "
                f"expected (1, {N}, {embed_dim})"
            )

    log.info(
        "JEPA encoder ready: embed=%d depth=%d heads=%d device=%s",
        encoder.embed_dim, len(encoder.blocks), encoder.blocks[0].attn.num_heads, device,
    )
    return encoder


# ─────────────────────────────────────────────────────────────────────────────
# JEPA panels (8 total: 2 spectral diffs reused by spectral.py + 6 JEPA-specific)
# ─────────────────────────────────────────────────────────────────────────────

def make_jepa_panels(
    before: np.ndarray,
    after: np.ndarray,
    encoder: S2Encoder,
    device: str,
    size: int = 128,
) -> Dict[str, Image.Image]:
    """Generate the 8 differential panels driven by the JEPA encoder.

    Inputs are float32 arrays of shape (13, 64, 64), values in [0, 1].

    Returned dict keys:
        delta_ndvi, delta_nbr,
        attention_multi, embedding_change, cropa_roads,
        delta_attn_role, head_disagreement, pca_semantic
    """
    panels: Dict[str, Image.Image] = {}
    grid = encoder.grid_size

    def prep(p: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(p.astype(np.float32))
        t = torch.clamp(t, 0.0, 1.0)
        t = (t - torch.from_numpy(S2_MEAN)[:, None, None]) / (
            torch.from_numpy(S2_STD)[:, None, None] + 1e-6
        )
        return t.unsqueeze(0).to(device)

    b_in, a_in = prep(before), prep(after)

    with torch.no_grad():
        b_attns = encoder.get_all_layer_attentions(b_in)   # (depth, H, N+1, N+1)
        a_attns = encoder.get_all_layer_attentions(a_in)
        b_emb = encoder.encode_patches(b_in)[0]            # (N, D)
        a_emb = encoder.encode_patches(a_in)[0]

    depth = b_attns.shape[0]
    third = max(1, depth // 3)

    def ndvi_arr(p: np.ndarray) -> np.ndarray:
        return (p[BAND_IDX["B8"]] - p[BAND_IDX["B4"]]) / (
            p[BAND_IDX["B8"]] + p[BAND_IDX["B4"]] + 1e-6
        )

    def nbr_arr(p: np.ndarray) -> np.ndarray:
        return (p[BAND_IDX["B8"]] - p[BAND_IDX["B11"]]) / (
            p[BAND_IDX["B8"]] + p[BAND_IDX["B11"]] + 1e-6
        )

    def norm_ch(x: np.ndarray) -> np.ndarray:
        return np.clip((x - x.min()) / (x.max() - x.min() + 1e-6), 0, 1)

    # 1. ΔNDVI — vegetation change magnitude
    panels["delta_ndvi"] = colorise(
        ndvi_arr(after) - ndvi_arr(before), "RdYlGn", -0.5, 0.5, size
    )

    # 2. ΔNBR — biomass / fire change
    panels["delta_nbr"] = colorise(
        nbr_arr(after) - nbr_arr(before), "RdYlGn", -0.5, 0.5, size
    )

    # 3. Multi-scale attention (R fine, G mid, B landscape) on the BEFORE tile
    cls_b = b_attns[:, :, 0, 1:]   # (depth, H, N) CLS->patch
    def layer_mean(layers: torch.Tensor) -> np.ndarray:
        return layers.mean(dim=(0, 1)).cpu().numpy().reshape(grid, grid)

    attn_rgb = np.stack([
        norm_ch(layer_mean(cls_b[:third])),
        norm_ch(layer_mean(cls_b[third:2 * third])),
        norm_ch(layer_mean(cls_b[2 * third:])),
    ], axis=-1)
    img = Image.fromarray((attn_rgb * 255).astype(np.uint8))
    panels["attention_multi"] = img.resize((size, size), Image.NEAREST)

    # 4. Embedding change — cosine distance per patch
    b_n = F.normalize(b_emb, dim=-1).cpu().numpy()
    a_n = F.normalize(a_emb, dim=-1).cpu().numpy()
    cos_dist = 1 - (b_n * a_n).sum(axis=-1).reshape(grid, grid)
    panels["embedding_change"] = colorise(cos_dist, "RdYlGn_r", size=size)

    # 5. CroPA — cross-patch correlation, hot = linear structures
    tok = F.normalize(a_emb, dim=-1).reshape(grid, grid, -1)
    sim = torch.zeros(grid, grid)
    sim[:, :-1] += (tok[:, :-1] * tok[:, 1:]).sum(-1).cpu()
    sim[:, 1:] += (tok[:, :-1] * tok[:, 1:]).sum(-1).cpu()
    sim[:-1] += (tok[:-1] * tok[1:]).sum(-1).cpu()
    sim[1:] += (tok[:-1] * tok[1:]).sum(-1).cpu()
    panels["cropa_roads"] = colorise((sim / 4).numpy(), "hot", size=size)

    # 6. Δ attention role — patches that gained/lost CLS importance
    b_cls_mean = b_attns[:, :, 0, 1:].mean(dim=(0, 1)).cpu().numpy().reshape(grid, grid)
    a_cls_mean = a_attns[:, :, 0, 1:].mean(dim=(0, 1)).cpu().numpy().reshape(grid, grid)
    panels["delta_attn_role"] = colorise(a_cls_mean - b_cls_mean, "RdBu_r", size=size)

    # 7. Head disagreement (after) — std across attention heads at the last layer
    head_attn = a_attns[-1, :, 0, 1:].cpu().numpy()       # (H, N)
    head_std = head_attn.std(axis=0).reshape(grid, grid)
    panels["head_disagreement"] = colorise(head_std, "hot", size=size)

    # 8. PCA semantic clusters — left = before, right = after
    try:
        from sklearn.decomposition import PCA

        def pca_rgb(emb_tensor: torch.Tensor) -> np.ndarray:
            e = emb_tensor.cpu().numpy()
            pca = PCA(n_components=3, random_state=0)
            rgb = pca.fit_transform(e)
            rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-6)
            return (rgb.reshape(grid, grid, 3) * 255).astype(np.uint8)

        combined = Image.new("RGB", (grid * 2, grid))
        combined.paste(Image.fromarray(pca_rgb(b_emb)), (0, 0))
        combined.paste(Image.fromarray(pca_rgb(a_emb)), (grid, 0))
        panels["pca_semantic"] = combined.resize((size * 2, size), Image.NEAREST)
    except ImportError:
        b_norm = b_emb.norm(dim=-1).cpu().numpy().reshape(grid, grid)
        a_norm = a_emb.norm(dim=-1).cpu().numpy().reshape(grid, grid)
        panels["pca_semantic"] = colorise(a_norm - b_norm, "coolwarm", size=size)

    return panels
