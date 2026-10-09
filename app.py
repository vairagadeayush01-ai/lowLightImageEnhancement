"""
app.py  –  Low-Light Image Enhancement  (Streamlit)
=====================================================
Model  : UNet (base=32, ReLU, no norm) trained on LoL dataset
File   : unet_lowlight_best.pth

Features
--------
- Upload your own image  OR
- Pick a pre-fetched LoL test image (low-light + optional ground-truth)
- Compare all classical + CNN methods side-by-side
- PSNR / SSIM metrics when ground-truth is available
- Download enhanced result
"""

import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ── Auto-fetch test images (runs once on first startup / cloud deploy) ──────
@st.cache_resource(show_spinner="📥 Downloading LoL test images (first run only)…")
def _ensure_test_images():
    """Download LoL test images from HuggingFace if not already present."""
    from fetching_test import fetch_test_images
    fetch_test_images()

_ensure_test_images()

# ──────────────────────────────────────────────────────────────
# Model definition  (matches unet_lowlight_best.pth exactly)
# Conv2d → InstanceNorm2d(affine=False) → LeakyReLU  ×2 per block
# ──────────────────────────────────────────────────────────────
def conv_block(inc: int, outc: int) -> nn.Sequential:
    # InstanceNorm2d(affine=False) has no learnable params but occupies an index slot,
    # shifting the 2nd Conv2d to index 3 — which is exactly how unet_lowlight_best.pth was saved.
    return nn.Sequential(
        nn.Conv2d(inc,  outc, 3, padding=1), nn.InstanceNorm2d(outc), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(outc, outc, 3, padding=1), nn.InstanceNorm2d(outc), nn.LeakyReLU(0.2, inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        # Encoder
        self.e1 = conv_block(3,  c1)
        self.e2 = conv_block(c1, c2)
        self.e3 = conv_block(c2, c3)
        # Bottleneck
        self.b  = conv_block(c3, c4)
        # Decoder
        self.u3 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.d3 = conv_block(c3 * 2, c3)
        self.u2 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.d2 = conv_block(c2 * 2, c2)
        self.u1 = nn.ConvTranspose2d(c2, c1, 2, 2)
        self.d1 = conv_block(c1 * 2, c1)
        self.out = nn.Conv2d(c1, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b  = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b),  e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return (x + self.out(d1)).clamp(0, 1)


# ──────────────────────────────────────────────────────────────
# Model loading (cached so it only runs once per session)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model() -> nn.Module:
    model = UNet(base=32)
    ckpt  = torch.load("unet_lowlight_best.pth", map_location="cpu")
    # Support both plain state-dict and wrapped checkpoints
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    return model.eval()


# ──────────────────────────────────────────────────────────────
# Enhancement methods
# ──────────────────────────────────────────────────────────────
def _gamma(img: np.ndarray, g: float) -> np.ndarray:
    return (255.0 * (img / 255.0) ** g).clip(0, 255).astype(np.uint8)


def _clahe(img: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@torch.no_grad()
def _cnn(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # Mild gamma pre-processing before CNN
    pre = _gamma(img, 0.4)
    x   = torch.from_numpy(pre).permute(2, 0, 1)[None].float().div(255)
    # Pad so spatial dims are multiples of 8 (3 pooling levels)
    ph, pw = (-h) % 8, (-w) % 8
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    out = load_model()(x)[0, :, :h, :w].permute(1, 2, 0).cpu().numpy()
    return (out * 255).clip(0, 255).astype(np.uint8)


METHODS: dict = {
    "CNN (U-Net)":    _cnn,
    "Gamma":          lambda x: _gamma(x, 0.30),
    "CLAHE":          lambda x: _clahe(x, clip=3.0, grid=8),
    "Gamma + CLAHE":  lambda x: _clahe(_gamma(x, 0.35), clip=2.0, grid=6),
}


# ──────────────────────────────────────────────────────────────
# Test-image helpers
# ──────────────────────────────────────────────────────────────
TEST_LOW  = Path("test_images/low")
TEST_HIGH = Path("test_images/high")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _get_test_images() -> list[str]:
    """Return sorted list of filenames available in the low-light test folder."""
    if not TEST_LOW.exists():
        return []
    return sorted(f.name for f in TEST_LOW.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resize_max(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    m = max(img.shape[:2])
    if m <= max_dim:
        return img
    scale = max_dim / m
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _png_bytes(img_bgr: np.ndarray) -> bytes:
    return cv2.imencode(".png", img_bgr)[1].tobytes()


# ──────────────────────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Low-Light Image Enhancement",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global typography & background ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #141429 40%, #1a1a3e 100%);
    color: #e2e8f0;
}

/* ── Hero header ─────────────────────────────────────────────── */
.hero {
    background: linear-gradient(90deg, #6c63ff 0%, #3ec6f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 2px;
    margin-bottom: 24px;
}

/* ── Cards ───────────────────────────────────────────────────── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 18px;
    backdrop-filter: blur(8px);
}

/* ── Metric badges ───────────────────────────────────────────── */
.metric-badge {
    display: inline-block;
    background: linear-gradient(90deg,#6c63ff33,#3ec6f533);
    border: 1px solid #6c63ff55;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    margin: 2px 4px;
}

/* ── Image caption pill ──────────────────────────────────────── */
.img-caption {
    text-align: center;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 6px;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(15,12,41,0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #cbd5e1; }

/* ── Buttons ─────────────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(90deg, #6c63ff, #3ec6f5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s;
}
.stDownloadButton > button:hover { opacity: 0.88; }

/* Tab selected indicator */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid #6c63ff;
    color: #a5b4fc;
}

/* Spinner */
.stSpinner > div { border-top-color: #6c63ff !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(108,99,255,0.06) !important;
    border: 1px dashed #6c63ff88 !important;
    border-radius: 12px !important;
}

/* Test image thumbnail grid */
.thumb-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
.thumb-item { text-align: center; cursor: pointer; }
.thumb-item img { border-radius: 8px; border: 2px solid transparent; transition: border-color 0.2s; }
.thumb-item img:hover { border-color: #6c63ff; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Sidebar – controls
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    method = st.selectbox(
        "Enhancement Method",
        list(METHODS.keys()),
        help="CNN U-Net is the deep-learning model trained on the LoL dataset.",
    )
    compare_all = st.checkbox("🔬 Compare all methods", value=False)

    st.markdown("---")
    st.markdown("### 🌑 Simulate darkness")
    darken = st.slider("Darkness factor (1.0 = off)", 0.05, 1.0, 1.0, 0.05,
                       help="Artificially darken the input to stress-test enhancement.")

    st.markdown("---")
    st.markdown("### 📐 Metrics")
    gt_file = st.file_uploader(
        "Upload ground-truth image (optional)",
        type=["png", "jpg", "jpeg"],
        help="If provided, PSNR and SSIM will be computed against this reference.",
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;'>"
        "Model: <b>UNet base=32</b><br>"
        "Dataset: <b>LoL (485 train / 15 test)</b><br>"
        "Source: <a style='color:#6c63ff' href='https://huggingface.co/datasets/geekyrakshit/LoL-Dataset' target='_blank'>HuggingFace 🤗</a>"
        "</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────
# Main header
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="hero">🌙 Low-Light Image Enhancement</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload a dark image <b>or</b> pick a LoL test sample — '
    'compare classical & deep-learning enhancement methods side-by-side.</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Input tabs: Upload  |  Test images
# ──────────────────────────────────────────────────────────────
tab_upload, tab_test = st.tabs(["📤 Upload Image", "🗂️ Dataset Test Images"])

img_bgr: np.ndarray | None = None
gt_bgr:  np.ndarray | None = None
source_label = ""

# ─── Tab 1: Upload ────────────────────────────────────────────
with tab_upload:
    st.markdown(
        '<div class="card">Upload a low-light PNG/JPG image from your device. '
        'Optionally provide a ground-truth image via the sidebar to compute PSNR & SSIM.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a low-light image",
        type=["png", "jpg", "jpeg"],
        key="upload_file",
    )
    if uploaded is not None:
        data   = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        source_label = f"Uploaded: {uploaded.name}"


# ─── Tab 2: Test images from dataset ─────────────────────────
with tab_test:
    test_names = _get_test_images()

    if not test_names:
        st.warning(
            "⚠️ No test images found.\n\n"
            "Run **`python fetching_test.py`** once to download the LoL dataset test split. "
            "Images will be saved to `test_images/low/` and `test_images/high/`."
        )
    else:
        st.success(f"✅ {len(test_names)} test image(s) available from the LoL dataset.")
        st.markdown(
            "<div style='color:#64748b;font-size:0.82rem;margin-bottom:8px;'>"
            "Source: "
            "<a href='https://huggingface.co/datasets/geekyrakshit/LoL-Dataset' "
            "target='_blank' style='color:#6c63ff;'>geekyrakshit/LoL-Dataset on HuggingFace 🤗</a>"
            "</div>",
            unsafe_allow_html=True,
        )

        # ── Clickable image card gallery ───────────────────────
        st.markdown("#### 🖼️ Click an image to select it for enhancement")

        # CSS for the clickable cards
        st.markdown("""
        <style>
        /* Card wrapper */
        div[data-testid="column"] .img-card-btn {
            width: 100%;
        }
        /* Make the "select" button look invisible — the image IS the button */
        .select-btn > button {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            width: 100% !important;
            cursor: pointer !important;
            transition: transform 0.15s ease;
        }
        .select-btn > button:hover {
            transform: scale(1.03);
        }
        /* Selected card glow */
        .card-selected {
            border: 2px solid #6c63ff !important;
            border-radius: 12px;
            box-shadow: 0 0 14px #6c63ff88;
        }
        </style>
        """, unsafe_allow_html=True)

        THUMB_W  = 140
        N_COLS   = 5
        selected = st.session_state.get("selected_test")

        for row_start in range(0, len(test_names), N_COLS):
            row_names = test_names[row_start : row_start + N_COLS]
            cols = st.columns(N_COLS)
            for col, name in zip(cols, row_names):
                path_low  = TEST_LOW / name
                thumb_bgr = _load_bgr(path_low)
                if thumb_bgr is None:
                    continue

                # Resize thumbnail (keep aspect ratio, fixed width)
                h_orig, w_orig = thumb_bgr.shape[:2]
                thumb_h = int(THUMB_W * h_orig / w_orig)
                thumb_rgb = cv2.resize(
                    _bgr_to_rgb(thumb_bgr),
                    (THUMB_W, thumb_h),
                    interpolation=cv2.INTER_AREA,
                )

                is_selected = (name == selected)

                with col:
                    # Glow border when selected
                    border_style = (
                        "border:2px solid #6c63ff; border-radius:12px; "
                        "box-shadow:0 0 16px #6c63ff99; padding:3px;"
                        if is_selected else
                        "border:2px solid rgba(255,255,255,0.08); border-radius:12px; padding:3px;"
                    )
                    st.markdown(
                        f"<div style='{border_style}'>",
                        unsafe_allow_html=True,
                    )
                    st.image(thumb_rgb, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Label + select button
                    label = f"✅ {name}" if is_selected else name
                    if st.button(label, key=f"card_{name}", use_container_width=True):
                        st.session_state["selected_test"] = name
                        st.rerun()

        # ── Info for currently selected image ──────────────────
        sel = st.session_state.get("selected_test")
        if sel:
            low_path  = TEST_LOW  / sel
            high_path = TEST_HIGH / sel

            st.markdown("---")
            col_link1, col_link2 = st.columns(2)
            with col_link1:
                st.markdown(f"📁 **Low-light:** `{low_path}`")
            with col_link2:
                if high_path.exists():
                    st.markdown(f"📁 **Ground-truth:** `{high_path}`")

            # Load images into pipeline
            loaded = _load_bgr(low_path)
            if loaded is not None:
                img_bgr      = loaded
                source_label = f"LoL test: {sel}"
                if high_path.exists() and gt_file is None:
                    gt_bgr = _load_bgr(high_path)
                    st.info(f"📐 Ground-truth auto-loaded — PSNR & SSIM will be computed.")
        else:
            st.markdown(
                "<div style='color:#64748b;font-size:0.85rem;margin-top:16px;'>"
                "👆 Click any image above to select it."
                "</div>",
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────
# Load GT from sidebar upload (overrides auto-loaded GT)
# ──────────────────────────────────────────────────────────────
if gt_file is not None:
    data   = np.frombuffer(gt_file.getvalue(), dtype=np.uint8)
    gt_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)

# ──────────────────────────────────────────────────────────────
# Enhancement pipeline
# ──────────────────────────────────────────────────────────────
if img_bgr is not None:
    # Resize very large images
    img_bgr = _resize_max(img_bgr, max_dim=900)

    # Apply darkness simulation
    if darken < 1.0:
        img_bgr = (img_bgr.astype(np.float32) * darken).clip(0, 255).astype(np.uint8)

    # Resize GT to match
    if gt_bgr is not None:
        gt_bgr = cv2.resize(gt_bgr, (img_bgr.shape[1], img_bgr.shape[0]))

    # ── Helper: run one method ────────────────────────────────
    def run_method(name: str) -> tuple[np.ndarray, str]:
        t0  = time.perf_counter()
        out = METHODS[name](img_bgr)
        ms  = (time.perf_counter() - t0) * 1000
        caption = f"**{name}** — {ms:.0f} ms"
        if gt_bgr is not None:
            p = psnr(gt_bgr, out, data_range=255)
            s = ssim(gt_bgr, out, channel_axis=2, data_range=255)
            caption += f"  |  PSNR `{p:.2f} dB`  |  SSIM `{s:.3f}`"
        return out, caption

    st.markdown("---")
    st.markdown(f"### 🔎 Results  <span style='font-size:0.8rem;color:#64748b;'>({source_label})</span>",
                unsafe_allow_html=True)

    # ── Compare all methods — 2×2 subplot grid ───────────────
    if compare_all:
        all_methods = list(METHODS.keys())   # 4 methods

        # Pre-run all methods
        results = {}
        for mname in all_methods:
            with st.spinner(f"Running {mname}…"):
                results[mname] = run_method(mname)

        # ── Layout: Input (left half) | 2×2 grid (right half) ─
        left, right = st.columns([1, 2], gap="medium")

        with left:
            st.markdown(
                "<div style='background:rgba(255,255,255,0.04);border:1px solid "
                "rgba(255,255,255,0.08);border-radius:14px;padding:10px;'>"
                "<p style='color:#94a3b8;font-size:0.78rem;text-align:center;"
                "margin-bottom:6px;'>📷 Input (low-light)</p>",
                unsafe_allow_html=True,
            )
            st.image(_bgr_to_rgb(img_bgr), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if gt_bgr is not None:
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                st.markdown(
                    "<div style='background:rgba(255,255,255,0.04);border:1px solid "
                    "rgba(255,255,255,0.08);border-radius:14px;padding:10px;'>"
                    "<p style='color:#94a3b8;font-size:0.78rem;text-align:center;"
                    "margin-bottom:6px;'>✅ Ground-truth (high-light)</p>",
                    unsafe_allow_html=True,
                )
                st.image(_bgr_to_rgb(gt_bgr), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            # 2 rows × 2 columns
            for row in range(2):
                r_cols = st.columns(2, gap="small")
                for col_idx in range(2):
                    mname = all_methods[row * 2 + col_idx]
                    out_bgr, cap = results[mname]
                    with r_cols[col_idx]:
                        st.markdown(
                            "<div style='background:rgba(255,255,255,0.04);border:1px solid "
                            "rgba(255,255,255,0.08);border-radius:14px;padding:10px;"
                            "margin-bottom:10px;'>",
                            unsafe_allow_html=True,
                        )
                        st.image(_bgr_to_rgb(out_bgr), use_container_width=True)
                        st.markdown(
                            f'<p class="img-caption" style="margin-top:6px;">{cap}</p>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

    # ── Single method view ────────────────────────────────────
    else:
        with st.spinner(f"Running **{method}**…"):
            out_bgr, cap = run_method(method)

        col_in, col_out = st.columns(2)
        with col_in:
            st.image(_bgr_to_rgb(img_bgr), caption="Input (low-light)", use_container_width=True)

        with col_out:
            st.image(_bgr_to_rgb(out_bgr), use_container_width=True)
            st.markdown(f'<p class="img-caption">{cap}</p>', unsafe_allow_html=True)

        if gt_bgr is not None:
            with st.expander("📷 Ground-truth reference"):
                st.image(_bgr_to_rgb(gt_bgr), caption="Ground-truth (high-light)", use_container_width=True)

        # Download
        st.markdown("")
        st.download_button(
            label="⬇️ Download enhanced image",
            data=_png_bytes(out_bgr),
            file_name=f"enhanced_{method.replace(' ','_').replace('(','').replace(')','')}.png",
            mime="image/png",
        )

else:
    # Landing placeholder
    st.markdown(
        '<div class="card" style="text-align:center;padding:48px 24px;">'
        '<div style="font-size:3rem;">🌙</div>'
        '<div style="font-size:1.1rem;color:#94a3b8;margin-top:12px;">'
        'Select the <b>Upload Image</b> tab to upload a photo,<br>'
        'or the <b>Dataset Test Images</b> tab to pick a LoL test sample.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )