# app.py
import time
from datetime import datetime

import numpy as np
import cv2
import streamlit as st

from segmentation import bgr_from_bytes, resize_keep_aspect, run_segmentation

# Optional: for a simple chart (built-in, no extra deps)
# Streamlit can chart from dict/list directly.

st.set_page_config(page_title="Aplikasi Segmentasi Citra", layout="wide")


# -----------------------------
# Session state init
# -----------------------------
if "history" not in st.session_state:
    # Each item: dict(ts, algo, params, img_shape, fg_pixels, fg_ratio, runtime_ms)
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # store last segmentation outputs (mask/overlay/meta)


# -----------------------------
# Sidebar: navigation + global settings
# -----------------------------
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Halaman", ["Dashboard", "Segmentation"], index=1)

st.sidebar.divider()
st.sidebar.subheader("Pengaturan Global")
max_side = st.sidebar.slider("Resize maksimum (untuk performa)", 400, 1600, 900, 50)
alpha = st.sidebar.slider("Alpha overlay", 0.0, 1.0, 0.5, 0.05)


def _mask_stats(mask01: np.ndarray):
    fg = int(mask01.sum())
    total = int(mask01.size)
    return fg, total, (fg / total if total else 0.0)


def _push_history(algo_key: str, params: dict, img_shape, mask01: np.ndarray, runtime_ms: float):
    fg, total, ratio = _mask_stats(mask01)
    st.session_state.history.append({
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algo": algo_key,
        "params": {k: v for k, v in params.items() if k != "weights_bytes"},  # don't store raw bytes
        "img_shape": tuple(img_shape),
        "fg_pixels": fg,
        "total_pixels": total,
        "fg_ratio": float(ratio),
        "runtime_ms": float(runtime_ms),
    })


# -----------------------------
# DASHBOARD PAGE
# -----------------------------
if page == "Dashboard":
    st.title("Dashboard")
    st.caption("Ringkasan penggunaan aplikasi, histori segmentasi, dan statistik hasil per sesi.")

    c1, c2, c3, c4 = st.columns(4)

    total_runs = len(st.session_state.history)
    last = st.session_state.history[-1] if total_runs else None

    c1.metric("Total runs (session)", total_runs)
    c2.metric("Algoritma terakhir", last["algo"] if last else "-")
    c3.metric("Runtime terakhir (ms)", f'{last["runtime_ms"]:.1f}' if last else "-")
    c4.metric("FG ratio terakhir", f'{last["fg_ratio"]:.3f}' if last else "-")

    st.divider()

    # Show last result previews
    st.subheader("Hasil Terakhir")
    if st.session_state.last_result is None:
        st.info("Belum ada proses segmentasi pada sesi ini. Buka halaman Segmentation untuk memulai.")
    else:
        lr = st.session_state.last_result
        img_rgb = lr["img_rgb"]
        overlay_rgb = lr["overlay_rgb"]
        mask_vis = lr["mask_vis"]
        meta = lr["meta"]
        fg, total, ratio = _mask_stats(lr["mask01"])

        col1, col2, col3 = st.columns([1.2, 1.2, 1])
        with col1:
            st.caption("Gambar Asli")
            st.image(img_rgb, use_container_width=True)
        with col2:
            st.caption("Overlay")
            st.image(overlay_rgb, use_container_width=True)
            st.write("Meta:", meta)
        with col3:
            st.caption("Mask (B/W)")
            st.image(mask_vis, clamp=True, use_container_width=True)
            st.metric("Foreground pixels", fg)
            st.metric("Foreground ratio", f"{ratio:.3f}")

    st.divider()

    # History table + simple analytics
    st.subheader("Histori Segmentasi (Session)")
    if total_runs == 0:
        st.info("Histori masih kosong.")
    else:
        # Show table
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Analitik Sederhana")

        # Aggregate by algorithm
        algo_counts = {}
        avg_runtime = {}
        avg_ratio = {}

        for row in st.session_state.history:
            a = row["algo"]
            algo_counts[a] = algo_counts.get(a, 0) + 1

        for a in algo_counts:
            rt = [r["runtime_ms"] for r in st.session_state.history if r["algo"] == a]
            rr = [r["fg_ratio"] for r in st.session_state.history if r["algo"] == a]
            avg_runtime[a] = float(np.mean(rt)) if rt else 0.0
            avg_ratio[a] = float(np.mean(rr)) if rr else 0.0

        colA, colB = st.columns(2)
        with colA:
            st.caption("Jumlah run per algoritma")
            st.bar_chart(algo_counts)
        with colB:
            st.caption("Rata-rata runtime (ms) per algoritma")
            st.bar_chart(avg_runtime)

        st.caption("Rata-rata foreground ratio per algoritma")
        st.bar_chart(avg_ratio)

        st.divider()
        if st.button("Clear histori session", type="secondary"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.success("Histori session dibersihkan.")

    st.stop()


# -----------------------------
# SEGMENTATION PAGE
# -----------------------------
st.title("Segmentation")
st.caption("Upload gambar → pilih algoritma → atur parameter → lihat mask & overlay → download output.")

uploaded = st.sidebar.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])

algo = st.sidebar.selectbox(
    "Pilih algoritma",
    [
        "otsu",
        "kmeans",
        "watershed",
        "deeplabv3_resnet50 (ResNet backbone)",
        "fcn_resnet50 (ResNet backbone)",
        "unet (SMP, butuh weights)",
        "fpn (SMP, butuh weights)",
    ],
)

params = {"alpha": float(alpha)}
algo_key = algo.split(" ")[0]  # "deeplabv3_resnet50", "unet", etc.

# Algorithm-specific parameters
if algo_key == "otsu":
    st.sidebar.subheader("Parameter Otsu")
    params["blur_ksize"] = st.sidebar.slider("Gaussian blur kernel (ganjil)", 1, 31, 5, 2)
    params["invert"] = st.sidebar.checkbox("Invert hasil (foreground/background ditukar)", value=False)

elif algo_key == "kmeans":
    st.sidebar.subheader("Parameter K-Means")
    params["k"] = st.sidebar.slider("Jumlah cluster (k)", 2, 10, 3, 1)
    params["attempts"] = st.sidebar.slider("Attempts", 1, 20, 5, 1)
    fg = st.sidebar.number_input("Foreground cluster index (opsional, -1 = otomatis)", value=-1, min_value=-1, max_value=50, step=1)
    params["fg_cluster"] = None if fg == -1 else int(fg)

elif algo_key == "watershed":
    st.sidebar.subheader("Parameter Watershed")
    params["blur_ksize"] = st.sidebar.slider("Gaussian blur kernel (ganjil)", 1, 31, 5, 2)
    params["morph_ksize"] = st.sidebar.slider("Morph kernel (ganjil)", 1, 31, 3, 2)
    params["dist_thresh"] = st.sidebar.slider("Distance threshold (seed FG)", 0.05, 0.95, 0.40, 0.01)

elif algo_key in ["deeplabv3_resnet50", "fcn_resnet50"]:
    st.sidebar.subheader("Parameter DeepLab/FCN (Pretrained)")
    params["conf_thresh"] = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.50, 0.01)
    params["target_class"] = st.sidebar.number_input("Target class id (default 15=person VOC)", value=15, min_value=0, max_value=200, step=1)

    # Guard: show only CPU by default; CUDA appears only if torch says available
    try:
        import torch
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    except Exception:
        device_options = ["cpu"]
    params["device"] = st.sidebar.selectbox("Device", device_options, index=0)

elif algo_key in ["unet", "fpn"]:
    st.sidebar.subheader("UNet/FPN (SMP) – Binary Segmentation")
    st.sidebar.write("Wajib upload weights (.pth/.pt) hasil training Anda agar hasil benar.")
    params["encoder_name"] = st.sidebar.selectbox(
        "Encoder (backbone)",
        ["resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2"],
        index=0
    )
    params["conf_thresh"] = st.sidebar.slider("Threshold mask (sigmoid)", 0.0, 1.0, 0.50, 0.01)

    try:
        import torch
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    except Exception:
        device_options = ["cpu"]
    params["device"] = st.sidebar.selectbox("Device", device_options, index=0)

    w = st.sidebar.file_uploader("Upload weights UNet/FPN (.pth/.pt)", type=["pth", "pt"])
    params["weights_bytes"] = w.read() if w is not None else None


if not uploaded:
    st.info("Silakan upload gambar terlebih dahulu.")
    st.stop()

img_bgr = bgr_from_bytes(uploaded.read())
img_bgr = resize_keep_aspect(img_bgr, max_side=max_side)

# Guard for unet/fpn missing weights
if algo_key in ["unet", "fpn"] and not params.get("weights_bytes"):
    st.warning("UNet/FPN dipilih tetapi weights belum diupload. Upload file .pth/.pt di sidebar.")
    st.stop()

# Run segmentation with timing
t0 = time.perf_counter()
with st.spinner("Menjalankan segmentasi..."):
    try:
        result = run_segmentation(img_bgr, algo_key, params=params)
    except Exception as e:
        st.error(f"Gagal melakukan segmentasi: {e}")
        st.stop()
runtime_ms = (time.perf_counter() - t0) * 1000.0

mask01 = result["mask01"]
overlay_bgr = result["overlay"]
meta = result["meta"]

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
mask_vis = (mask01 * 255).astype(np.uint8)

# Save last result for dashboard
st.session_state.last_result = {
    "img_rgb": img_rgb,
    "overlay_rgb": overlay_rgb,
    "mask_vis": mask_vis,
    "mask01": mask01,
    "meta": meta,
}

# Push history entry
_push_history(
    algo_key=algo_key,
    params=params,
    img_shape=img_bgr.shape,
    mask01=mask01,
    runtime_ms=runtime_ms,
)

col1, col2, col3 = st.columns([1.2, 1.2, 1])

with col1:
    st.subheader("Gambar Asli")
    st.image(img_rgb, use_container_width=True)

with col2:
    st.subheader("Overlay Mask")
    st.image(overlay_rgb, use_container_width=True)
    st.caption(f"Meta: {meta}")
    st.caption(f"Runtime: {runtime_ms:.1f} ms")

with col3:
    st.subheader("Mask (B/W)")
    st.image(mask_vis, clamp=True, use_container_width=True)
    fg, total, ratio = _mask_stats(mask01)
    st.metric("Foreground pixels", fg)
    st.metric("Foreground ratio", f"{ratio:.3f}")

st.divider()
st.subheader("Download Output")

overlay_png = cv2.imencode(".png", cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
mask_png = cv2.imencode(".png", mask_vis)[1].tobytes()

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Overlay (PNG)", overlay_png, file_name="overlay.png", mime="image/png")
with c2:
    st.download_button("Download Mask (PNG)", mask_png, file_name="mask.png", mime="image/png")

st.info("Tip: Buka halaman Dashboard untuk melihat ringkasan dan histori run.")
