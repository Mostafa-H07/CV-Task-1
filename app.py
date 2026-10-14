import streamlit as st
import cv2 as cv
import numpy as np
import ctypes
import os
import hashlib
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

# =====================================================================
# C++ ENGINE LINKING
# =====================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

# Keep DLL search directories alive while the application is running
dll_directories = []

if os.name == "nt":
    # OpenCV DLL directory
    opencv_bin = r"C:\opencv\opencv\build\x64\vc16\bin"
    if os.path.isdir(opencv_bin):
        dll_directories.append(os.add_dll_directory(opencv_bin))

    # Optional MinGW DLL directory
    mingw_bin = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(mingw_bin):
        dll_directories.append(os.add_dll_directory(mingw_bin))

dll_path = os.path.join(base_dir, "engine.dll")
if not os.path.isfile(dll_path):
    raise FileNotFoundError(f"engine.dll not found at: {dll_path}")

cpp = ctypes.CDLL(dll_path)

U8P = ctypes.POINTER(ctypes.c_uint8)
I32P = ctypes.POINTER(ctypes.c_int32)
F32P = ctypes.POINTER(ctypes.c_float)

# --- Main C++ pipeline (Tasks 1-7) ---
cpp.run_pipeline.argtypes = [
    U8P, ctypes.c_int, ctypes.c_int,               # Input
    ctypes.c_int, ctypes.c_float, ctypes.c_float,  # Noise
    ctypes.c_int, ctypes.c_int, ctypes.c_int,      # Filter
    ctypes.c_float, ctypes.c_float,                # Gaussian kernel
    U8P, U8P, I32P                                 # Noisy, output, histogram
]
cpp.run_pipeline.restype = None

# --- Histogram statistics + Task 8 ---
cpp.hist_to_pdf_cdf.argtypes = [I32P, F32P, F32P]
cpp.hist_to_pdf_cdf.restype = None
cpp.color_analysis.argtypes = [U8P, ctypes.c_int, ctypes.c_int, U8P, U8P, I32P, F32P, F32P]
cpp.color_analysis.restype = None

# --- Frequency-domain C++ API (Tasks 9 & 10) ---
cpp.fd_create.restype = ctypes.c_void_p
cpp.fd_create.argtypes = [U8P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
cpp.fd_free.argtypes = [ctypes.c_void_p]
cpp.fd_free.restype = None
cpp.fd_spectrum.argtypes = [ctypes.c_void_p, U8P]
cpp.fd_spectrum.restype = None
cpp.fd_filter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                          ctypes.c_int, ctypes.c_int, U8P, U8P]
cpp.fd_filter.restype = None
cpp.fd_hybrid.restype = ctypes.c_int
cpp.fd_hybrid.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                          ctypes.c_float, ctypes.c_float, ctypes.c_int, U8P]

NOISE_IDS = {"None": 0, "Uniform": 1, "Gaussian": 2, "Salt & Pepper": 3}
FILTER_IDS = {"None": 0, "Average": 1, "Gaussian": 2, "Median": 3,
              "Sobel": 4, "Prewitt": 5, "Roberts": 6, "Canny": 7,
              "Equalize": 8, "Normalize": 9}
KIND_IDS = {"Ideal": 0, "Gaussian": 1, "Butterworth": 2}
NO_NOISE = (0, 0.0, 0.0)


def _ptr(array, ptr_type):
    return array.ctypes.data_as(ptr_type)


def _p(array):
    return _ptr(array, U8P)


# =====================================================================
# PYTHON <-> C++ WRAPPERS (no image processing happens in Python)
# =====================================================================
def _distribution(hist):
    """Histogram -> {hist, pdf, cdf}; the PDF/CDF are computed in C++."""
    hist = np.ascontiguousarray(hist, dtype=np.int32)
    pdf = np.zeros(256, np.float32)
    cdf = np.zeros(256, np.float32)
    cpp.hist_to_pdf_cdf(_ptr(hist, I32P), _ptr(pdf, F32P), _ptr(cdf, F32P))
    return {"hist": hist, "pdf": pdf, "cdf": cdf}


@st.cache_data(show_spinner=False, max_entries=64)
def process(img, noise, filter_id, k_size=3, direction=0, g_mean=0.0, g_std=1.0):
    """Runs the C++ pipeline. Returns (noisy image, filtered image, distribution)."""
    img = np.ascontiguousarray(img)
    noisy = np.zeros_like(img)
    out = np.zeros_like(img)
    hist = np.zeros(256, dtype=np.int32)
    cpp.run_pipeline(
        _p(img), img.shape[0], img.shape[1],
        int(noise[0]), float(noise[1]), float(noise[2]),
        int(filter_id), int(k_size), int(direction),
        float(g_mean), float(g_std),
        _p(noisy), _p(out), _ptr(hist, I32P)
    )
    return noisy, out, _distribution(hist)


@st.cache_data(show_spinner=False, max_entries=8)
def color_analysis(bgr):
    """Task 8 in C++: gray image, R/G/B/Gray histograms + PDF + CDF, equalized gray."""
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    h, w = bgr.shape[:2]
    gray = np.empty((h, w), np.uint8)
    equalized = np.empty((h, w), np.uint8)
    hist = np.zeros((4, 256), np.int32)
    pdf = np.zeros((4, 256), np.float32)
    cdf = np.zeros((4, 256), np.float32)
    cpp.color_analysis(_p(bgr), h, w, _p(gray), _p(equalized),
                       _ptr(hist, I32P), _ptr(pdf, F32P), _ptr(cdf, F32P))
    stats = {name: {"hist": hist[i], "pdf": pdf[i], "cdf": cdf[i]}
             for i, name in enumerate(("R", "G", "B", "Gray"))}
    return gray, equalized, stats


class Spectrum:
    """Owns a C++ FreqDomain object. The FFT is computed once, in the constructor."""

    def __init__(self, img, shape):
        self.img = np.ascontiguousarray(img, dtype=np.uint8)
        self.shape = shape
        self.h = cpp.fd_create(_p(self.img), self.img.shape[0], self.img.shape[1],
                               shape[0], shape[1])
        if not self.h:
            raise RuntimeError("Could not create the frequency-domain object.")

    def __del__(self):
        try:
            if getattr(self, "h", None):
                cpp.fd_free(self.h)
                self.h = None
        except Exception:
            pass

    def view(self):
        out = np.empty(self.shape, np.uint8)
        cpp.fd_spectrum(self.h, _p(out))
        return out

    def filter(self, kind, d0, order, high_pass):
        img = np.empty(self.shape, np.uint8)
        spec = np.empty(self.shape, np.uint8)
        cpp.fd_filter(self.h, KIND_IDS[kind], float(d0), int(order),
                      int(high_pass), _p(img), _p(spec))
        return img, spec


def hybrid(lo, hi, kind, d_lo, d_hi, order):
    out = np.empty(lo.shape, np.uint8)
    ok = cpp.fd_hybrid(lo.h, hi.h, KIND_IDS[kind], float(d_lo), float(d_hi),
                       int(order), _p(out))
    return out if ok else None


@st.cache_resource(show_spinner=False, max_entries=8)
def get_spectrum(_img, key, shape):
    # _img is excluded from hashing; (key, shape) identify the cache entry.
    return Spectrum(_img, shape)


# =====================================================================
# PLOTTING (display only)
# =====================================================================
BAR_COLORS = {"Gray": "#8A94A6", "R": "#E5484D", "G": "#30A46C", "B": "#3E63DD"}
CDF_COLOR = "#F5A524"
XTICKS = [0, 32, 64, 96, 128, 160, 192, 224, 255]


def _theme():
    """(text color, grid color) matching the active Streamlit theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return ("#E6E8EB", "#4A505C") if dark else ("#1F2328", "#C9CED6")


def _style(ax, fg, grid):
    ax.patch.set_alpha(0)
    ax.spines["top"].set_visible(False)
    for side in ("left", "bottom", "right"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=fg, labelsize=8, length=3)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)


def _pct(value, _pos=None):
    """0.001 -> '0.1%', 0.4 -> '40%' (works for linear and log axes)."""
    return f"{value * 100:g}%"


def _show_figure(fig):
    """Renders a matplotlib figure as a transparent PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.image(buf, width="stretch")


def plot_hist(dist, title=None, color=None, log=None):
    """Two aligned panels sharing the x axis: PDF (top) and CDF (bottom)."""
    if log is None:
        log = st.session_state.get("log_scale", False)
    fg, grid = _theme()
    color = color or BAR_COLORS["Gray"]
    x = np.arange(256)
    pdf, cdf = dist["pdf"], dist["cdf"]
    peak = float(pdf.max()) if pdf.max() > 0 else 1.0
    top_bin = int(pdf.argmax())

    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(6, 4.8), dpi=120,
        gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_alpha(0)
    _style(ax, fg, grid)
    _style(ax2, fg, grid)

    # ---- PDF (top) ----
    ax.bar(x, pdf, width=1.0, color=color, edgecolor="none", zorder=2)
    ax.set_ylabel("PDF")
    if log:
        nonzero = pdf[pdf > 0]
        ax.set_yscale("log")
        ax.set_ylim(float(nonzero.min()) / 2 if nonzero.size else 1e-6, peak * 2)
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, peak * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.grid(axis="y", color=grid, alpha=0.5, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    # label a dominant bin (e.g. a white background at 255) and link it to the CDF jump
    if peak >= 0.10:
        ax.annotate(f"{top_bin}: {peak * 100:.0f}%", xy=(top_bin, peak),
                    xytext=(-6, -2), textcoords="offset points",
                    ha="right", va="top", color=fg, fontsize=8)
        for a in (ax, ax2):
            a.axvline(top_bin, color=fg, alpha=0.3, linewidth=0.8,
                      linestyle=":", zorder=1)

    # ---- CDF (bottom) ----
    ax2.plot(x, cdf, color=CDF_COLOR, linewidth=2, zorder=3)
    ax2.fill_between(x, cdf, color=CDF_COLOR, alpha=0.15, zorder=2)
    ax2.set_ylim(0, 1.05)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax2.set_ylabel("CDF", color=CDF_COLOR)
    ax2.tick_params(axis="y", colors=CDF_COLOR)
    ax2.grid(axis="y", color=grid, alpha=0.5, linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xlim(-0.5, 255.5)
    ax2.set_xticks(XTICKS)
    ax2.set_xlabel("Intensity  (0 = black, 255 = white)")

    if title:
        ax.set_title(title, color=fg, fontsize=10, loc="left")
    fig.subplots_adjust(hspace=0.12)
    _show_figure(fig)


def plot_cdf_compare(curves):
    """curves: {name: (distribution, color)} drawn on the same axes."""
    fg, grid = _theme()
    x = np.arange(256)

    fig, ax = plt.subplots(figsize=(6, 3.3), dpi=120)
    fig.patch.set_alpha(0)
    _style(ax, fg, grid)

    ax.plot([0, 255], [0, 1], linestyle="--", color=fg, alpha=0.5, linewidth=1.2,
            label="Ideal (uniform)")
    for name, (dist, color) in curves.items():
        ax.plot(x, dist["cdf"], color=color, linewidth=2, label=name)

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(XTICKS)
    ax.set_xlabel("Intensity  (0 = black, 255 = white)")
    ax.set_ylabel("CDF")
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.grid(color=grid, alpha=0.5, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=fg, loc="lower right")
    fig.tight_layout()
    _show_figure(fig)


def show(col, title, img):
    with col:
        st.subheader(title)
        st.image(img, width="stretch")


def render_color_section(file_bytes):
    """Task 8: everything is computed in C++, Python only displays it."""
    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    if bgr is None:
        st.warning("Could not read the color image.")
        return

    gray, equalized, stats = color_analysis(bgr)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("Color Image")
        st.image(bgr, channels="BGR", width="stretch")
    show(g2, "Grayscale", gray)
    show(g3, "Equalized (gray CDF mapping)", equalized)

    st.markdown("##### R, G, B and Gray histograms with their CDF")
    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, name in zip(row1 + row2, ("R", "G", "B", "Gray")):
        with col:
            plot_hist(stats[name], title=f"{name} channel", color=BAR_COLORS[name])

    _, _, eq_dist = process(equalized, NO_NOISE, FILTER_IDS["None"])
    left, right = st.columns(2)
    with left:
        plot_hist(eq_dist, title="Gray after equalization", color="#30A46C")
    with right:
        st.markdown(
            "The orange **CDF** of the gray image is the mapping curve: "
            "`new value = round(255 x CDF[old value])`. After equalization the "
            "gray levels are spread over the full range and the CDF becomes "
            "close to a straight line, unless a single gray level holds a large "
            "share of the pixels (e.g. a pure white background), because pixels "
            "with the same value always get the same new value."
        )


# =====================================================================
# PAGE
# =====================================================================
st.set_page_config(layout="wide", page_title="CV Studio")
st.title("Computer Vision Processing (C++ Engine)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    st.caption("All processing controls are inside the tabs.")
    st.markdown("**Graph options**")
    st.checkbox(
        "Log scale for histograms", key="log_scale",
        help="Use it when a few bins (black background, salt & pepper noise...) "
             "hide the rest of the histogram."
    )

if uploaded_file is None:
    st.info("Upload an image from the sidebar to start.")
    st.stop()

file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
og_img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)
if og_img is None:
    st.error("Could not read the uploaded image.")
    st.stop()

og_img = np.ascontiguousarray(og_img)
shape = og_img.shape
image_key = hashlib.md5(file_bytes.tobytes()).hexdigest()
S1 = get_spectrum(og_img, image_key, shape)          # FFT of the main image (once)
max_cutoff = max(1, max(shape) // 2)

tab_noise, tab_edge, tab_hist, tab_freq, tab_hybrid = st.tabs([
    "Noise & Smoothing",
    "Edge Detection",
    "Histogram & Contrast",
    "Frequency Domain",
    "Hybrid Images",
])

# ---------------------------------------------------------------------
# TAB 1: Noise + low-pass filters
# ---------------------------------------------------------------------
with tab_noise:
    c_noise, c_filter = st.columns(2)

    with c_noise:
        st.markdown("#### Noise")
        noise_type = st.selectbox("Noise type", list(NOISE_IDS), key="noise_type")
        p1, p2 = 0.0, 0.0
        if noise_type == "Uniform":
            p1, p2 = st.slider("Range (lower, upper)", -100, 100, (-30, 30), key="n_uni")
        elif noise_type == "Gaussian":
            p1 = st.slider("Mean", -50.0, 50.0, 0.0, key="n_gmean")
            p2 = st.slider("Std Dev", 1.0, 100.0, 25.0, key="n_gstd")
        elif noise_type == "Salt & Pepper":
            p1 = st.slider("Ratio (Pepper/Salt)", 0.0, 9999.0, 1.0, 1.0, key="n_ratio")
            p2 = st.slider("Density", 0.0, 1.0, 0.05, 0.01, key="n_density")

    with c_filter:
        st.markdown("#### Low-pass filter")
        smooth = st.selectbox("Filter", ["None", "Average", "Gaussian", "Median"],
                              key="smooth_filter")
        k_size = 3
        g_mean, g_std = 0.0, 1.0
        if smooth != "None":
            k_size = st.select_slider("Kernel size", options=[3, 5, 7], value=3, key="smooth_k")
        if smooth == "Gaussian":
            g_mean = st.slider("Gaussian kernel mean", -3.0, 3.0, 0.0, 0.1, key="g_mean")
            g_std = st.slider("Gaussian kernel std dev", 0.1, 5.0, 1.0, 0.1, key="g_std")

    noise_args = (NOISE_IDS[noise_type], p1, p2)
    noisy_img, smooth_img, smooth_dist = process(
        og_img, noise_args, FILTER_IDS[smooth], k_size, 0, g_mean, g_std
    )
    _, _, noisy_dist = process(og_img, noise_args, FILTER_IDS["None"])

    r1, r2, r3 = st.columns(3)
    show(r1, "Original", og_img)
    show(r2, "Noisy", noisy_img)
    show(r3, "Filtered", smooth_img)

    st.subheader("Histogram & CDF")
    h1, h2 = st.columns(2)
    with h1:
        plot_hist(noisy_dist, title="Noisy image")
    with h2:
        plot_hist(smooth_dist, title="Filtered image", color=BAR_COLORS["B"])

# ---------------------------------------------------------------------
# TAB 2: Edge detection
# ---------------------------------------------------------------------
with tab_edge:
    d1, d2 = st.columns([3, 2])
    method = d1.radio("Detector", ["Sobel", "Prewitt", "Roberts", "Canny"],
                      horizontal=True, key="edge_method")
    use_noisy = d2.checkbox("Add the noise selected in the first tab", key="edge_noisy")
    edge_noise = noise_args if use_noisy else NO_NOISE

    if method == "Canny":
        src, edges, edge_dist = process(og_img, edge_noise, FILTER_IDS["Canny"])
        e1, e2 = st.columns(2)
        show(e1, "Input", src)
        show(e2, "Canny edges", edges)
    else:
        src, gx, edge_dist = process(og_img, edge_noise, FILTER_IDS[method], 3, 0)
        _, gy, _ = process(og_img, edge_noise, FILTER_IDS[method], 3, 1)
        e1, e2, e3 = st.columns(3)
        show(e1, "Input", src)
        show(e2, f"{method} - X direction", gx)
        show(e3, f"{method} - Y direction", gy)
        st.caption("X direction highlights vertical edges, Y direction highlights horizontal edges.")

    with st.expander("Histogram & CDF of the result"):
        left, _ = st.columns(2)
        with left:
            plot_hist(edge_dist, color=BAR_COLORS["B"])

# ---------------------------------------------------------------------
# TAB 3: Histogram, equalization, normalization, color -> gray
# ---------------------------------------------------------------------
with tab_hist:
    st.subheader("Equalization & Normalization")

    _, eq_img, eq_dist = process(og_img, NO_NOISE, FILTER_IDS["Equalize"])
    _, nm_img, nm_dist = process(og_img, NO_NOISE, FILTER_IDS["Normalize"])
    _, _, og_dist = process(og_img, NO_NOISE, FILTER_IDS["None"])

    items = [
        ("Original", og_img, og_dist, BAR_COLORS["Gray"]),
        ("Equalized", eq_img, eq_dist, BAR_COLORS["G"]),
        ("Normalized", nm_img, nm_dist, BAR_COLORS["B"]),
    ]
    for col, (title, img, dist, color) in zip(st.columns(3), items):
        show(col, title, img)
        with col:
            plot_hist(dist, color=color)

    st.markdown("##### CDF comparison")
    cmp_left, cmp_right = st.columns(2)
    with cmp_left:
        plot_cdf_compare({title: (dist, color) for title, _, dist, color in items})
    with cmp_right:
        st.markdown(
            "- **Equalization** spreads the gray levels, so its CDF gets close to "
            "the straight dashed line (a uniform histogram).\n"
            "- **Normalization** only stretches the range linearly to 0-255, so "
            "the shape of the CDF stays the same as the original."
        )

    st.divider()
    st.subheader("Color to Gray, R/G/B Histograms and CDF")
    render_color_section(file_bytes)

# ---------------------------------------------------------------------
# TAB 4: Frequency-domain filters
# ---------------------------------------------------------------------
with tab_freq:
    k1, k2, k3 = st.columns(3)
    fd_kind = k1.selectbox("Mask type", list(KIND_IDS), key="fd_kind")
    fd_d0 = k2.slider("Cutoff D0 (px)", 1, max_cutoff, min(30, max_cutoff), key="fd_d0")
    fd_order = 2
    if fd_kind == "Butterworth":
        fd_order = k3.slider("Butterworth order", 1, 10, 2, key="fd_order")

    lp_img, lp_spec = S1.filter(fd_kind, fd_d0, fd_order, False)
    hp_img, hp_spec = S1.filter(fd_kind, fd_d0, fd_order, True)

    s0, s1, s2 = st.columns(3)
    with s0:
        st.subheader("Original + Spectrum")
        st.image(og_img, width="stretch")
        st.image(S1.view(), width="stretch")
    with s1:
        st.subheader("Low-pass")
        st.image(lp_img, width="stretch")
        st.image(lp_spec, width="stretch")
    with s2:
        st.subheader("High-pass")
        st.image(hp_img, width="stretch")
        st.image(hp_spec, width="stretch")

# ---------------------------------------------------------------------
# TAB 5: Hybrid images
# ---------------------------------------------------------------------
with tab_hybrid:
    other = st.file_uploader("Second image", type=["png", "jpg", "jpeg"], key="hybrid_up")

    if other is None:
        st.info("Upload a second image. By default the main image gives the low "
                "frequencies and the second image gives the high frequencies.")
    else:
        other_bytes = other.getvalue()
        other_img = cv.imdecode(np.frombuffer(other_bytes, np.uint8), cv.IMREAD_GRAYSCALE)

        if other_img is None:
            st.error("Could not read the second image.")
        else:
            # resized to the main image's size inside C++
            S2 = get_spectrum(other_img, hashlib.md5(other_bytes).hexdigest(), shape)

            m1, m2, m3 = st.columns(3)
            hy_kind = m1.selectbox("Mask type", list(KIND_IDS), key="hy_kind")
            hy_order = 2
            if hy_kind == "Butterworth":
                hy_order = m2.slider("Butterworth order", 1, 10, 2, key="hy_order")
            swap = m3.checkbox("Swap roles (main image = high freq)", key="hy_swap")

            m4, m5 = st.columns(2)
            d_lo = m4.slider("Low-pass cutoff", 1, max_cutoff, min(15, max_cutoff), key="hy_lo")
            d_hi = m5.slider("High-pass cutoff", 1, max_cutoff, min(40, max_cutoff), key="hy_hi")

            lo, hi = (S2, S1) if swap else (S1, S2)
            hy = hybrid(lo, hi, hy_kind, d_lo, d_hi, hy_order)

            if hy is None:
                st.error("Images could not be combined (size mismatch).")
            else:
                r1, r2, r3 = st.columns(3)
                show(r1, "Low-freq source", lo.img)
                show(r2, "High-freq source", hi.img)
                show(r3, "Hybrid", hy)

                st.caption("Hybrid downscaled 4x (the low-frequency image should dominate):")
                st.image(hy, width=max(shape[1] // 4, 64))