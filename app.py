import streamlit as st
import cv2 as cv
import numpy as np
import ctypes
import os
import matplotlib.pyplot as plt

# --- Link C++ Engine ---
base_dir = os.path.dirname(os.path.abspath(__file__))
mingw_dll_directory = None
if os.name == "nt":
    mingw_bin = os.environ.get("MINGW64_BIN", r"C:\msys64\mingw64\bin")
    if os.path.isdir(mingw_bin):
        mingw_dll_directory = os.add_dll_directory(mingw_bin)

cpp = ctypes.CDLL(os.path.join(base_dir, "engine.dll"))
cpp.run_pipeline.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, # Input
    ctypes.c_int, ctypes.c_float, ctypes.c_float,               # Noise
    ctypes.c_int, ctypes.c_int, ctypes.c_int,                   # Filter
    ctypes.c_float, ctypes.c_float,                              # Gaussian kernel mean/stddev
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
    ctypes.POINTER(ctypes.c_int32)                               # Histogram bins
]

st.set_page_config(layout="wide", page_title="CV Studio")
st.title("Computer Vision Processing (C++ Engine)")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg"])
    
    noise_type = st.selectbox("Noise", ["None", "Uniform", "Gaussian", "Salt & Pepper"])
    np1, np2 = 0.0, 0.0
    if noise_type == "Uniform":
        np1, np2 = st.slider("Range (Lower, Upper)", -100, 100, (-30, 30))
    elif noise_type == "Gaussian":
        np1 = st.slider("Mean", -50.0, 50.0, 0.0)
        np2 = st.slider("Std Dev", 1.0, 100.0, 25.0)
    elif noise_type == "Salt & Pepper":
        np1 = st.slider("Ratio (Pepper/Salt)", 0.0, 9999.0, 1.0, 1.0)
        np2 = st.slider("Density", 0.0, 1.0, 0.05, 0.01)

    filter_dict = {"None":0, "Average":1, "Gaussian":2, "Median":3, "Sobel":4, "Prewitt":5, "Roberts":6, "Canny (OpenCV)":7, "Equalize":8, "Normalize":9}
    filter_choice = st.selectbox("Filter/Edges", list(filter_dict.keys()))

    filter_mean, filter_stddev = 0.0, 1.0
    if filter_choice == "Gaussian":
        filter_mean = st.slider("Gaussian kernel mean", -3.0, 3.0, 0.0, 0.1)
        filter_stddev = st.slider("Gaussian kernel std dev", 0.1, 5.0, 1.0, 0.1)
    
    direction = 0 if st.radio("Direction", ["X", "Y"], horizontal=True) == "X" else 1
    k_size = st.select_slider("Kernel Size", options=[3, 5, 7], value=3)

if uploaded_file is not None:
    # 1. Read Grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    og_img = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    # 2. Allocate memory for C++ to write into
    noisy_img = np.zeros_like(og_img)
    out_img = np.zeros_like(og_img)
    out_hist = np.zeros(256, dtype=np.int32)

    # 3. Execute Native C++ OOP Pipeline
    n_idx = ["None", "Uniform", "Gaussian", "Salt & Pepper"].index(noise_type)
    
    cpp.run_pipeline(
        og_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), og_img.shape[0], og_img.shape[1],
        n_idx, float(np1), float(np2),
        filter_dict[filter_choice], k_size, direction,
        float(filter_mean), float(filter_stddev),
        noisy_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out_hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    )

    # 4. Render
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original Image")
        st.image(og_img, width="stretch")
    with c2:
        st.subheader("Noisy Image")
        st.image(noisy_img, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Filtered Result")
        st.image(out_img, width="stretch")
    with c4:
        st.subheader("Histogram & PDF")
        intensity = np.arange(256)
        pdf = out_hist / out_hist.sum() if out_hist.sum() else out_hist.astype(float)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(intensity, out_hist, width=1.0, color="black", alpha=0.75)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(0, 255)
        ax2 = ax.twinx()
        ax2.plot(intensity, pdf, color="tab:blue")
        ax2.set_ylabel("Probability")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- Task 8: Color -> Gray, R/G/B histograms + CDF, equalization via gray CDF ---
    st.divider()
    st.header("Color to Gray, RGB Histograms and CDF")

    bgr = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = np.round(0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    channels = [("R", R, "red"), ("G", G, "green"), ("B", B, "blue"), ("Gray", gray, "gray")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    cdfs = {}
    for ax, (name, ch, col) in zip(axes, channels):
        h = np.bincount(ch.ravel(), minlength=256)
        pdf = h / h.sum()
        cdf = np.cumsum(pdf)
        cdfs[name] = cdf
        ax.bar(np.arange(256), pdf, color=col, width=1.0)
        ax.set_title(f"{name} histogram (PDF) + CDF")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("PDF")
        ax2 = ax.twinx()
        ax2.plot(cdf, color="black")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("CDF")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    eq_gray = np.round(255 * cdfs["Gray"]).astype(np.uint8)[gray]
    g1, g2, g3 = st.columns(3)
    with g1:
        st.subheader("Color Image")
        st.image(rgb, width="stretch")
    with g2:
        st.subheader("Grayscale")
        st.image(gray, width="stretch")
    with g3:
        st.subheader("Equalized (gray CDF mapping)")
        st.image(eq_gray, width="stretch")