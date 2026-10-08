# C++ Computer Vision Engine

This project provides a Streamlit interface backed by a C++ image-processing engine. Python handles the interface and image upload; `engine.cpp` is compiled into `engine.dll` and called through `ctypes`.

## Features

- Noise: Uniform, Gaussian, Salt & Pepper
- Filters: Average, Gaussian, Median
- Edge detection: Sobel, Prewitt, Roberts, Canny (OpenCV)
- Histogram equalization (CDF mapping): `new = round(255 * CDF[old])`
- Normalization (linear stretch to 0-255): `new = (old - min) / (max - min) * 255`
- Color to grayscale (`0.299 R + 0.587 G + 0.114 B`), with R, G, B and gray histograms (PDF) and CDF curves, plus equalization using the gray CDF

All processing is done in the spatial domain (no Fourier or other transforms).

## Requirements

- Windows 10 or later
- Python 3.10 or later
- MSYS2 with the MinGW-w64 toolchain
- Python packages: `streamlit`, `numpy`, `opencv-python`, `matplotlib`

## Setup

### 1. Install MSYS2

Install MSYS2 from [msys2.org](https://www.msys2.org/), using the default location if possible: `C:\msys64`.

Open PowerShell and install the compiler, OpenCV, and Qt6 runtime:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -Syu --noconfirm"
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-gcc mingw-w64-x86_64-opencv mingw-w64-x86_64-qt6-base"
```

If MSYS2 asks to close and reopen its terminal during the first update, do so, then run the second command.

If MSYS2 is installed somewhere else, set its MinGW directory before building and running:

```powershell
$env:MINGW64_BIN = "D:\path\to\msys64\mingw64\bin"
```

### 2. Create a Python environment

From the project folder:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit numpy opencv-python matplotlib
```

### 3. Build the C++ engine

Keep `app.py` and `engine.cpp` in the same folder. In PowerShell (repeat the `$env:Path` line in every new terminal):

```powershell
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
g++ -O3 -shared -o engine.dll engine.cpp -IC:\msys64\mingw64\include\opencv5 -LC:\msys64\mingw64\lib -lopencv_core -lopencv_imgproc
```

If the compiler cannot find the OpenCV headers, check whether they are in `include\opencv4` instead of `include\opencv5` and adjust the `-I` path.

If MSYS2 uses a different location, replace the `C:\msys64` paths with the matching paths.

### 4. Run the application

Use the virtual environment's Python directly. After the build step, `python` may resolve to MSYS2's Python instead of the venv:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## Project files

- `app.py`: Streamlit interface, Python-to-C++ bridge, and the color/RGB histogram and CDF section
- `engine.cpp`: native image-processing implementation (noise, filters, edges, equalize, normalize, histogram drawing)
- `engine.dll`: generated build output, not committed to Git

Rebuild `engine.dll` after changing `engine.cpp`. Python-only interface changes do not require rebuilding it.

## Troubleshooting

### `Could not find module engine.dll`

Make sure `engine.dll` exists in the same folder as `app.py`, and that the MSYS2 MinGW directory is available through `MINGW64_BIN` or the default `C:\msys64\mingw64\bin` path.

### `g++ is not recognized`

The MinGW path is not set in this terminal. Run:

```powershell
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
```

### `No module named streamlit` (or `matplotlib`)

`python` is resolving to MSYS2's Python, not the venv. Use the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit numpy opencv-python matplotlib
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Missing Qt6 or OpenCV DLLs

Install the runtime packages again:

```powershell
& "C:\msys64\usr\bin\bash.exe" -lc "pacman -S --needed --noconfirm mingw-w64-x86_64-opencv mingw-w64-x86_64-qt6-base"
```

### PowerShell blocks environment activation

Run this once in the current PowerShell session, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```