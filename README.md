# C++ Computer Vision Engine

This project provides a Streamlit interface backed by a C++ image-processing engine. Python handles the interface and image upload; `engine.cpp` is compiled into `engine.dll` and called through `ctypes`.

## Requirements

- Windows 10 or later
- Python 3.10 or later
- MSYS2 with the MinGW-w64 toolchain

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
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit numpy opencv-python
```

### 3. Build the C++ engine

In the same PowerShell terminal:

```powershell
$env:Path = "C:\msys64\mingw64\bin;$env:Path"
g++ -O3 -shared -o engine.dll engine.cpp -IC:\msys64\mingw64\include\opencv5 -LC:\msys64\mingw64\lib -lopencv_core -lopencv_imgproc
```

If MSYS2 uses a different location, replace the `C:\msys64` paths with the matching paths.

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the URL printed by Streamlit, usually `http://localhost:8501`.

## Project files

- `app.py`: Streamlit interface and Python-to-C++ bridge
- `engine.cpp`: native image-processing implementation
- `engine.dll`: generated build output, not committed to Git

Rebuild `engine.dll` after changing `engine.cpp`. Python-only interface changes do not require rebuilding it.

## Troubleshooting

### `Could not find module engine.dll`

Make sure `engine.dll` exists in the same folder as `app.py`, and that the MSYS2 MinGW directory is available through `MINGW64_BIN` or the default `C:\msys64\mingw64\bin` path.

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
