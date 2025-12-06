# -*- mode: python ; coding: utf-8 -*-
import json
import os
import platform
import shutil
import sys
import urllib.request
from pathlib import Path

# ============================================================================
# Lensfun Setup Hook
# ============================================================================
def get_download_url(asset_name):
    """Gets the download URL for a given asset from the latest GitHub release."""
    api_url = "https://api.github.com/repos/shenmintao/lensfun/releases/latest"
    try:
        with urllib.request.urlopen(api_url) as response:
            data = json.loads(response.read().decode())
            for asset in data["assets"]:
                if asset["name"] == asset_name:
                    return asset["browser_download_url"]
    except Exception as e:
        print(f"Error fetching release info from GitHub: {e}", file=sys.stderr)
        return None
    return None

def download_and_extract_lensfun():
    """Downloads and extracts the appropriate Lensfun library if not present."""
    vendor_dir = Path("src/raw_alchemy/vendor/lensfun")
    # If the directory already exists and is not empty, skip the download.
    if vendor_dir.exists() and any(vendor_dir.iterdir()):
        print("Lensfun directory already exists and is not empty. Skipping download.")
        return

    vendor_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    if system == "windows":
        asset_name = "lensfun-windows.zip"
    elif system == "linux":
        asset_name = "lensfun-linux.tar.gz"
    elif system == "darwin":
        asset_name = "lensfun-macos.tar.gz"
    else:
        print(f"Unsupported system: {system}", file=sys.stderr)
        sys.exit(1)

    download_url = get_download_url(asset_name)
    if not download_url:
        print(f"Could not find download URL for {asset_name}", file=sys.stderr)
        sys.exit(1)

    archive_path = Path(asset_name)
    try:
        print(f"Downloading Lensfun for {system} from {download_url}...")
        with urllib.request.urlopen(download_url) as response, open(
            archive_path, "wb"
        ) as out_file:
            shutil.copyfileobj(response, out_file)

        print("Extracting Lensfun...")
        shutil.unpack_archive(archive_path, vendor_dir)
        os.remove(archive_path)
        print("Lensfun setup complete.")

    except Exception as e:
        print(f"Error during Lensfun setup: {e}", file=sys.stderr)
        sys.exit(1)

# --- Run the setup hook before Analysis ---
download_and_extract_lensfun()


# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
strip_executable = True if sys.platform.startswith('linux') or sys.platform == 'darwin' else False

# --- Platform-specific binaries ---
binaries_list = []
if sys.platform == 'darwin' or sys.platform.startswith('linux'):
    import os
    import rawpy

    # Find the path to libraw_r library within the rawpy package
    rawpy_path = os.path.dirname(rawpy.__file__)
    lib_file = None
    for f in os.listdir(rawpy_path):
        if f.startswith('libraw_r'):
            lib_file = os.path.join(rawpy_path, f)
            break
    if lib_file:
        binaries_list.append((lib_file, '.'))


a = Analysis(
    ['src/raw_alchemy/gui.py'],
    pathex=[],
    binaries=[],
    datas=[('src/raw_alchemy/vendor', 'vendor'), ('icon.ico', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'pandas',
        'IPython',
        'PyQt5',
        'PySide2',
        'qtpy',
        'test',
        'doctest',
        'distutils',
        'setuptools',
        'wheel',
        'pkg_resources',
        'Cython',
        'PyInstaller',
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RawAlchemy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=strip_executable,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=strip_executable,
    upx=False,
    upx_exclude=[],
    name='RawAlchemy',
)