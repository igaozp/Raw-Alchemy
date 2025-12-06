# -*- mode: python ; coding: utf-8 -*-
import sys

# --- Platform-specific settings ---
# Enable strip on Linux and macOS for a smaller executable.
# On Windows, stripping can sometimes cause issues with antivirus software
# or runtime behavior, so it's safer to leave it disabled.
# On macOS, stripping the Python shared library can lead to runtime errors,
# such as "Failed to load Python shared library". Disabling strip for macOS
# is a safer approach to ensure all necessary symbols are preserved.
strip_executable = True if sys.platform.startswith('linux') else False


# --- Platform-specific binaries ---
binaries_list = []
if sys.platform == 'darwin':
    import os
    import rawpy

    # Find the path to libraw_r.dylib within the rawpy package
    rawpy_path = os.path.dirname(rawpy.__file__)
    libraw_dylib_path = os.path.join(rawpy_path, 'libraw_r.dylib')
    binaries_list.append((libraw_dylib_path, '.'))


a = Analysis(
    ['src/raw_alchemy/gui.py'],
    pathex=[],
    binaries=binaries_list,
    datas=[('src/raw_alchemy/vendor', 'vendor'), ('icon.ico', '.')],
    hiddenimports=['tkinter'],
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

# On macOS, BUNDLE is used, which has its own icon parameter.
# The .ico format is for Windows, so we remove it from datas on macOS.
if sys.platform == 'darwin':
    a.datas = [item for item in a.datas if item[0] != 'icon.ico']

pyz = PYZ(a.pure)

# Platform-specific EXE and BUNDLE for macOS .app creation
if sys.platform == 'darwin':
    # For macOS, create a one-folder bundle (.app)
    exe = EXE(
        pyz,
        a.scripts,
        [],
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
        icon='icon.icns',
    )
    # The COLLECT name must be different from the EXE name to avoid a file/directory name collision in the dist/ folder.
    # BUNDLE will use this folder to create the final .app with the correct name.
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=strip_executable,
        upx=False,
        name='RawAlchemy_COLLECT',
    )
    app = BUNDLE(
        coll,
        name='RawAlchemy.app',
        icon='icon.icns',
        bundle_identifier=None,
    )
else:
    # For Windows and Linux, create a one-file executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='RawAlchemy',
        debug=False,
        bootloader_ignore_signals=False,
        strip=strip_executable,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.ico',
    )
