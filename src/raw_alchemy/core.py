import gc
import rawpy
import numpy as np
import colour
import tifffile
from PIL import Image
import pillow_heif
import os
from typing import Optional

# 尝试导入同级目录下的 utils，如果失败则尝试绝对导入 (方便不同运行环境调试)
try:
    from . import utils
except ImportError:
    import utils

# ==========================================
#              1. 常量定义 & 映射表
# ==========================================

# 映射：Log 空间名称 -> 对应的线性色域 (Linear Gamut)
LOG_TO_WORKING_SPACE = {
    'F-Log': 'F-Gamut',
    'F-Log2': 'F-Gamut',
    'F-Log2C': 'F-Gamut C',
    'V-Log': 'V-Gamut',
    'N-Log': 'N-Gamut',
    'L-Log': 'ITU-R BT.2020',
    'Canon Log 2': 'Cinema Gamut',
    'Canon Log 3': 'Cinema Gamut',
    'S-Log3': 'S-Gamut3',
    'S-Log3.Cine': 'S-Gamut3.Cine',
    'Arri LogC3': 'ARRI Wide Gamut 3',
    'Arri LogC4': 'ARRI Wide Gamut 4',
    'Log3G10': 'REDWideGamutRGB',
}

# 映射：复合名称 -> colour 库识别的 Log 编码函数名称
LOG_ENCODING_MAP = {
    'S-Log3.Cine': 'S-Log3',
    'F-Log2C': 'F-Log2',
}

# 测光模式选项
METERING_MODES = [
    'average',        # 几何平均 (默认)
    'center-weighted',# 中央重点
    'highlight-safe', # 高光保护 (ETTR)
    'hybrid',         # 混合 (平均 + 高光限制)
    'matrix',         # 矩阵/评价测光
]

# ==========================================
#              2. 核心处理函数
# ==========================================

def process_image(
    raw_path: str,
    output_path: str,
    log_space: str,
    lut_path: Optional[str],
    exposure: Optional[float] = None, # None=自动, Float=手动EV
    lens_correct: bool = True,
    metering_mode: str = 'hybrid',
    custom_db_path: Optional[str] = None,
    log_queue: Optional[object] = None, # 多进程通信队列
):
    filename = os.path.basename(raw_path)

    # 内部日志辅助函数
    def _log(message):
        if log_queue:
            # 发送结构化日志：{'id':文件名, 'msg':消息}
            # 注意：如果是 Queue 对象，使用 .put()
            if hasattr(log_queue, 'put'):
                log_queue.put({'id': filename, 'msg': message})
            else:
                # 兼容 CLI 模式传入 print 函数的情况
                print(f"[{filename}] {message}")
        else:
            print(f"[{filename}] {message}")

    _log(f"🧪 [Raw Alchemy] Processing: {raw_path}")

    # --- Step 1: 解码 RAW (统一至 ProPhoto RGB / 16-bit Linear) ---
    _log(f"  🔹 [Step 1] Decoding RAW...")
    with rawpy.imread(raw_path) as raw:
        # 提取 EXIF (用于镜头校正)
        exif_data = utils.extract_lens_exif(raw, logger=_log)

        # 解码: 必须使用 16-bit 以保留 Log 转换所需的动态范围
        prophoto_linear = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.ProPhoto,
            bright=1.0,
            highlight_mode=2, # 2=Blend (防止高光死白)
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
        )
        # 转为 Float32 (0.0 - 1.0) 进行数学运算
        img = prophoto_linear.astype(np.float32) / 65535.0
        
        # 立即释放内存
        del prophoto_linear 
        gc.collect()

    source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']

    # --- Step 2: 曝光控制 ---
    gain = 1.0
    if exposure is not None:
        # 路径 A: 手动曝光
        _log(f"  🔹 [Step 2] Manual Exposure Override ({exposure:+.2f} stops)")
        gain = 2.0 ** exposure
        utils.apply_gain_inplace(img, gain)
    else:
        # 路径 B: 自动测光
        _log(f"  🔹 [Step 2] Auto Exposure ({metering_mode})")
        if metering_mode == 'center-weighted':
            img = utils.auto_expose_center_weighted(img, source_cs, target_gray=0.18, logger=_log)
        elif metering_mode == 'highlight-safe':
            img = utils.auto_expose_highlight_safe(img, clip_threshold=1.0, logger=_log)
        elif metering_mode == 'average':
            img = utils.auto_expose_linear(img, source_cs, target_gray=0.18, logger=_log)
        elif metering_mode == 'matrix':
            img = utils.auto_expose_matrix(img, source_cs, target_gray=0.18, logger=_log)
        else: # hybrid as default
            img = utils.auto_expose_hybrid(img, source_cs, target_gray=0.18, logger=_log)

    # --- Step 3: 镜头校正 & 风格化 ---
    if lens_correct:
        _log("  🔹 [Step 3] Applying Lens Correction...")
        img = utils.apply_lens_correction(
            img,
            exif_data=exif_data,
            custom_db_path=custom_db_path,
            logger=_log
        )

    # 稍微增加饱和度和对比度，为 LUT 转换打底
    _log("  🔹 [Step 3.5] Applying Camera-Match Boost...")
    img = utils.apply_saturation_and_contrast(img, saturation=1.25, contrast=1.1)

    # --- Step 4: 色彩空间转换 (ProPhoto Linear -> Log) ---
    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)
    
    if not log_color_space_name:
         raise ValueError(f"Unknown Log Space: {log_space}")

    _log(f"  🔹 [Step 4] Color Transform (ProPhoto -> {log_color_space_name} -> {log_curve_name})")

    # 4.1 Gamut 变换 (矩阵运算)
    M = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name],
    )
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    utils.apply_matrix_inplace(img, M)
    
    # 4.2 Log 编码
    # Log 函数无法处理负值，需裁剪微小底噪
    np.maximum(img, 1e-6, out=img) 
    img = colour.cctf_encoding(img, function=log_curve_name)

    # --- Step 5: 应用 LUT ---
    if lut_path:
        _log(f"  🔹 [Step 5] Applying LUT {os.path.basename(lut_path)}...")
        try:
            lut = colour.read_LUT(lut_path)
            
            # 3D LUT 使用 Numba 加速
            if isinstance(lut, colour.LUT3D):
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                
                utils.apply_lut_inplace(img, lut.table, lut.domain[0], lut.domain[1])
            else:
                # 1D LUT 使用 colour 库默认方法
                img = lut.apply(img)

            # LUT 可能导致数值溢出，需裁剪到 [0.0, 1.0]
            np.clip(img, 0.0, 1.0, out=img)
            
        except Exception as e:
            _log(f"  ❌ [Error] applying LUT: {e}")

    # --- Step 6: 保存 (关键优化部分) ---
    _log(f"  💾 Saving to {os.path.basename(output_path)}...")
    
    file_ext = os.path.splitext(output_path)[1].lower()
    output_image_uint16 = None

    try:
        # === A. 16-bit TIFF (无损母版) ===
        if file_ext in ['.tif', '.tiff']:
            _log("    Format: TIFF (16-bit, ZLIB Optimized)")
            output_image_uint16 = (img * 65535).astype(np.uint16)
            
            tifffile.imwrite(
                output_path,
                output_image_uint16,
                photometric='rgb',
                compression='zlib',
                # 【优化】predictor=2 (水平差分) 大幅提升照片压缩率
                predictor=2,       
                # 【优化】level=8 平衡速度和体积
                compressionargs={'level': 8} 
            )

        # === B. 10-bit HEIF (高质量分享) ===
        elif file_ext in ['.heic', '.heif']:
            _log("    Format: HEIF (10-bit, High Quality)")
            output_image_uint16 = (img * 65535).astype(np.uint16)
            
            # 使用 pillow_heif 直接写入
            heif_file = pillow_heif.from_bytes(
                mode='RGB;16',
                size=(output_image_uint16.shape[1], output_image_uint16.shape[0]),
                data=output_image_uint16.tobytes()
            )
            # 【优化】quality=-1 (无损/最高画质), bit_depth=10, 保持 4:4:4
            heif_file.save(output_path, quality=-1, bit_depth=10)

        # === C. 8-bit JPEG (通用预览) ===
        else:
            _log(f"    Format: {file_ext.upper()} (8-bit High Quality)")
            # 转换为 8-bit
            output_image_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            
            # 针对 JPG 的特殊优化参数
            save_params = {}
            if file_ext in ['.jpg', '.jpeg']:
                save_params = {
                    'quality': 95,
                    'subsampling': 0,
                    'optimize': True
                }
            
            Image.fromarray(output_image_uint8).save(output_path, **save_params)

        _log(f"  ✅ Saved: {output_path}")

    except Exception as e:
        _log(f"  ❌ [Error] Failed to save file: {e}")
        import traceback
        traceback.print_exc()
    
    # --- 最终清理 ---
    del img
    if output_image_uint16 is not None:
        del output_image_uint16
    gc.collect()