"""
Lensfun库的Python包装器
用于镜头畸变、色差和暗角校正
"""

import ctypes
import numpy as np
from typing import Optional, Tuple
import platform
import os
import sys

def _get_base_path():
    """
    Gets the base path for data files.
    Handles running as a script and as a frozen PyInstaller executable.
    """
    if getattr(sys, 'frozen', False):
        # For frozen apps (both one-file and one-dir), the base path is the
        # directory containing the executable.
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal script.
        return os.path.dirname(os.path.abspath(__file__))

# 根据平台加载正确的库
def _load_lensfun_library():
    """加载lensfun动态库"""
    system = platform.system()
    base_path = _get_base_path()
    lensfun_dir = os.path.join(base_path, "vendor", "lensfun")
    lib_dir = os.path.join(lensfun_dir, "lib")
    bin_dir = os.path.join(lensfun_dir, "bin")

    lib_path = None
    if system == "Windows":
        lib_path = os.path.join(lib_dir, "lensfun.dll")
        # Add bin directory to DLL search path for dependencies
        if os.path.isdir(bin_dir) and hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(bin_dir)
    elif system == "Darwin":
        lib_path = os.path.join(lib_dir, "liblensfun.dylib")
    else:  # Linux and other Unix-like
        lib_path = os.path.join(lib_dir, "liblensfun.so")

    try:
        if lib_path and os.path.exists(lib_path):
            return ctypes.CDLL(lib_path)
        else:
            # Fallback to system paths if not found in vendor
            if system == "Windows":
                return ctypes.CDLL("lensfun.dll")
            elif system == "Darwin":
                return ctypes.CDLL("liblensfun.dylib")
            else:
                return ctypes.CDLL("liblensfun.so")
    except OSError:
        return None


# 加载库
_lensfun = _load_lensfun_library()
if _lensfun is None:
    print("  ⚠️ [Lensfun] Library not loaded. Lens correction disabled.")


# ============================================================================
# Lensfun 常量定义
# ============================================================================

# 像素格式
LF_PF_U8 = 0
LF_PF_U16 = 1
LF_PF_U32 = 2
LF_PF_F32 = 3
LF_PF_F64 = 4

# 校正标志
LF_MODIFY_TCA = 0x00000001          # 横向色差
LF_MODIFY_VIGNETTING = 0x00000002   # 暗角
LF_MODIFY_DISTORTION = 0x00000008   # 畸变
LF_MODIFY_GEOMETRY = 0x00000010     # 几何投影
LF_MODIFY_SCALE = 0x00000020        # 缩放
LF_MODIFY_ALL = ~0

# 镜头类型
LF_UNKNOWN = 0
LF_RECTILINEAR = 1
LF_FISHEYE = 2
LF_PANORAMIC = 3
LF_EQUIRECTANGULAR = 4
LF_FISHEYE_ORTHOGRAPHIC = 5
LF_FISHEYE_STEREOGRAPHIC = 6
LF_FISHEYE_EQUISOLID = 7
LF_FISHEYE_THOBY = 8

# 颜色组件角色
LF_CR_END = 0
LF_CR_NEXT = 1
LF_CR_UNKNOWN = 2
LF_CR_INTENSITY = 3
LF_CR_RED = 4
LF_CR_GREEN = 5
LF_CR_BLUE = 6

# 颜色组件宏
def LF_CR_3(a, b, c):
    """定义3个组件的像素格式 (RGB)"""
    return a | (b << 4) | (c << 8)

LF_CR_RGB = LF_CR_3(LF_CR_RED, LF_CR_GREEN, LF_CR_BLUE)


# ============================================================================
# C结构体定义
# ============================================================================

class lfDatabase(ctypes.Structure):
    """数据库对象 (不透明)"""
    pass

class lfCamera(ctypes.Structure):
    """相机对象 (不透明)"""
    pass

class lfLens(ctypes.Structure):
    """镜头对象 (不透明)"""
    pass

class lfModifier(ctypes.Structure):
    """校正修改器对象 (不透明)"""
    pass


# ============================================================================
# 函数签名定义
# ============================================================================

if _lensfun:
    # 数据库函数
    _lensfun.lf_db_create.restype = ctypes.POINTER(lfDatabase)
    _lensfun.lf_db_create.argtypes = []
    
    _lensfun.lf_db_destroy.restype = None
    _lensfun.lf_db_destroy.argtypes = [ctypes.POINTER(lfDatabase)]
    
    _lensfun.lf_db_load.restype = ctypes.c_int
    _lensfun.lf_db_load.argtypes = [ctypes.POINTER(lfDatabase)]
    
    _lensfun.lf_db_load_path.restype = ctypes.c_int
    _lensfun.lf_db_load_path.argtypes = [ctypes.POINTER(lfDatabase), ctypes.c_char_p]

    _lensfun.lf_db_load_str.restype = ctypes.c_int
    _lensfun.lf_db_load_str.argtypes = [ctypes.POINTER(lfDatabase), ctypes.c_char_p, ctypes.c_size_t]
    
    _lensfun.lf_db_find_cameras_ext.restype = ctypes.POINTER(ctypes.POINTER(lfCamera))
    _lensfun.lf_db_find_cameras_ext.argtypes = [
        ctypes.POINTER(lfDatabase),
        ctypes.c_char_p,  # maker
        ctypes.c_char_p,  # model
        ctypes.c_int      # sflags
    ]
    
    _lensfun.lf_db_find_lenses.restype = ctypes.POINTER(ctypes.POINTER(lfLens))
    _lensfun.lf_db_find_lenses.argtypes = [
        ctypes.POINTER(lfDatabase),
        ctypes.POINTER(lfCamera),
        ctypes.c_char_p,  # maker
        ctypes.c_char_p,  # model
        ctypes.c_int      # sflags
    ]
    
    # 修改器函数
    _lensfun.lf_modifier_create.restype = ctypes.POINTER(lfModifier)
    _lensfun.lf_modifier_create.argtypes = [
        ctypes.POINTER(lfLens),
        ctypes.c_float,   # focal
        ctypes.c_float,   # crop
        ctypes.c_int,     # width
        ctypes.c_int,     # height
        ctypes.c_int,     # pixel_format
        ctypes.c_int      # reverse
    ]
    
    _lensfun.lf_modifier_destroy.restype = None
    _lensfun.lf_modifier_destroy.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_distortion_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_distortion_correction.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_tca_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_tca_correction.argtypes = [ctypes.POINTER(lfModifier)]
    
    _lensfun.lf_modifier_enable_vignetting_correction.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_vignetting_correction.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float,  # aperture
        ctypes.c_float   # distance
    ]
    
    _lensfun.lf_modifier_enable_projection_transform.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_projection_transform.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_int  # target_projection
    ]
    
    _lensfun.lf_modifier_enable_scaling.restype = ctypes.c_int
    _lensfun.lf_modifier_enable_scaling.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float  # scale
    ]
    
    _lensfun.lf_modifier_apply_subpixel_geometry_distortion.restype = ctypes.c_int
    _lensfun.lf_modifier_apply_subpixel_geometry_distortion.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_float,                    # xu
        ctypes.c_float,                    # yu
        ctypes.c_int,                      # width
        ctypes.c_int,                      # height
        ctypes.POINTER(ctypes.c_float)     # res
    ]
    
    _lensfun.lf_modifier_apply_color_modification.restype = ctypes.c_int
    _lensfun.lf_modifier_apply_color_modification.argtypes = [
        ctypes.POINTER(lfModifier),
        ctypes.c_void_p,  # pixels
        ctypes.c_float,   # x
        ctypes.c_float,   # y
        ctypes.c_int,     # width
        ctypes.c_int,     # height
        ctypes.c_int,     # comp_role
        ctypes.c_int      # row_stride
    ]
    
    _lensfun.lf_free.restype = None
    _lensfun.lf_free.argtypes = [ctypes.c_void_p]

    _lensfun.lf_modifier_get_auto_scale.restype = ctypes.c_float
    _lensfun.lf_modifier_get_auto_scale.argtypes = [ctypes.POINTER(lfModifier)]


# ============================================================================
# Python包装类
# ============================================================================

class LensfunDatabase:
    """Lensfun数据库包装器"""
    
    def __init__(self, custom_db_path: Optional[str] = None, logger: callable = print):
        if not _lensfun:
            raise RuntimeError("Lensfun library not loaded")
        self.db = _lensfun.lf_db_create()
        if not self.db:
            raise RuntimeError("Could not create lensfun database")
        
        # 检查本地数据库路径
        base_path = _get_base_path()
        db_path = os.path.join(base_path, "vendor", "lensfun", "share", "lensfun", "version_2")
        
        result = -1
        if os.path.isdir(db_path):
            logger(f"  ✨ [Lensfun] Found local database, loading from: {db_path}")
            result = _lensfun.lf_db_load_path(self.db, db_path.encode('utf-8'))
        else:
            logger(f"  ℹ️ [Lensfun] Local database not found, loading from system default paths.")
            result = _lensfun.lf_db_load(self.db)

        # Check loading result
        if result != 0:
            error_msg = f"Failed to load lensfun database, error code: {result}"
            if result == 2:  # LF_IO_ERROR
                error_msg += "\n  💡 [Hint] Database file not found or could not be read."
                error_msg += f"\n     - Check if the path is correct: {db_path if os.path.isdir(db_path) else 'System paths'}"
                error_msg += "\n     - Ensure file permissions are correct."
            raise RuntimeError(error_msg)
        
        # 加载用户自定义数据库
        if custom_db_path and os.path.exists(custom_db_path):
            logger(f"  ✨ [Lensfun] Loading custom database from: {custom_db_path}")
            try:
                with open(custom_db_path, 'rb') as f:
                    xml_data = f.read()
                
                if xml_data:
                    # lf_db_load_str用于从字符串加载XML数据
                    result = _lensfun.lf_db_load_str(self.db, xml_data, len(xml_data))
                    if result != 0:
                        error_msg = f"Failed to load custom lensfun database from file: {custom_db_path}, error code: {result}"
                        if result == 1:  # LF_WRONG_FORMAT
                            error_msg += "\n  💡 [Hint] The XML data has the wrong format. Please check if the file is a valid Lensfun database file."
                        elif result == 2:  # LF_NO_DATABASE
                            error_msg += "\n  💡 [Hint] No database could be loaded from the provided data. The file might be empty or corrupted."
                        raise RuntimeError(error_msg)
            except IOError as e:
                raise RuntimeError(f"Could not read custom database file: {custom_db_path}. Error: {e}")
    
    def __del__(self):
        if hasattr(self, 'db') and self.db:
            _lensfun.lf_db_destroy(self.db)
    
    def find_camera(self, maker: Optional[str], model: str) -> Optional[ctypes.POINTER(lfCamera)]:
        """查找相机"""
        maker_b = maker.encode('utf-8') if maker else None
        model_b = model.encode('utf-8')
        
        cameras = _lensfun.lf_db_find_cameras_ext(self.db, maker_b, model_b, 0)
        if cameras and cameras[0]:
            return cameras[0]
        return None
    
    def find_lens(self, camera: Optional[ctypes.POINTER(lfCamera)], 
                  maker: Optional[str], model: str) -> Optional[ctypes.POINTER(lfLens)]:
        """查找镜头"""
        maker_b = maker.encode('utf-8') if maker else None
        model_b = model.encode('utf-8')
        
        lenses = _lensfun.lf_db_find_lenses(self.db, camera, maker_b, model_b, 0)
        if lenses and lenses[0]:
            return lenses[0]
        return None


class LensfunModifier:
    """Lensfun校正修改器包装器"""
    
    def __init__(self, lens: ctypes.POINTER(lfLens), focal: float, crop: float,
                 width: int, height: int, pixel_format: int = LF_PF_F32, reverse: bool = False):
        if not _lensfun:
            raise RuntimeError("Lensfun library not loaded")
        
        self.modifier = _lensfun.lf_modifier_create(
            lens, focal, crop, width, height, pixel_format, int(reverse)
        )
        if not self.modifier:
            raise RuntimeError("Could not create lensfun modifier")
        
        self.width = width
        self.height = height
    
    def __del__(self):
        if hasattr(self, 'modifier') and self.modifier:
            _lensfun.lf_modifier_destroy(self.modifier)
    
    def enable_distortion_correction(self) -> int:
        """启用畸变校正"""
        return _lensfun.lf_modifier_enable_distortion_correction(self.modifier)
    
    def enable_tca_correction(self) -> int:
        """启用横向色差校正"""
        return _lensfun.lf_modifier_enable_tca_correction(self.modifier)
    
    def enable_vignetting_correction(self, aperture: float, distance: float = 1000.0) -> int:
        """启用暗角校正"""
        return _lensfun.lf_modifier_enable_vignetting_correction(
            self.modifier, aperture, distance
        )
    
    def enable_projection_transform(self, target_projection: int) -> int:
        """启用投影变换"""
        return _lensfun.lf_modifier_enable_projection_transform(
            self.modifier, target_projection
        )
    
    def enable_scaling(self, scale: float) -> int:
        """启用缩放"""
        return _lensfun.lf_modifier_enable_scaling(self.modifier, scale)

    def get_auto_scale(self) -> float:
        """获取自动缩放比例"""
        return _lensfun.lf_modifier_get_auto_scale(self.modifier)
    
    def apply_subpixel_geometry_distortion(self, xu: float, yu: float, 
                                           width: int, height: int) -> Optional[np.ndarray]:
        """应用子像素几何畸变校正
        
        返回: shape为 (height, width, 2, 3) 的数组，存储R/G/B三通道的(x,y)坐标
        """
        # 分配输出缓冲区: width * height * 2 * 3
        res_size = width * height * 2 * 3
        res = (ctypes.c_float * res_size)()
        
        result = _lensfun.lf_modifier_apply_subpixel_geometry_distortion(
            self.modifier, xu, yu, width, height, res
        )
        
        if result:
            # 转换为numpy数组并重塑
            arr = np.ctypeslib.as_array(res)
            return arr.reshape(height, width, 3, 2)  # (h, w, RGB, xy)
        return None
    
    def apply_color_modification(self, pixels: np.ndarray, x: float, y: float,
                                 width: int, height: int) -> bool:
        """应用颜色修改（暗角校正）
        
        参数:
            pixels: 像素数据，会被原地修改
        """
        # 确保数据类型正确
        if pixels.dtype != np.float32:
            raise ValueError("Pixel data must be of type float32")
        
        # 获取数据指针
        pixels_ptr = pixels.ctypes.data_as(ctypes.c_void_p)
        row_stride = width * pixels.shape[2] * pixels.itemsize
        
        result = _lensfun.lf_modifier_apply_color_modification(
            self.modifier, pixels_ptr, x, y, width, height, LF_CR_RGB, row_stride
        )
        
        return bool(result)


# ============================================================================
# 便捷函数
# ============================================================================

def apply_lens_correction(
    image: np.ndarray,
    camera_maker: Optional[str],
    camera_model: str,
    lens_maker: Optional[str],
    lens_model: str,
    focal_length: float,
    aperture: float,
    crop_factor: Optional[float] = None,
    correct_distortion: bool = True,
    correct_tca: bool = True,
    correct_vignetting: bool = True,
    distance: float = 1000.0,
    custom_db_path: Optional[str] = None,
    logger: callable = print,
) -> np.ndarray:
    """应用镜头校正到图像
    
    参数:
        image: 输入图像，shape为 (height, width, 3)，范围0-1
        camera_maker: 相机制造商
        camera_model: 相机型号
        lens_maker: 镜头制造商
        lens_model: 镜头型号
        focal_length: 焦距 (mm)
        aperture: 光圈值 (f-number)
        crop_factor: 裁剪系数，如果为None则从相机信息获取
        correct_distortion: 是否校正畸变
        correct_tca: 是否校正横向色差
        correct_vignetting: 是否校正暗角
        distance: 对焦距离 (米)
    
    返回:
        校正后的图像（与输入相同dtype）
    """
    if not _lensfun:
        logger("  ⚠️ [Lensfun] Library not loaded. Skipping lens correction.")
        return image
    
    # 记住原始dtype以便最后转换回去
    original_dtype = image.dtype
    
    # 转换为float32（如果不是的话）
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    height, width = image.shape[:2]
    
    # 创建数据库并查找相机和镜头
    db = LensfunDatabase(custom_db_path=custom_db_path, logger=logger)
    camera = db.find_camera(camera_maker, camera_model)
    lens = db.find_lens(camera, lens_maker, lens_model)
    
    if not lens:
        logger(f"  ⚠️ [Lensfun] Lens not found: {lens_maker} {lens_model}. Skipping correction.")
        return image
    
    # 确定裁剪系数
    if crop_factor is None:
        if camera:
            # 从相机对象获取crop factor (需要访问C结构体成员)
            # 简化处理：使用默认值1.0
            crop_factor = 1.0
        else:
            crop_factor = 1.0
    
    # 创建修改器
    modifier = LensfunModifier(lens, focal_length, crop_factor, width, height, LF_PF_F32)
    
    # 启用所需的校正并应用自动缩放
    if correct_distortion:
        modifier.enable_distortion_correction()
        # 获取并应用自动缩放以消除黑边
        auto_scale = modifier.get_auto_scale()
        modifier.enable_scaling(1.0/auto_scale)
        logger(f"  ⚖️ [Lensfun] Auto-scaling enabled with factor: {auto_scale:.4f}")

    if correct_tca:
        modifier.enable_tca_correction()
    
    if correct_vignetting:
        modifier.enable_vignetting_correction(aperture, distance)
    
    # 创建输出图像
    output = np.zeros_like(image)
    
    # 步骤1: 应用颜色修改（暗角）
    # 这是原位操作，会直接修改 image 数组。
    # 后续的几何校正会从这个修改后的 image 中读取数据，所以这是期望的行为。
    if correct_vignetting:
        modifier.apply_color_modification(image, 0.0, 0.0, width, height)
    
    # 步骤2: 应用几何畸变和TCA校正
    if correct_distortion or correct_tca:
        coords = modifier.apply_subpixel_geometry_distortion(0.0, 0.0, width, height)
        
        if coords is not None:
            # 使用scipy的map_coordinates进行插值
            from scipy.ndimage import map_coordinates
            
            for c in range(3):  # R, G, B
                coords_c = coords[:, :, c, :]
                coordinates = np.array([coords_c[:, :, 1], coords_c[:, :, 0]])
                
                output[:, :, c] = map_coordinates(
                    image[:, :, c],
                    coordinates,
                    order=3,
                    mode='constant',
                    cval=0.0
                )
        else:
            output = image
    else:
        output = image
    
    # 转换回原始dtype
    if output.dtype != original_dtype:
        output = output.astype(original_dtype)
    
    return output

