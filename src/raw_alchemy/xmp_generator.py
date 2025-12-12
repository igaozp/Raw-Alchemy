# -*- coding: utf-8 -*-
"""
Adobe XMP Profile Generator (Full Feature)

Features:
1. Color Space Transform (Linear ProPhoto -> Target Log -> User LUT).
2. Tetrahedral Interpolation for high-quality resizing.
3. Adobe RGBTable Binary Format (Delta Encoded, Zlib Compressed, Base85).
4. Full range amount slider (0-200%).

Dependencies: pip install colour-science numpy
"""

import base64
import hashlib
import struct
import time
import uuid
import zlib
from io import BytesIO

import numpy as np
import colour
try:
    from .constants import LOG_ENCODING_MAP, LOG_TO_WORKING_SPACE, METERING_MODES
except ImportError:
    from constants import LOG_ENCODING_MAP, LOG_TO_WORKING_SPACE, METERING_MODES

# --- Constants & Mappings ---

# Adobe Custom Base85 Characters
ADOBE_Z85_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?`'|()[]{}@%$#"

# --- Core Logic ---

def adobe_base85_encode(data: bytes) -> str:
    """Encodes binary data into Adobe's custom Base85 format."""
    length = len(data)
    encoded_chars = []
    
    for i in range(0, length, 4):
        chunk = data[i : i + 4]
        if len(chunk) < 4:
            chunk = chunk + b'\x00' * (4 - len(chunk))
        
        val = struct.unpack('<I', chunk)[0]
        
        # DNG SDK Logic: val / 85 ... output in reverse modulus order
        # Actually DNG outputs: c1, c2, c3, c4, c5 where c1 is val % 85
        for _ in range(5):
            encoded_chars.append(ADOBE_Z85_CHARS[val % 85])
            val //= 85
            
    if length % 4 == 0:
        return "".join(encoded_chars)
    else:
        # Padding logic matching DNG SDK
        rem = length % 4
        # 1 byte -> 2 chars, 2 bytes -> 3 chars, 3 bytes -> 4 chars
        needed_chars = (len(data) // 4) * 5 + (rem + 1)
        return "".join(encoded_chars[:needed_chars])

def int_round(arr):
    """Matches C++ int_round: floor(n + 0.5)"""
    return np.floor(arr + 0.5).astype(np.int32)

def tetrahedral_resample(data, input_size, output_size):
    """
    Performs Tetrahedral Interpolation on a 3D LUT (numpy array).
    data shape: (B, G, R, 3) or (Z, Y, X, 3)
    """
    if input_size == output_size:
        return data

    ratio = (input_size - 1.0) / (output_size - 1.0)
    
    # Generate output grid coordinates (0..output_size-1)
    grid = np.arange(output_size, dtype=np.float64)
    # Create 3D mesh: Z(Blue), Y(Green), X(Red)
    zz, yy, xx = np.meshgrid(grid, grid, grid, indexing='ij')
    
    # Map to input coordinates
    bs = zz * ratio
    gs = yy * ratio
    rs = xx * ratio

    # Lower bounds
    lb = np.clip(np.floor(bs).astype(np.int32), 0, input_size - 1)
    lg = np.clip(np.floor(gs).astype(np.int32), 0, input_size - 1)
    lr = np.clip(np.floor(rs).astype(np.int32), 0, input_size - 1)

    # Upper bounds
    ub = np.clip(lb + 1, 0, input_size - 1)
    ug = np.clip(lg + 1, 0, input_size - 1)
    ur = np.clip(lr + 1, 0, input_size - 1)

    # Fractions
    fb = bs - lb
    fg = gs - lg
    fr = rs - lr

    # Helper for readability
    def get(z, y, x): return data[z, y, x]

    # 8 Corners
    c000 = get(lb, lg, lr); c100 = get(ub, lg, lr) # Note: ub is axis 0 (Blue) change
    c010 = get(lb, ug, lr); c110 = get(ub, ug, lr)
    c001 = get(lb, lg, ur); c101 = get(ub, lg, ur) # ur is axis 2 (Red) change
    c011 = get(lb, ug, ur); c111 = get(ub, ug, ur)

    # Note on axes: Since input `data` is (B, G, R), 
    # fb corresponds to axis 0, fg to axis 1, fr to axis 2.
    f0, f1, f2 = fb[..., None], fg[..., None], fr[..., None]
    
    # Tetrahedral Logic (6 cases)
    # 1. f0 >= f1 >= f2 (Blue >= Green >= Red)
    mask = (f0 >= f1) & (f1 >= f2); m = mask[..., None]
    out = m * ((1-f0)*c000 + (f0-f1)*c100 + (f1-f2)*c110 + f2*c111)
    
    # 2. f0 > f2 > f1
    mask = (f0 > f2) & (f2 > f1); m = mask[..., None]
    out += m * ((1-f0)*c000 + (f0-f2)*c100 + (f2-f1)*c101 + f1*c111)
    
    # 3. f2 > f0 > f1
    mask = (f2 > f0) & (f0 > f1); m = mask[..., None]
    out += m * ((1-f2)*c000 + (f2-f0)*c001 + (f0-f1)*c101 + f1*c111)
    
    # 4. f2 > f1 >= f0
    mask = (f2 > f1) & (f1 >= f0); m = mask[..., None]
    out += m * ((1-f2)*c000 + (f2-f1)*c001 + (f1-f0)*c011 + f0*c111)
    
    # 5. f1 >= f2 > f0
    mask = (f1 >= f2) & (f2 > f0); m = mask[..., None]
    out += m * ((1-f1)*c000 + (f1-f2)*c010 + (f2-f0)*c011 + f0*c111)
    
    # 6. f1 > f0 >= f2
    mask = (f1 > f0) & (f0 >= f2); m = mask[..., None]
    out += m * ((1-f1)*c000 + (f1-f0)*c010 + (f0-f2)*c110 + f2*c111)

    return out

def apply_cst_pipeline(user_lut_path, log_space, output_size=33):
    """
    Loads user LUT, creates ProPhoto Identity, transforms to Log, applies LUT.
    Returns: (output_size, final_data_numpy)
    """
    print(f"Processing: Reading {user_lut_path}...")
    user_lut = colour.read_LUT(user_lut_path)
    # Ensure standard (B, G, R, 3) shape for 3D LUTs
    # colour-science reads .cube as (Size, Size, Size, 3) usually.
    
    # 1. Create Identity Grid in Linear ProPhoto RGB (ACR Working Space)
    # We create it directly at the target output size to avoid resizing later if possible
    # But for accuracy, we usually want to process at the LUT's native size then downsample, 
    # OR create identity at 32x32x32 directly. 
    # ACR standard is 32 (or 33). Let's use 32 directly for the grid.
    
    domain = np.linspace(0, 1, output_size)
    # Grid shape: (B, G, R, 3) -> (32, 32, 32, 3)
    # Note: meshgrid indexing 'ij' gives Z, Y, X order
    B, G, R = np.meshgrid(domain, domain, domain, indexing='ij')
    prophoto_linear = np.stack([R, G, B], axis=-1) # Stack as RGB for color math
    
    log_color_space_name = LOG_TO_WORKING_SPACE.get(log_space)
    log_curve_name = LOG_ENCODING_MAP.get(log_space, log_space)

    print(f"  - Pipeline: ProPhoto Linear -> {log_color_space_name} -> {log_curve_name} -> LUT")
        
    # A. Gamut Transform: ProPhoto RGB -> Target Gamut (Linear)
    matrix = colour.matrix_RGB_to_RGB(
        colour.RGB_COLOURSPACES['ProPhoto RGB'],
        colour.RGB_COLOURSPACES[log_color_space_name]
    )
    # Apply matrix (dot product on last axis)
    target_gamut_linear = np.einsum('...ij,...j->...i', matrix, prophoto_linear)
    target_gamut_linear = np.maximum(target_gamut_linear, 1e-7)
    # B. Transfer Function: Linear -> Log
    log_encoded = colour.cctf_encoding(target_gamut_linear, function=log_curve_name)
        
    # C. Apply User LUT
    # Since our grid is 33x33x33 but user LUT might be 33x33x33 or 65x65x65,
    # we interpolate the user LUT at the log_encoded coordinates.
    print(f"  - Applying User LUT ({user_lut.size}^3) to grid...")
    final_rgb = user_lut.apply(log_encoded, interpolator=colour.algebra.table_interpolation_tetrahedral)
    
    # --- Debug Feature: Output Pipeline Cube ---
    try:
        # Generate a unique debug filename or overwrite a standard one
        debug_filename = f"debug_pipeline_{output_size}.cube"
        print(f"  [DEBUG] Writing pipeline output to {debug_filename}...")
        
        # Create a LUT3D object for export
        # Note: Ensure the data is clamped if necessary, but usually we want to see raw output
        debug_lut = colour.LUT3D(table=final_rgb, name=f"Debug Pipeline {log_space}")
        colour.write_LUT(debug_lut, debug_filename)
        print(f"  [DEBUG] Successfully wrote {debug_filename}")
    except Exception as e:
        print(f"  [DEBUG] Failed to write debug cube: {e}")
    # -------------------------------------------

    return output_size, final_rgb

def generate_rgb_table_stream(data, size, min_amt=0, max_amt=200):
    """
    Encodes the numpy data into DNG RGBTable binary format.
    data shape expected: (B, G, R, 3) where last dim is [r_val, g_val, b_val]
    """
    stream = BytesIO()
    def write_u32(val): stream.write(struct.pack('<I', val))
    def write_double(val): stream.write(struct.pack('<d', val))
    
    # Header
    write_u32(1) # btt_RGBTable (MUST BE 1)
    write_u32(1) # Version
    write_u32(3) # Dimensions
    write_u32(size) # Divisions
    
    # Data Processing
    # 1. Clip and Scale
    data = np.clip(data, 0.0, 1.0)
    data_u16 = int_round(data * 65535)
    
    # 2. Prepare Identity (Nop) Curve
    indices = np.arange(size, dtype=np.int32)
    nop_curve = (indices * 0xFFFF + (size >> 1)) // (size - 1)
    
    # 3. Prepare Nop Grid matching Data Shape (B, G, R)
    # axis 0=B, 1=G, 2=R.
    grid_b, grid_g, grid_r = np.meshgrid(nop_curve, nop_curve, nop_curve, indexing='ij')
    
    # 4. Calculate Deltas (Sample - Identity)
    # Data is [R, G, B] values.
    # delta_r = val_r - grid_r (where grid_r varies along axis 2)
    delta_r = data_u16[..., 0] - grid_r
    delta_g = data_u16[..., 1] - grid_g
    delta_b = data_u16[..., 2] - grid_b
    
    # 5. Reorder for DNG Loop: R(outer), G, B(inner)
    # Current shape (B, G, R).
    # We want flattened order equivalent to: for r: for g: for b: write(r,g,b)
    # So we need to transpose dimensions to (R, G, B) before flattening.
    # Current axes: 0=B, 1=G, 2=R. Target: 2, 1, 0.
    delta_r = delta_r.transpose(2, 1, 0)
    delta_g = delta_g.transpose(2, 1, 0)
    delta_b = delta_b.transpose(2, 1, 0)
    
    # Stack [DeltaR, DeltaG, DeltaB]
    deltas = np.stack((delta_r, delta_g, delta_b), axis=-1)
    
    # Flatten
    flat_deltas = deltas.flatten().astype(np.uint16)
    
    # Write Payload
    if struct.pack('<H', 1) == b'\x01\x00':
        stream.write(flat_deltas.tobytes())
    else:
        stream.write(flat_deltas.byteswap().tobytes())
        
    # Footer
    write_u32(0) # ProPhoto
    write_u32(1) # Linear Gamma (Since we baked the curve in)
    write_u32(0) # Gamut Extend
    write_double(min_amt * 0.01) # 0.0
    write_double(max_amt * 0.01) # 2.0
    
    return stream.getvalue()

def create_xmp_profile(profile_name, lut_path, log_space=None):
    """
    Main Entry Point.
    """
    # 1. Generate UUIDs
    profile_uuid = str(uuid.uuid4()).replace('-', '').upper()
    
    try:
        # 2. Process Color Pipeline & Resizing
        # Target size 32 is standard for ACR RGBTable
        size, data = apply_cst_pipeline(lut_path, log_space, output_size=33)
        
        # 3. Binary Encoding
        raw_bytes = generate_rgb_table_stream(data, size, min_amt=0, max_amt=200)
        
        # 4. Fingerprinting
        m = hashlib.md5()
        m.update(raw_bytes)
        fingerprint = m.hexdigest().upper()
        
        # 5. Compression & ASCII Encoding
        # 4-byte LE size header
        header = struct.pack('<I', len(raw_bytes))
        compressed = zlib.compress(raw_bytes, level=zlib.Z_DEFAULT_COMPRESSION)
        encoded_data = adobe_base85_encode(header + compressed)
        
    except Exception as e:
        print(f"Error creating profile: {e}")
        return ""

    # 6. XML Generation (Matches testabc.txt)
    xmp_template = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        ">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   crs:PresetType="Look"
   crs:Cluster=""
   crs:UUID="{profile_uuid}"
   crs:SupportsAmount="True"
   crs:SupportsColor="True"
   crs:SupportsMonochrome="True"
   crs:SupportsHighDynamicRange="True"
   crs:SupportsNormalDynamicRange="True"
   crs:SupportsSceneReferred="True"
   crs:SupportsOutputReferred="True"
   crs:RequiresRGBTables="False"
   crs:ShowInPresets="True"
   crs:ShowInQuickActions="False"
   crs:CameraModelRestriction=""
   crs:Copyright=""
   crs:ContactInfo=""
   crs:Version="14.3"
   crs:ProcessVersion="11.0"
   crs:ConvertToGrayscale="False"
   crs:RGBTable="{fingerprint}"
   crs:Table_{fingerprint}="{encoded_data}"
   crs:HasSettings="True">
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{profile_name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:ShortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:ShortName>
   <crs:SortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default"/>
    </rdf:Alt>
   </crs:SortName>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">Profiles</rdf:li>
    </rdf:Alt>
   </crs:Group>
   <crs:Description>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">Generated by xmp_generator</rdf:li>
    </rdf:Alt>
   </crs:Description>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""
    return xmp_template
