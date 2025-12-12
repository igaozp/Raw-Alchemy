import os
import concurrent.futures
from . import core

# Supported RAW file extensions (lowercase)
SUPPORTED_RAW_EXTENSIONS = [
    '.dng', '.cr2', '.cr3', '.nef', '.arw', '.rw2', '.raf', '.orf', '.pef', '.srw'
]

def process_path(
    input_path,
    output_path,
    log_space,
    lut_path,
    exposure,
    lens_correct,
    custom_db_path,
    metering_mode,
    jobs,
    logger_func, # A function to handle logging, e.g., print or queue.put
    output_format: str = 'tif',
    generate_tiff_only: bool = False,
    generate_xmp_profile: bool = False,
):
    """
    Orchestrates the processing of a single file or a directory of files.
    Supports different processing modes based on user selection.
    """
    
    def log_message(msg):
        if hasattr(logger_func, 'put'):
            logger_func.put(msg)
        else:
            logger_func(msg)

    def send_signal(data):
        if hasattr(logger_func, 'put'):
            logger_func.put(data)

    # --- Task Definition ---
    is_batch = os.path.isdir(input_path)
    
    if is_batch:
        if not os.path.isdir(output_path):
            error_msg = "For batch processing, the output path must be a directory."
            log_message(f"❌ Error: {error_msg}")
            raise ValueError(error_msg)
        
        raw_files = [
            f for ext in SUPPORTED_RAW_EXTENSIONS
            for f in os.listdir(input_path) if f.lower().endswith(ext)
        ]

        if not raw_files:
            log_message("⚠️ No supported RAW files found in the input directory.")
            return # Don't raise an error, just inform the user.
        
        count = len(raw_files)
        log_message(f"🔍 Found {count} RAW files for parallel processing.")
        send_signal({'total_files': count})

        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {}
            for filename in raw_files:
                common_kwargs = {
                    'raw_path': os.path.join(input_path, filename),
                    'exposure': exposure,
                    'lens_correct': lens_correct,
                    'custom_db_path': custom_db_path,
                    'metering_mode': metering_mode,
                    'log_queue': logger_func if hasattr(logger_func, 'put') else None
                }

                if generate_xmp_profile:
                    target_func = core.process_with_xmp
                    task_kwargs = {
                        **common_kwargs,
                        'output_path': os.path.join(output_path, os.path.splitext(filename)[0]),
                        'log_space': log_space,
                        'lut_path': lut_path,
                    }
                elif generate_tiff_only:
                    target_func = core.generate_prophoto_tiff
                    task_kwargs = {
                        **common_kwargs,
                        'output_path': os.path.join(output_path, f"{os.path.splitext(filename)[0]}.tif"),
                    }
                else:
                    target_func = core.process_image
                    task_kwargs = {
                        **common_kwargs,
                        'output_path': os.path.join(output_path, f"{os.path.splitext(filename)[0]}.{output_format}"),
                        'log_space': log_space,
                        'lut_path': lut_path,
                    }
                
                future = executor.submit(target_func, **task_kwargs)
                futures[future] = filename

            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    log_msg = f"❌ Generated an exception: {exc}"
                    if hasattr(logger_func, 'put'):
                        logger_func.put({'id': filename, 'msg': log_msg})
                    else:
                        log_message(f"[{filename}] {log_msg}")
                finally:
                    send_signal({'status': 'done'})
        
        log_message("\n🎉 Batch processing complete.")

    else: # Single file processing
        send_signal({'total_files': 1})
        log_message("⚙️ Processing single file...")

        common_kwargs = {
            'raw_path': input_path,
            'exposure': exposure,
            'lens_correct': lens_correct,
            'custom_db_path': custom_db_path,
            'metering_mode': metering_mode,
            'log_queue': logger_func if hasattr(logger_func, 'put') else None
        }

        try:
            if generate_xmp_profile:
                # For single file, output_path can be a file base name or a directory
                if os.path.isdir(output_path):
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    output_base = os.path.join(output_path, base_name)
                else:
                    output_base = os.path.splitext(output_path)[0]

                core.process_with_xmp(
                    **common_kwargs,
                    output_path=output_base,
                    log_space=log_space,
                    lut_path=lut_path,
                )
            elif generate_tiff_only:
                if os.path.isdir(output_path):
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    final_output_path = os.path.join(output_path, f"{base_name}.tif")
                else:
                    final_output_path = f"{os.path.splitext(output_path)[0]}.tif"
                
                core.generate_prophoto_tiff(
                    **common_kwargs,
                    output_path=final_output_path,
                )
            else:
                if os.path.isdir(output_path):
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    final_output_path = os.path.join(output_path, f"{base_name}.{output_format}")
                else:
                    final_output_path = output_path

                core.process_image(
                    **common_kwargs,
                    output_path=final_output_path,
                    log_space=log_space,
                    lut_path=lut_path,
                )
        finally:
            send_signal({'status': 'done'})
            log_message("\n🎉 Single file processing complete.")