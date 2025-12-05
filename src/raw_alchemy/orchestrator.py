import os
import concurrent.futures
from raw_alchemy import core

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
):
    """
    Orchestrates the processing of a single file or a directory of files.

    This function contains the common logic for both CLI and GUI to determine
    whether to run a single file process or a batch process.
    """
    # Helper to handle both queue.put and direct function calls for logging
    def log_message(msg):
        if hasattr(logger_func, 'put'):
            logger_func.put(msg)
        else:
            logger_func(msg)
    if os.path.isdir(input_path):
        # --- Batch Processing (Parallel) ---
        if not os.path.isdir(output_path):
            error_msg = "For batch processing, the output path must be a directory."
            log_message(f"❌ Error: {error_msg}")
            raise ValueError(error_msg)

        raw_files = []
        for ext in SUPPORTED_RAW_EXTENSIONS:
            raw_files.extend([f for f in os.listdir(input_path) if f.lower().endswith(ext)])

        if not raw_files:
            log_message("⚠️ No supported RAW files found in the input directory.")
            raise ValueError("No RAW files found.")

        log_message(f"🔍 Found {len(raw_files)} RAW files for parallel processing.")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    core.process_image,
                    raw_path=os.path.join(input_path, filename),
                    output_path=os.path.join(output_path, f"{os.path.splitext(filename)[0]}.tif"),
                    log_space=log_space,
                    lut_path=lut_path,
                    exposure=exposure,
                    lens_correct=lens_correct,
                    custom_db_path=custom_db_path,
                    metering_mode=metering_mode,
                    log_queue=logger_func if hasattr(logger_func, 'put') else None # Pass queue directly if it is one
                ): filename for filename in raw_files
            }
            
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                try:
                    future.result()  # Check for exceptions from the process
                except Exception as exc:
                    # GUI expects a dict, CLI expects a string. We'll log a structured-like string.
                    log_msg = f"[{filename}] ❌ Generated an exception: {exc}"
                    if hasattr(logger_func, 'put'):
                        logger_func.put({'id': filename, 'msg': f'❌ Generated an exception: {exc}'})
                    else:
                        log_message(log_msg)
        
        log_message("\n🎉 Batch processing complete.")

    else:
        # --- Single File Processing ---
        final_output_path = output_path
        if os.path.isdir(output_path):
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            final_output_path = os.path.join(output_path, f"{file_name}.tif")
        
        log_message("⚙️ Processing single file...")
        core.process_image(
            raw_path=input_path,
            output_path=final_output_path,
            log_space=log_space,
            lut_path=lut_path,
            exposure=exposure,
            lens_correct=lens_correct,
            custom_db_path=custom_db_path,
            metering_mode=metering_mode,
            log_queue=logger_func if hasattr(logger_func, 'put') else None
        )
        log_message("\n🎉 Single file processing complete.")