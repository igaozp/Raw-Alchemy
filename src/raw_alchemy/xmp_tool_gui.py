# -*- coding: utf-8 -*-
"""
GUI Tool for batch-generating XMP profiles from LUTs.

This module provides a separate window (`Toplevel`) that allows users to:
1. Select a target Log space.
2. Select one or more .cube LUT files, or a folder containing them.
3. Choose an output directory to save the generated .xmp profiles.
4. Start the generation process.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading

# Handle imports to allow running as a script or a package
try:
    from . import core, xmp_generator
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from raw_alchemy import core, xmp_generator

class XMPToolWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("XMP Profile Generator")
        self.geometry("700x250") # Further reduced height
        self.transient(master) # Keep this window on top of the main app
        self.grab_set() # Modal behavior

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # --- Settings Frame ---
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding=10)
        settings_frame.pack(fill="x", pady=5)

        # Log Space
        ttk.Label(settings_frame, text="Target Log Space:").grid(row=0, column=0, sticky="w", pady=5)
        self.log_space_var = tk.StringVar(value=list(core.LOG_TO_WORKING_SPACE.keys())[0])
        ttk.OptionMenu(settings_frame, self.log_space_var, self.log_space_var.get(), *core.LOG_TO_WORKING_SPACE.keys()).grid(row=0, column=1, sticky="ew", padx=5)

        # LUT Input
        ttk.Label(settings_frame, text="LUT File/Folder:").grid(row=1, column=0, sticky="w", pady=5)
        self.lut_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.lut_path_var).grid(row=1, column=1, sticky="ew", padx=5)
        
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.grid(row=1, column=2, padx=5)
        ttk.Button(btn_frame, text="File...", command=self.browse_lut_file).pack(side="left")
        ttk.Button(btn_frame, text="Folder...", command=self.browse_lut_folder).pack(side="left", padx=2)

        # Output Directory
        ttk.Label(settings_frame, text="Output Folder:").grid(row=2, column=0, sticky="w", pady=5)
        self.output_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.output_path_var).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_output_folder).grid(row=2, column=2, padx=5)

        settings_frame.columnconfigure(1, weight=1)

        # --- Action Frame ---
        action_frame = ttk.Frame(main_frame, padding=10)
        action_frame.pack(fill="x")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.start_button = ttk.Button(action_frame, text="Start Generation", command=self.start_generation_thread)
        self.start_button.pack(side="right")

    def browse_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_path_var.set(path)

    def browse_lut_file(self):
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube")])
        if path:
            self.lut_path_var.set(path)

    def browse_lut_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.lut_path_var.set(path)

    def start_generation_thread(self):
        if not self.lut_path_var.get():
            messagebox.showerror("Error", "Please select a LUT file or folder.")
            return
        if not self.output_path_var.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return

        self.start_button.config(state="disabled")
        self.progress_var.set(0)
        
        t = threading.Thread(target=self.run_generation)
        t.daemon = True
        t.start()

    def run_generation(self):
        log_space = self.log_space_var.get()
        output_dir = self.output_path_var.get()
        input_path = self.lut_path_var.get()

        luts_to_process = []
        if os.path.isfile(input_path) and input_path.lower().endswith('.cube'):
            luts_to_process.append(input_path)
        elif os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(".cube"):
                    luts_to_process.append(os.path.join(input_path, filename))
        
        if not luts_to_process:
            self.master.after(0, lambda: messagebox.showerror("Error", "No .cube files found in the specified path."))
            self.master.after(0, lambda: self.start_button.config(state="normal"))
            return

        total_luts = len(luts_to_process)
        for i, lut_path in enumerate(luts_to_process):
            try:
                lut_name = os.path.basename(os.path.splitext(lut_path)[0])
                profile_name = f"RA - {log_space} - {lut_name}"
                
                xmp_content = xmp_generator.create_xmp_profile(
                    profile_name=profile_name,
                    log_space=log_space,
                    lut_path=lut_path,
                )
                
                output_filename = os.path.join(output_dir, f"{profile_name}.xmp")
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(xmp_content)

            except Exception as e:
                print(f"Error generating XMP for {lut_path}: {e}")
            
            progress = ((i + 1) / total_luts) * 100
            self.progress_var.set(progress)

        self.master.after(0, self.generation_finished, total_luts)

    def generation_finished(self, count):
        self.start_button.config(state="normal")
        messagebox.showinfo("Success", f"Generated {count} XMP profiles in:\n{self.output_path_var.get()}")
        self.destroy()

if __name__ == '__main__':
    # Example of how to launch this window from a main app
    root = tk.Tk()
    root.title("Main App")
    
    def open_tool():
        tool_window = XMPToolWindow(root)
    
    ttk.Button(root, text="Open XMP Generator", command=open_tool).pack(pady=20)
    
    root.mainloop()