import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import os
import threading
import queue
import multiprocessing
import sys
import time

# 假设 core 和 orchestrator 在 raw_alchemy 包中
from raw_alchemy import core, orchestrator
from raw_alchemy.orchestrator import SUPPORTED_RAW_EXTENSIONS

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class GuiApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Raw Alchemy")
        self.master.geometry("1000x950")
        
        # --- Icon Setting ---
        try:
            if sys.platform.startswith('win'):
                icon_path = resource_path("icon.ico")
                if os.path.exists(icon_path): self.master.iconbitmap(icon_path)
            else:
                icon_path = resource_path("icon.png")
                if os.path.exists(icon_path):
                    icon_image = tk.PhotoImage(file=icon_path)
                    self.master.iconphoto(True, icon_image)
        except Exception as e:
            print(f"Icon load warning: {e}")

        self.pack(fill="both", expand=True)
        
        # --- Queues ---
        # gui_queue: 仅用于主线程读取，更新 UI
        self.gui_queue = queue.Queue()
        
        self.create_widgets()
        
        # Start the GUI update loop
        self.master.after(100, self.process_gui_queue)

    def create_widgets(self):
        # --- Frame for IO ---
        io_frame = ttk.LabelFrame(self, text="Input / Output Source", padding=(10, 5))
        io_frame.pack(padx=10, pady=5, fill="x")

        # Input
        ttk.Label(io_frame, text="Input Path:").grid(row=0, column=0, sticky="w")
        self.input_path_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.input_path_var, width=60).grid(row=0, column=1, padx=5, sticky="ew")
        
        btn_f1 = ttk.Frame(io_frame)
        btn_f1.grid(row=0, column=2, padx=5)
        ttk.Button(btn_f1, text="File...", command=self.browse_input_file).pack(side="left")
        ttk.Button(btn_f1, text="Folder...", command=self.browse_input_folder).pack(side="left", padx=2)

        # Output
        ttk.Label(io_frame, text="Output Path:").grid(row=1, column=0, sticky="w")
        self.output_path_var = tk.StringVar()
        ttk.Entry(io_frame, textvariable=self.output_path_var, width=60).grid(row=1, column=1, padx=5, sticky="ew")

        btn_f2 = ttk.Frame(io_frame)
        btn_f2.grid(row=1, column=2, padx=5)
        ttk.Button(btn_f2, text="Save As...", command=self.browse_output_file).pack(side="left")
        ttk.Button(btn_f2, text="Folder...", command=self.browse_output_folder).pack(side="left", padx=2)
        
        # Output Format
        ttk.Label(io_frame, text="Output Format:").grid(row=2, column=0, sticky="w", pady=5)
        self.output_format_var = tk.StringVar(value='tif')
        ttk.OptionMenu(io_frame, self.output_format_var, 'tif', 'tif', 'heif', 'jpg').grid(row=2, column=1, sticky="w", padx=5)
        self.output_format_var.trace_add("write", self.on_output_format_change)
        
        io_frame.columnconfigure(1, weight=1)

        # --- Frame for Processing Settings ---
        settings_frame = ttk.LabelFrame(self, text="Processing Core", padding=(10, 5))
        settings_frame.pack(padx=10, pady=5, fill="x")

        # Row 0: Log Space
        ttk.Label(settings_frame, text="Log Space:").grid(row=0, column=0, sticky="w", pady=5)
        self.log_space_var = tk.StringVar(value=list(core.LOG_TO_WORKING_SPACE.keys())[0])
        ttk.OptionMenu(settings_frame, self.log_space_var, self.log_space_var.get(), *core.LOG_TO_WORKING_SPACE.keys()).grid(row=0, column=1, columnspan=3, sticky="ew", padx=5)

        # Row 1: LUT
        ttk.Label(settings_frame, text="LUT (.cube):").grid(row=1, column=0, sticky="w", pady=5)
        self.lut_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.lut_path_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_lut).grid(row=1, column=3, sticky="ew", padx=5)

        # Row 2: Lensfun
        ttk.Label(settings_frame, text="Lens DB (XML):").grid(row=2, column=0, sticky="w", pady=5)
        self.custom_lensfun_db_path_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.custom_lensfun_db_path_var).grid(row=2, column=1, columnspan=2, sticky="ew", padx=5)
        ttk.Button(settings_frame, text="Browse...", command=self.browse_lensfun_db).grid(row=2, column=3, sticky="ew", padx=5)

        # Row 3: CPU Jobs
        ttk.Label(settings_frame, text="CPU Threads:").grid(row=3, column=0, sticky="w", pady=5)
        self.jobs_var = tk.IntVar(value=min(4, multiprocessing.cpu_count()))
        ttk.Spinbox(settings_frame, from_=1, to=multiprocessing.cpu_count(), textvariable=self.jobs_var, width=5).grid(row=3, column=1, sticky="w", padx=5)

        settings_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(2, weight=1)

        # --- Frame for Exposure ---
        exp_frame = ttk.LabelFrame(self, text="Exposure Control", padding=(10, 5))
        exp_frame.pack(padx=10, pady=5, fill="x")

        self.exposure_mode_var = tk.StringVar(value="Auto")
        
        # Use grid layout for two rows
        exp_frame.columnconfigure(1, weight=1)

        # --- Row 0: Auto ---
        auto_frame = ttk.Frame(exp_frame)
        auto_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        ttk.Radiobutton(auto_frame, text="Auto Exposure: ", variable=self.exposure_mode_var, value="Auto", command=self.toggle_exposure_controls).pack(side="left")
        
        self.auto_opts_frame = ttk.Frame(auto_frame)
        self.auto_opts_frame.pack(side="left", padx=10)
        self.metering_mode_var = tk.StringVar(value='matrix')
        ttk.OptionMenu(self.auto_opts_frame, self.metering_mode_var, 'matrix', *core.METERING_MODES).pack(side="left", padx=5)

        # --- Row 1: Manual ---
        manual_frame = ttk.Frame(exp_frame)
        manual_frame.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Radiobutton(manual_frame, text="Manual EV: ", variable=self.exposure_mode_var, value="Manual", command=self.toggle_exposure_controls).pack(side="left")

        self.manual_opts_frame = ttk.Frame(manual_frame)
        self.manual_opts_frame.pack(side="left", padx=10, fill="x", expand=True)
        
        self.exposure_stops_var = tk.DoubleVar(value=0.0)
        self.exposure_scale = ttk.Scale(self.manual_opts_frame, from_=-5.0, to=5.0, variable=self.exposure_stops_var, command=lambda v: self.exposure_stops_var.set(round(float(v), 2)))
        self.exposure_scale.pack(side="left", fill="x", expand=True)
        ttk.Entry(self.manual_opts_frame, textvariable=self.exposure_stops_var, width=6).pack(side="left", padx=5)
        
        self.toggle_exposure_controls()

        # --- Log & Progress ---
        log_frame = ttk.Frame(self)
        log_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, state="disabled", height=10, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        
        # Tags for coloring logs
        self.log_text.tag_config("ERROR", foreground="red")
        self.log_text.tag_config("SUCCESS", foreground="green")
        self.log_text.tag_config("INFO", foreground="blue")
        self.log_text.tag_config("ID", foreground="gray")

        # Progress Bar & Actions
        action_frame = ttk.Frame(self)
        action_frame.pack(padx=10, pady=10, fill="x")
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(side="left")
        
        self.progress_label = ttk.Label(action_frame, text="Ready", width=16, anchor='w')
        self.progress_label.pack(side="left", padx=(10, 0))

        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.pack(side="right")

    def on_output_format_change(self, *args):
        # 只有当输出路径是文件（有扩展名）时才自动替换扩展名
        # 如果是文件夹，则不动
        current_path = self.output_path_var.get()
        if not current_path: return
        
        if os.path.isdir(current_path):
            return # 是文件夹，不改
            
        root, ext = os.path.splitext(current_path)
        if ext: # 看起来像是个文件
            new_ext = self.output_format_var.get()
            # 简单映射
            if new_ext == 'heif': new_ext = '.heif'
            elif new_ext == 'tif': new_ext = '.tif'
            elif new_ext == 'jpg': new_ext = '.jpg'
            self.output_path_var.set(root + new_ext)

    def toggle_exposure_controls(self):
        mode = self.exposure_mode_var.get()
        if mode == "Auto":
            for child in self.auto_opts_frame.winfo_children(): child.configure(state="normal")
            for child in self.manual_opts_frame.winfo_children(): child.configure(state="disabled")
        else:
            for child in self.auto_opts_frame.winfo_children(): child.configure(state="disabled")
            for child in self.manual_opts_frame.winfo_children(): child.configure(state="normal")

    # --- Browsing ---
    def browse_input_file(self):
        exts = " ".join([f"*{e} *{e.upper()}" for e in SUPPORTED_RAW_EXTENSIONS])
        path = filedialog.askopenfilename(filetypes=[("RAW Images", exts), ("All Files", "*.*")])
        if path: self.input_path_var.set(path)

    def browse_input_folder(self):
        path = filedialog.askdirectory()
        if path: self.input_path_var.set(path)

    def browse_output_file(self):
        fmt = self.output_format_var.get()
        ext = f".{fmt}"
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(f"{fmt.upper()} Image", f"*{ext}")])
        if path: self.output_path_var.set(path)

    def browse_output_folder(self):
        path = filedialog.askdirectory()
        if path: self.output_path_var.set(path)

    def browse_lut(self):
        path = filedialog.askopenfilename(filetypes=[("LUT files", "*.cube")])
        if path: self.lut_path_var.set(path)

    def browse_lensfun_db(self):
        path = filedialog.askopenfilename(filetypes=[("Lensfun XML", "*.xml")])
        if path: self.custom_lensfun_db_path_var.set(path)

    # --- Logic ---

    def log_gui(self, msg, level="NORMAL", file_id=None):
        """将消息放入 GUI 队列"""
        self.gui_queue.put({
            'type': 'log', 
            'msg': msg, 
            'level': level, 
            'id': file_id
        })

    def update_progress(self, current, total):
        """将进度信息放入 GUI 队列"""
        self.gui_queue.put({
            'type': 'progress',
            'current': current,
            'total': total
        })

    def process_gui_queue(self):
        """主线程定时器：处理队列中的 UI 更新请求"""
        try:
            while True:
                item = self.gui_queue.get_nowait()
                
                if item['type'] == 'log':
                    self.log_text.config(state="normal")
                    
                    # 插入 ID (灰色)
                    if item.get('id'):
                        self.log_text.insert(tk.END, f"[{item['id']}] ", "ID")
                    
                    # 插入消息
                    tag = item.get('level', 'NORMAL')
                    self.log_text.insert(tk.END, f"{item['msg']}\n", tag)
                    
                    self.log_text.see(tk.END)
                    self.log_text.config(state="disabled")
                
                elif item['type'] == 'progress':
                    curr = item['current']
                    total = item['total']
                    pct = (curr / total) * 100 if total > 0 else 0
                    self.progress_var.set(pct)
                    self.progress_label.config(text=f"{curr}/{total}")

        except queue.Empty:
            pass
        finally:
            self.master.after(50, self.process_gui_queue)

    def start_processing_thread(self):
        if not self.input_path_var.get() or not self.output_path_var.get():
            messagebox.showerror("Error", "Please select both Input and Output paths.")
            return

        self.start_button.config(state="disabled")
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        self.progress_var.set(0)
        self.progress_label.config(text="Initializing...")
        
        # 启动工作线程
        t = threading.Thread(target=self.run_orchestrator)
        t.daemon = True
        t.start()

    def run_orchestrator(self):
        # 1. 收集参数
        params = {
            'input_path': self.input_path_var.get(),
            'output_path': self.output_path_var.get(),
            'log_space': self.log_space_var.get(),
            'output_format': self.output_format_var.get(),
            'lut_path': self.lut_path_var.get() or None,
            'custom_db_path': self.custom_lensfun_db_path_var.get() or None,
            'jobs': self.jobs_var.get(),
            'lens_correct': True
        }
        
        if self.exposure_mode_var.get() == "Manual":
            params['exposure'] = self.exposure_stops_var.get()
            params['metering_mode'] = None
        else:
            params['exposure'] = None
            params['metering_mode'] = self.metering_mode_var.get()

        # 2. 创建多进程通信桥梁
        manager = multiprocessing.Manager()
        mp_queue = manager.Queue()
        
        # 启动一个“监视线程”，负责把多进程队列的数据搬运到 Tkinter 队列
        monitor = threading.Thread(target=self.monitor_mp_queue, args=(mp_queue,))
        monitor.daemon = True
        monitor.start()

        try:
            # 3. 调用 Orchestrator (假设 orchestrator 已更新以支持 output_format)
            # 注意：这里我们传递 mp_queue 进去
            orchestrator.process_path(
                **params,
                logger_func=mp_queue 
            )
            self.log_gui("All tasks completed.", "SUCCESS")
            
        except Exception as e:
            self.log_gui(f"Critical Error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复按钮 (使用 after 确保在主线程执行)
            self.master.after(0, lambda: self.start_button.config(state="normal"))
            # 发送结束信号给监视线程
            mp_queue.put(None) 

    def monitor_mp_queue(self, mp_q):
        """
        后台线程：从 Multiprocessing Queue 读取数据，
        转发给 Tkinter Queue。
        """
        processed_count = 0
        total_files = 0
        
        while True:
            try:
                item = mp_q.get()
                if item is None: break # 结束信号

                # 处理不同类型的信号
                if isinstance(item, dict):
                    # 1. 结构化日志
                    if 'msg' in item:
                        level = "ERROR" if "Error" in item['msg'] else "NORMAL"
                        self.gui_queue.put({
                            'type': 'log',
                            'msg': item['msg'],
                            'id': item.get('id'),
                            'level': level
                        })
                    
                    # 2. 进度初始化信号 (Orchestrator 需要发这个)
                    if 'total_files' in item:
                        total_files = item['total_files']
                        self.update_progress(0, total_files)
                        
                    # 3. 完成信号
                    if 'status' in item and item['status'] == 'done':
                        processed_count += 1
                        self.update_progress(processed_count, total_files)

                else:
                    # 简单的字符串日志
                    self.gui_queue.put({'type': 'log', 'msg': str(item)})

            except Exception:
                break

def launch_gui():
    multiprocessing.freeze_support()
    root = tk.Tk()
    # 尝试设置高 DPI 缩放 (Windows)
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    
    app = GuiApplication(master=root)
    app.mainloop()

if __name__ == "__main__":
    launch_gui()