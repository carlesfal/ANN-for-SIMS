"""Main Tkinter GUI application for ANNSIMS."""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

from annsims.config import PipelineConfig


class ANNSIMSApp:
    """Main desktop application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ANNSIMS - ANN for SIMS Prediction")
        self.root.geometry("950x700")
        self.root.minsize(800, 600)

        self.config = PipelineConfig()
        self.data_path = tk.StringVar(value="")
        self.running = False
        self._cancel_flag = False
        self._runner = None

        self._build_ui()

    def _build_ui(self):
        """Build the main UI with notebook tabs."""
        # Menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Data...", command=self._browse_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tabs
        self._build_config_tab()
        self._build_data_tab()
        self._build_training_tab()
        self._build_results_tab()

    # ------------------------------------------------------------------
    # CONFIG TAB
    # ------------------------------------------------------------------
    def _build_config_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Configuration")

        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        self.config_vars = {}
        row = 0

        # Helper to add labeled entries
        def add_section(parent, title, r):
            ttk.Label(parent, text=title, font=("", 10, "bold")).grid(
                row=r, column=0, columnspan=4, sticky="w", pady=(10, 3), padx=5)
            return r + 1

        def add_entry(parent, label, attr, r, col=0, width=12):
            ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(getattr(self.config, attr)))
            entry = ttk.Entry(parent, textvariable=var, width=width)
            entry.grid(row=r, column=col + 1, sticky="w", padx=5, pady=2)
            self.config_vars[attr] = var
            return r

        def add_check(parent, label, attr, r, col=0):
            var = tk.BooleanVar(value=getattr(self.config, attr))
            cb = ttk.Checkbutton(parent, text=label, variable=var)
            cb.grid(row=r, column=col, columnspan=2, sticky="w", padx=5, pady=2)
            self.config_vars[attr] = var
            return r

        def add_combo(parent, label, attr, values, r, col=0):
            ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(getattr(self.config, attr)))
            combo = ttk.Combobox(parent, textvariable=var, values=values, width=14, state="readonly")
            combo.grid(row=r, column=col + 1, sticky="w", padx=5, pady=2)
            self.config_vars[attr] = var
            return r

        f = scroll_frame

        # Hill Pre-training
        row = add_section(f, "Hill Pre-training", row)
        row = add_entry(f, "N Inputs:", "n_inputs", row) + 1
        row = add_entry(f, "N Synthetic:", "n_synthetic", row) + 1
        row = add_entry(f, "Hill V_max:", "hill_v_max", row) + 1
        row = add_entry(f, "Hill K:", "hill_k", row) + 1
        row = add_entry(f, "Hill N:", "hill_n", row) + 1
        row = add_entry(f, "Hill X_min:", "hill_x_min", row) + 1
        row = add_entry(f, "Hill X_max:", "hill_x_max", row) + 1
        row = add_entry(f, "Pretrain Epochs:", "pretrain_epochs", row) + 1

        # Data Split
        row = add_section(f, "Data Split", row)
        row = add_entry(f, "Train %:", "train_percent", row) + 1
        row = add_entry(f, "Val %:", "val_percent", row) + 1
        row = add_entry(f, "Test %:", "test_percent", row) + 1
        row = add_entry(f, "N Labels:", "n_labels", row) + 1
        row = add_entry(f, "Target Col (-1=auto):", "target_col", row) + 1
        row = add_combo(f, "Separator:", "sep", [r"\t", ",", ";", " "], row) + 1

        # Training
        row = add_section(f, "Training", row)
        row = add_check(f, "Disable GPU", "disable_gpu", row) + 1
        row = add_entry(f, "Tuner Trials:", "tuner_trials", row) + 1
        row = add_entry(f, "K Folds:", "k_folds", row) + 1
        row = add_entry(f, "Random Seed:", "random_seed", row) + 1
        row = add_entry(f, "Tuner Epochs:", "tuner_epochs", row) + 1
        row = add_entry(f, "CV Epochs:", "cv_epochs", row) + 1
        row = add_entry(f, "Final Epochs:", "final_epochs", row) + 1
        row = add_check(f, "Do Optional Retrain (Train+Val)", "do_optional_retrain", row) + 1

        # Weight Transfer
        row = add_section(f, "Weight Transfer", row)
        row = add_combo(f, "Transfer Mode:", "transfer_mode", ["smart", "strict"], row) + 1
        row = add_check(f, "Transfer Best Fold Weights", "transfer_best_fold", row) + 1

        # Prediction Intervals
        row = add_section(f, "Prediction Intervals", row)
        row = add_combo(f, "PI Calibration:", "pi_calibration", ["val", "oof"], row) + 1
        row = add_entry(f, "PI Alpha:", "pi_alpha", row) + 1

        # Output
        row = add_section(f, "Output", row)
        row = add_entry(f, "Export Directory:", "export_dir", row, width=25) + 1

        # Plot settings
        row = add_section(f, "Plot Settings", row)
        row = add_entry(f, "Feature X:", "feature_x", row, width=18) + 1
        row = add_entry(f, "Feature Y:", "feature_y", row, width=18) + 1
        row = add_entry(f, "Grid N (single):", "grid_n_single", row) + 1
        row = add_entry(f, "Grid N (multi):", "grid_n_multi", row) + 1
        row = add_entry(f, "Max Pairs:", "max_pairs", row) + 1
        row = add_entry(f, "DPI:", "dpi", row) + 1
        row = add_entry(f, "Fig Width (cm):", "fig_width_cm", row) + 1
        row = add_check(f, "Overlay Train Scatter", "overlay_train_scatter", row) + 1
        row = add_combo(f, "Hold Mode:", "hold_mode", ["median_train", "mean_train", "row"], row) + 1
        row = add_combo(f, "Range Mode:", "range_mode", ["quantile", "minmax"], row) + 1

    # ------------------------------------------------------------------
    # DATA TAB
    # ------------------------------------------------------------------
    def _build_data_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data")

        frame = ttk.LabelFrame(tab, text="Data File", padding=10)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame, text="File:").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Entry(frame, textvariable=self.data_path, width=60).grid(
            row=0, column=1, sticky="ew", padx=5)
        ttk.Button(frame, text="Browse...", command=self._browse_data).grid(
            row=0, column=2, padx=5)
        frame.columnconfigure(1, weight=1)

        # Data preview
        preview_frame = ttk.LabelFrame(tab, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.data_preview = scrolledtext.ScrolledText(preview_frame, height=20,
                                                      font=("Courier", 9), state="disabled")
        self.data_preview.pack(fill=tk.BOTH, expand=True)

        ttk.Button(tab, text="Load & Preview", command=self._load_preview).pack(pady=5)

    # ------------------------------------------------------------------
    # TRAINING TAB
    # ------------------------------------------------------------------
    def _build_training_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Training")

        # Control buttons
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)

        self.run_btn = ttk.Button(ctrl_frame, text="Run Pipeline", command=self._start_training)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = ttk.Button(ctrl_frame, text="Cancel", command=self._cancel_training,
                                     state="disabled")
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(tab, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = ttk.Label(tab, text="Ready")
        self.status_label.pack(padx=10, anchor="w")

        # Log output
        log_frame = ttk.LabelFrame(tab, text="Pipeline Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=25,
                                                   font=("Courier", 9), state="disabled")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # RESULTS TAB
    # ------------------------------------------------------------------
    def _build_results_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Results")

        self.results_text = scrolledtext.ScrolledText(tab, height=30,
                                                      font=("Courier", 10), state="disabled")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(btn_frame, text="Open Export Directory",
                   command=self._open_export_dir).pack(side=tk.LEFT, padx=5)

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def _browse_data(self):
        path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("All supported", "*.tsv *.csv *.txt *.xlsx *.xls"),
                ("TSV files", "*.tsv"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.data_path.set(path)
            self._load_preview()

    def _load_preview(self):
        path = self.data_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Please select a valid data file.")
            return

        try:
            from annsims.pipeline import load_table
            sep = self._get_separator()
            df = load_table(path, sep=sep)
            preview = f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            preview += f"Columns: {list(df.columns)}\n\n"
            preview += "First 10 rows:\n"
            preview += df.head(10).to_string()
            preview += f"\n\nLast 5 rows:\n"
            preview += df.tail(5).to_string()
            preview += f"\n\nDescribe:\n"
            preview += df.describe().to_string()

            self.data_preview.config(state="normal")
            self.data_preview.delete("1.0", tk.END)
            self.data_preview.insert("1.0", preview)
            self.data_preview.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{e}")

    def _get_separator(self):
        sep_str = self.config_vars.get("sep")
        if sep_str:
            val = sep_str.get()
            if val == r"\t":
                return "\t"
            return val
        return "\t"

    def _apply_config(self):
        """Read all config vars from the GUI and update self.config."""
        cfg = self.config
        for attr, var in self.config_vars.items():
            val = var.get()
            current = getattr(cfg, attr)
            try:
                if isinstance(current, bool):
                    setattr(cfg, attr, bool(var.get()) if isinstance(var, tk.BooleanVar) else val.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(cfg, attr, int(val))
                elif isinstance(current, float):
                    setattr(cfg, attr, float(val))
                else:
                    # Handle separator display
                    if attr == "sep" and val == r"\t":
                        setattr(cfg, attr, "\t")
                    else:
                        setattr(cfg, attr, val)
            except (ValueError, TypeError) as e:
                messagebox.showerror("Config Error", f"Invalid value for '{attr}': {val}\n{e}")
                return False
        return True

    def _start_training(self):
        if self.running:
            return

        path = self.data_path.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Please select a data file first.")
            self.notebook.select(1)  # Switch to Data tab
            return

        if not self._apply_config():
            return

        try:
            self.config.validate()
        except ValueError as e:
            messagebox.showerror("Config Error", str(e))
            return

        self.running = True
        self._cancel_flag = False
        self.run_btn.config(state="disabled")
        self.cancel_btn.config(state="normal")
        self.status_label.config(text="Running...")
        self.progress_var.set(0)

        # Clear log
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state="disabled")

        # Switch to training tab
        self.notebook.select(2)

        # Run in background thread
        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _cancel_training(self):
        self._cancel_flag = True
        self.status_label.config(text="Cancelling...")

    def _run_pipeline(self):
        """Run the pipeline in a background thread."""
        from annsims.pipeline import PipelineRunner

        runner = PipelineRunner(
            config=self.config,
            data_path=self.data_path.get(),
            log_fn=self._thread_log,
            progress_fn=self._thread_progress,
            cancelled_fn=lambda: self._cancel_flag,
        )
        self._runner = runner

        try:
            runner.run()
            self.root.after(0, self._on_pipeline_done)
        except Exception as e:
            self.root.after(0, lambda: self._on_pipeline_error(str(e)))

    def _thread_log(self, msg):
        """Thread-safe log append."""
        self.root.after(0, lambda: self._append_log(msg))

    def _thread_progress(self, value):
        """Thread-safe progress update."""
        self.root.after(0, lambda: self.progress_var.set(value))

    def _append_log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def _on_pipeline_done(self):
        self.running = False
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")

        if self._cancel_flag:
            self.status_label.config(text="Cancelled")
            self.progress_var.set(0)
        else:
            self.status_label.config(text="Complete!")
            self.progress_var.set(100)
            self._show_results()
            messagebox.showinfo("Done", "Pipeline completed successfully!\n"
                                "Check the Results tab for metrics.")

    def _on_pipeline_error(self, error_msg):
        self.running = False
        self.run_btn.config(state="normal")
        self.cancel_btn.config(state="disabled")
        self.status_label.config(text="Error!")
        messagebox.showerror("Pipeline Error", f"An error occurred:\n{error_msg}")

    def _show_results(self):
        """Display metrics in the Results tab."""
        if self._runner is None:
            return

        runner = self._runner
        text = "=" * 60 + "\n"
        text += "           ANNSIMS PIPELINE RESULTS\n"
        text += "=" * 60 + "\n\n"

        if runner.cv_summary:
            text += "--- Cross-Validation Summary ---\n"
            cv = runner.cv_summary
            text += f"  R2:   {cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f}\n"
            text += f"  RMSE: {cv['rmse_mean']:.4f} +/- {cv['rmse_std']:.4f}\n"
            text += f"  MAE:  {cv['mae_mean']:.4f} +/- {cv['mae_std']:.4f}\n"
            text += f"  Best Fold R2: {cv['best_fold_r2']:.4f}\n"
            text += f"  Predicted R2 (Q2): {cv['pred_R2']:.4f}\n\n"

        for label, metrics in [("Train", runner.metrics_train),
                               ("Validation", runner.metrics_val),
                               ("Test", runner.metrics_test)]:
            if metrics:
                text += f"--- {label} Metrics ---\n"
                for k, v in metrics.items():
                    if isinstance(v, float):
                        text += f"  {k}: {v:.6f}\n"
                    else:
                        text += f"  {k}: {v}\n"
                text += "\n"

        if runner.plot_paths:
            text += f"--- Generated Plots ---\n"
            text += f"  {len(runner.plot_paths)} 3D surface plots saved\n"
            text += f"  Location: {self.config.export_dir}/plots_3d/\n\n"

        text += f"--- Exported Files ---\n"
        text += f"  Directory: {self.config.export_dir}/\n"
        text += f"  - final_model.keras\n"
        text += f"  - scaler_X.pkl\n"
        text += f"  - scaler_y.pkl\n"
        text += f"  - results_train.csv\n"
        text += f"  - results_val.csv\n"
        text += f"  - results_test.csv\n"
        text += f"  - plots_3d/ (TIFF files)\n"

        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)
        self.results_text.config(state="disabled")

        # Switch to results tab
        self.notebook.select(3)

    def _open_export_dir(self):
        export_dir = self.config.export_dir
        if os.path.exists(export_dir):
            if sys.platform == "win32":
                os.startfile(export_dir)
            elif sys.platform == "darwin":
                os.system(f'open "{export_dir}"')
            else:
                os.system(f'xdg-open "{export_dir}"')
        else:
            messagebox.showinfo("Info", f"Export directory '{export_dir}' does not exist yet.\n"
                                "Run the pipeline first.")

    def run(self):
        """Start the application main loop."""
        self.root.mainloop()
