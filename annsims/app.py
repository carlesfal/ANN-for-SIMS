"""Modern CustomTkinter GUI application for ANNSIMS."""

import os
import sys
import threading
import tkinter as tk

import customtkinter as ctk

from annsims.config import PipelineConfig


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ANNSIMSApp(ctk.CTk):
    """Main desktop application window with modern UI."""

    def __init__(self):
        super().__init__()

        self.title("ANNSIMS - ANN for SIMS Prediction")
        self.geometry("1050x750")
        self.minsize(900, 650)

        self.config_obj = PipelineConfig()
        self.data_path = ctk.StringVar(value="")
        self.running = False
        self._cancel_flag = False
        self._runner = None

        self._build_ui()

    def _build_ui(self):
        """Build the main UI with tabview."""
        # Header frame
        header = ctk.CTkFrame(self, height=60, corner_radius=0,
                              fg_color=("gray86", "gray17"))
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="ANNSIMS",
                     font=ctk.CTkFont(size=22, weight="bold")).pack(
            side="left", padx=20, pady=10)
        ctk.CTkLabel(header, text="ANN for SIMS Prediction",
                     font=ctk.CTkFont(size=13),
                     text_color=("gray40", "gray60")).pack(
            side="left", padx=5, pady=10)

        # Appearance toggle
        self.appearance_var = ctk.StringVar(value="Dark")
        appearance_menu = ctk.CTkOptionMenu(
            header, values=["Light", "Dark", "System"],
            variable=self.appearance_var,
            command=self._change_appearance, width=100,
            font=ctk.CTkFont(size=11))
        appearance_menu.pack(side="right", padx=20, pady=10)
        ctk.CTkLabel(header, text="Theme:",
                     font=ctk.CTkFont(size=11)).pack(side="right", pady=10)

        # Main tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=(10, 15))

        self.tabview.add("Configuration")
        self.tabview.add("Data")
        self.tabview.add("Training")
        self.tabview.add("Results")

        self._build_config_tab()
        self._build_data_tab()
        self._build_training_tab()
        self._build_results_tab()

    def _change_appearance(self, mode):
        ctk.set_appearance_mode(mode)

    # ------------------------------------------------------------------
    # CONFIG TAB
    # ------------------------------------------------------------------
    def _build_config_tab(self):
        tab = self.tabview.tab("Configuration")

        scroll = ctk.CTkScrollableFrame(tab, corner_radius=8)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.config_vars = {}

        def add_section(parent, title, row):
            label = ctk.CTkLabel(parent, text=title,
                                 font=ctk.CTkFont(size=14, weight="bold"),
                                 text_color=("gray10", "#4fc3f7"))
            label.grid(row=row, column=0, columnspan=4, sticky="w",
                       pady=(18, 6), padx=8)
            sep = ctk.CTkFrame(parent, height=2, corner_radius=1,
                               fg_color=("gray70", "gray35"))
            sep.grid(row=row + 1, column=0, columnspan=4, sticky="ew",
                     padx=8, pady=(0, 8))
            return row + 2

        def add_entry(parent, label, attr, row, col=0, width=120):
            ctk.CTkLabel(parent, text=label,
                         font=ctk.CTkFont(size=12)).grid(
                row=row, column=col, sticky="w", padx=8, pady=4)
            var = ctk.StringVar(value=str(getattr(self.config_obj, attr)))
            entry = ctk.CTkEntry(parent, textvariable=var, width=width,
                                 corner_radius=6)
            entry.grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)
            self.config_vars[attr] = var
            return row + 1

        def add_check(parent, label, attr, row, col=0):
            var = ctk.BooleanVar(value=getattr(self.config_obj, attr))
            cb = ctk.CTkCheckBox(parent, text=label, variable=var,
                                 font=ctk.CTkFont(size=12),
                                 corner_radius=4)
            cb.grid(row=row, column=col, columnspan=2, sticky="w",
                    padx=8, pady=4)
            self.config_vars[attr] = var
            return row + 1

        def add_combo(parent, label, attr, values, row, col=0):
            ctk.CTkLabel(parent, text=label,
                         font=ctk.CTkFont(size=12)).grid(
                row=row, column=col, sticky="w", padx=8, pady=4)
            var = ctk.StringVar(value=str(getattr(self.config_obj, attr)))
            combo = ctk.CTkOptionMenu(parent, variable=var, values=values,
                                      width=140, corner_radius=6)
            combo.grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)
            self.config_vars[attr] = var
            return row + 1

        row = 0

        row = add_section(scroll, "Hill Pre-training", row)
        row = add_entry(scroll, "N Inputs:", "n_inputs", row)
        row = add_entry(scroll, "N Synthetic:", "n_synthetic", row)
        row = add_entry(scroll, "Hill V_max:", "hill_v_max", row)
        row = add_entry(scroll, "Hill K:", "hill_k", row)
        row = add_entry(scroll, "Hill N:", "hill_n", row)
        row = add_entry(scroll, "Hill X_min:", "hill_x_min", row)
        row = add_entry(scroll, "Hill X_max:", "hill_x_max", row)
        row = add_entry(scroll, "Pretrain Epochs:", "pretrain_epochs", row)

        row = add_section(scroll, "Data Split", row)
        row = add_entry(scroll, "Train %:", "train_percent", row)
        row = add_entry(scroll, "Val %:", "val_percent", row)
        row = add_entry(scroll, "Test %:", "test_percent", row)
        row = add_entry(scroll, "N Labels:", "n_labels", row)
        row = add_entry(scroll, "Target Col (-1=auto):", "target_col", row)
        row = add_combo(scroll, "Separator:", "sep", ["\\t", ",", ";", " "], row)

        row = add_section(scroll, "Training", row)
        row = add_check(scroll, "Disable GPU", "disable_gpu", row)
        row = add_entry(scroll, "Tuner Trials:", "tuner_trials", row)
        row = add_entry(scroll, "K Folds:", "k_folds", row)
        row = add_entry(scroll, "Random Seed:", "random_seed", row)
        row = add_entry(scroll, "Tuner Epochs:", "tuner_epochs", row)
        row = add_entry(scroll, "CV Epochs:", "cv_epochs", row)
        row = add_entry(scroll, "Final Epochs:", "final_epochs", row)
        row = add_check(scroll, "Do Optional Retrain (Train+Val)", "do_optional_retrain", row)

        row = add_section(scroll, "Weight Transfer", row)
        row = add_combo(scroll, "Transfer Mode:", "transfer_mode", ["smart", "strict"], row)

        row = add_section(scroll, "Prediction Intervals", row)
        row = add_combo(scroll, "PI Calibration:", "pi_calibration", ["val", "oof"], row)
        row = add_entry(scroll, "PI Alpha:", "pi_alpha", row)

        row = add_section(scroll, "Output", row)
        row = add_entry(scroll, "Export Directory:", "export_dir", row, width=200)

        row = add_section(scroll, "Plot Settings", row)
        row = add_entry(scroll, "Feature X:", "feature_x", row, width=160)
        row = add_entry(scroll, "Feature Y:", "feature_y", row, width=160)
        row = add_entry(scroll, "Grid N (single):", "grid_n_single", row)
        row = add_entry(scroll, "Grid N (multi):", "grid_n_multi", row)
        row = add_entry(scroll, "Max Pairs:", "max_pairs", row)
        row = add_entry(scroll, "DPI:", "dpi", row)
        row = add_check(scroll, "Overlay Train Scatter", "overlay_train_scatter", row)
        row = add_combo(scroll, "Hold Mode:", "hold_mode",
                        ["median_train", "mean_train", "row"], row)
        row = add_combo(scroll, "Range Mode:", "range_mode",
                        ["quantile", "minmax"], row)

    # ------------------------------------------------------------------
    # DATA TAB
    # ------------------------------------------------------------------
    def _build_data_tab(self):
        tab = self.tabview.tab("Data")

        file_frame = ctk.CTkFrame(tab, corner_radius=10)
        file_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(file_frame, text="Data File",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=15, pady=(12, 5))

        row_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        row_frame.pack(fill="x", padx=15, pady=(0, 12))

        ctk.CTkEntry(row_frame, textvariable=self.data_path,
                     placeholder_text="Select a data file...",
                     corner_radius=6).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(row_frame, text="Browse",
                      command=self._browse_data, width=100,
                      corner_radius=6).pack(side="left", padx=(0, 5))
        ctk.CTkButton(row_frame, text="Load & Preview",
                      command=self._load_preview, width=120,
                      corner_radius=6,
                      fg_color=("green", "#2e7d32"),
                      hover_color=("darkgreen", "#1b5e20")).pack(side="left")

        preview_frame = ctk.CTkFrame(tab, corner_radius=10)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ctk.CTkLabel(preview_frame, text="Preview",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=15, pady=(12, 5))

        self.data_preview = ctk.CTkTextbox(preview_frame, corner_radius=6,
                                           font=ctk.CTkFont(family="Courier", size=11),
                                           state="disabled")
        self.data_preview.pack(fill="both", expand=True, padx=15, pady=(0, 12))

    # ------------------------------------------------------------------
    # TRAINING TAB
    # ------------------------------------------------------------------
    def _build_training_tab(self):
        tab = self.tabview.tab("Training")

        ctrl_frame = ctk.CTkFrame(tab, corner_radius=10)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        btn_row = ctk.CTkFrame(ctrl_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=15, pady=12)

        self.run_btn = ctk.CTkButton(
            btn_row, text="Run Pipeline", command=self._start_training,
            width=150, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=("#2196f3", "#1976d2"),
            hover_color=("#1976d2", "#0d47a1"))
        self.run_btn.pack(side="left", padx=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            btn_row, text="Cancel", command=self._cancel_training,
            width=100, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13),
            fg_color=("gray50", "gray40"),
            hover_color=("gray40", "gray30"),
            state="disabled")
        self.cancel_btn.pack(side="left")

        self.status_label = ctk.CTkLabel(
            btn_row, text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=("gray40", "gray60"))
        self.status_label.pack(side="right", padx=10)

        self.progress_bar = ctk.CTkProgressBar(ctrl_frame,
                                               corner_radius=4, height=8)
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 12))
        self.progress_bar.set(0)

        log_frame = ctk.CTkFrame(tab, corner_radius=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        ctk.CTkLabel(log_frame, text="Pipeline Log",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=15, pady=(12, 5))

        self.log_text = ctk.CTkTextbox(log_frame, corner_radius=6,
                                       font=ctk.CTkFont(family="Courier", size=11),
                                       state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))

    # ------------------------------------------------------------------
    # RESULTS TAB
    # ------------------------------------------------------------------
    def _build_results_tab(self):
        tab = self.tabview.tab("Results")

        results_frame = ctk.CTkFrame(tab, corner_radius=10)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        header_row = ctk.CTkFrame(results_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(12, 5))

        ctk.CTkLabel(header_row, text="Pipeline Results",
                     font=ctk.CTkFont(size=14, weight="bold")).pack(
            side="left")

        ctk.CTkButton(header_row, text="Open Export Folder",
                      command=self._open_export_dir, width=140,
                      corner_radius=6).pack(side="right")

        self.results_text = ctk.CTkTextbox(results_frame, corner_radius=6,
                                           font=ctk.CTkFont(family="Courier", size=11),
                                           state="disabled")
        self.results_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))

    # ------------------------------------------------------------------
    # ACTIONS
    # ------------------------------------------------------------------
    def _browse_data(self):
        path = ctk.filedialog.askopenfilename(
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
            self._show_warning("Please select a valid data file.")
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

            self.data_preview.configure(state="normal")
            self.data_preview.delete("1.0", "end")
            self.data_preview.insert("1.0", preview)
            self.data_preview.configure(state="disabled")
        except Exception as e:
            self._show_error(f"Failed to load data:\n{e}")

    def _get_separator(self):
        sep_str = self.config_vars.get("sep")
        if sep_str:
            val = sep_str.get()
            if val == "\\t":
                return "\t"
            return val
        return "\t"

    def _apply_config(self):
        """Read all config vars from the GUI and update self.config_obj."""
        cfg = self.config_obj
        for attr, var in self.config_vars.items():
            val = var.get()
            current = getattr(cfg, attr)
            try:
                if isinstance(current, bool):
                    if isinstance(var, ctk.BooleanVar):
                        setattr(cfg, attr, bool(var.get()))
                    else:
                        setattr(cfg, attr, val.lower() in ("true", "1", "yes"))
                elif isinstance(current, int):
                    setattr(cfg, attr, int(val))
                elif isinstance(current, float):
                    setattr(cfg, attr, float(val))
                else:
                    if attr == "sep" and val == "\\t":
                        setattr(cfg, attr, "\t")
                    else:
                        setattr(cfg, attr, val)
            except (ValueError, TypeError) as e:
                self._show_error(f"Invalid value for '{attr}': {val}\n{e}")
                return False
        return True

    def _start_training(self):
        if self.running:
            return

        path = self.data_path.get()
        if not path or not os.path.exists(path):
            self._show_warning("Please select a data file first.")
            self.tabview.set("Data")
            return

        if not self._apply_config():
            return

        try:
            self.config_obj.validate()
        except ValueError as e:
            self._show_error(f"Configuration Error:\n{e}")
            return

        self.running = True
        self._cancel_flag = False
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_label.configure(text="Running...",
                                    text_color=("#1976d2", "#64b5f6"))
        self.progress_bar.set(0)

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

        self.tabview.set("Training")

        thread = threading.Thread(target=self._run_pipeline, daemon=True)
        thread.start()

    def _cancel_training(self):
        self._cancel_flag = True
        self.status_label.configure(text="Cancelling...",
                                    text_color=("orange", "#ff9800"))

    def _run_pipeline(self):
        """Run the pipeline in a background thread."""
        from annsims.pipeline import PipelineRunner

        runner = PipelineRunner(
            config=self.config_obj,
            data_path=self.data_path.get(),
            log_fn=self._thread_log,
            progress_fn=self._thread_progress,
            cancelled_fn=lambda: self._cancel_flag,
        )
        self._runner = runner

        try:
            runner.run()
            self.after(0, self._on_pipeline_done)
        except Exception as e:
            self.after(0, lambda: self._on_pipeline_error(str(e)))

    def _thread_log(self, msg):
        """Thread-safe log append."""
        self.after(0, lambda: self._append_log(msg))

    def _thread_progress(self, value):
        """Thread-safe progress update."""
        self.after(0, lambda: self.progress_bar.set(value / 100.0))

    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_pipeline_done(self):
        self.running = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

        if self._cancel_flag:
            self.status_label.configure(text="Cancelled",
                                        text_color=("orange", "#ff9800"))
            self.progress_bar.set(0)
        else:
            self.status_label.configure(text="Complete!",
                                        text_color=("green", "#66bb6a"))
            self.progress_bar.set(1.0)
            self._show_results()
            self._show_info("Pipeline completed successfully!\n"
                           "Check the Results tab for metrics.")

    def _on_pipeline_error(self, error_msg):
        self.running = False
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_label.configure(text="Error!",
                                    text_color=("red", "#ef5350"))
        self._show_error(f"An error occurred:\n{error_msg}")

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
            text += "--- Generated Plots ---\n"
            text += f"  {len(runner.plot_paths)} 3D surface plots saved\n"
            text += f"  Location: {self.config_obj.export_dir}/plots_3d/\n\n"

        text += "--- Exported Files ---\n"
        text += f"  Directory: {self.config_obj.export_dir}/\n"
        text += "  - final_model.keras / .h5\n"
        text += "  - scaler_X.pkl / scaler_y.pkl\n"
        text += "  - model_statistics.txt\n"
        text += "  - model_scheme.txt\n"
        text += "  - mse_evolution.png\n"
        text += "  - train/val/test_predictions.xlsx + .csv\n"
        text += "  - train/val/test_full_with_all_columns.xlsx + .csv\n"
        text += "  - plots/ (diagnostic plots)\n"
        text += "  - plots_3d/ (3D surface PNG files)\n"

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", text)
        self.results_text.configure(state="disabled")

        self.tabview.set("Results")

    def _open_export_dir(self):
        export_dir = self.config_obj.export_dir
        if os.path.exists(export_dir):
            if sys.platform == "win32":
                os.startfile(export_dir)
            elif sys.platform == "darwin":
                os.system(f'open "{export_dir}"')
            else:
                os.system(f'xdg-open "{export_dir}"')
        else:
            self._show_info(f"Export directory '{export_dir}' does not exist yet.\n"
                           "Run the pipeline first.")

    # ------------------------------------------------------------------
    # DIALOG HELPERS
    # ------------------------------------------------------------------
    def _show_warning(self, message):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Warning")
        dialog.geometry("400x180")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.after(10, lambda: self._center_dialog(dialog))

        frame = ctk.CTkFrame(dialog, corner_radius=0, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="Warning",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=("orange", "#ff9800")).pack(pady=(0, 10))
        ctk.CTkLabel(frame, text=message,
                     font=ctk.CTkFont(size=12), wraplength=340).pack(pady=5)
        ctk.CTkButton(frame, text="OK", command=dialog.destroy,
                      width=80, corner_radius=6).pack(pady=(15, 0))

    def _show_error(self, message):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Error")
        dialog.geometry("500x250")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.after(10, lambda: self._center_dialog(dialog))

        frame = ctk.CTkFrame(dialog, corner_radius=0, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="Error",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=("red", "#ef5350")).pack(pady=(0, 10))

        text_box = ctk.CTkTextbox(frame, height=100, corner_radius=6,
                                  font=ctk.CTkFont(size=11))
        text_box.pack(fill="both", expand=True, pady=5)
        text_box.insert("1.0", message)
        text_box.configure(state="disabled")

        ctk.CTkButton(frame, text="OK", command=dialog.destroy,
                      width=80, corner_radius=6).pack(pady=(10, 0))

    def _show_info(self, message):
        dialog = ctk.CTkToplevel(self)
        dialog.title("Info")
        dialog.geometry("400x180")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)
        dialog.after(10, lambda: self._center_dialog(dialog))

        frame = ctk.CTkFrame(dialog, corner_radius=0, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(frame, text="Success",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=("green", "#66bb6a")).pack(pady=(0, 10))
        ctk.CTkLabel(frame, text=message,
                     font=ctk.CTkFont(size=12), wraplength=340).pack(pady=5)
        ctk.CTkButton(frame, text="OK", command=dialog.destroy,
                      width=80, corner_radius=6).pack(pady=(15, 0))

    def _center_dialog(self, dialog):
        dialog.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dialog.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def run(self):
        """Start the application main loop."""
        self.mainloop()
