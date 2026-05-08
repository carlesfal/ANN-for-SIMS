from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QStatusBar,
    QTabWidget,
)

from app.controllers.data_controller import DataController
from app.controllers.model_controller import ModelController
from app.controllers.predict_controller import PredictController
from app.models.inference import load_ann_model
from app.models.trainer import TrainingConfig
from app.utils.config import AppConfig, load_config, save_config
from app.utils.file_io import load_tabular_data, save_dataframe_csv
from app.views.data_view import DataView
from app.views.model_view import ModelView
from app.views.predict_view import PredictView


class TrainingWorker(QObject):
    finished = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, model_controller: ModelController, splits, config: TrainingConfig) -> None:
        super().__init__()
        self.model_controller = model_controller
        self.splits = splits
        self.config = config

    def run(self) -> None:
        try:
            model, history = self.model_controller.train_from_splits(self.splits, self.config)
            self.finished.emit(model, history)
        except Exception as exc:  # pragma: no cover - UI error path
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ANNSIMS Desktop")
        self.resize(1280, 900)

        self.config: AppConfig = load_config()

        self.data_controller = DataController()
        self.model_controller = ModelController()
        self.predict_controller = PredictController()

        self.tabs = QTabWidget()
        self.data_view = DataView()
        self.model_view = ModelView()
        self.predict_view = PredictView()

        self.tabs.addTab(self.data_view, "Data")
        self.tabs.addTab(self.model_view, "Model")
        self.tabs.addTab(self.predict_view, "Predict")
        self.setCentralWidget(self.tabs)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        dock = self.addDockWidget  # for readability while keeping imports minimal
        from PyQt6.QtWidgets import QDockWidget

        log_dock = QDockWidget("Application Log", self)
        log_dock.setWidget(self.log_panel)
        dock_area = 8  # Qt.RightDockWidgetArea
        dock(dock_area, log_dock)

        status = QStatusBar()
        self.setStatusBar(status)

        self._make_menu()
        self._wire_events()

    def _make_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_action = file_menu.addAction("Open Data")
        export_action = file_menu.addAction("Export Results")
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")

        model_menu = menu.addMenu("Model")
        train_action = model_menu.addAction("Train")
        load_action = model_menu.addAction("Load Model")
        save_action = model_menu.addAction("Save Model")

        help_menu = menu.addMenu("Help")
        about_action = help_menu.addAction("About")

        open_action.triggered.connect(self.on_load_data)
        export_action.triggered.connect(self.on_export_results)
        exit_action.triggered.connect(self.close)
        train_action.triggered.connect(self.on_train_model)
        load_action.triggered.connect(self.on_load_model)
        save_action.triggered.connect(self.on_save_model)
        about_action.triggered.connect(self.on_about)

    def _wire_events(self) -> None:
        self.data_view.load_requested.connect(self.on_load_data)
        self.data_view.preprocess_requested.connect(self.on_preprocess)
        self.model_view.train_requested.connect(self.on_train_model)
        self.model_view.save_model_requested.connect(self.on_save_model)
        self.model_view.load_model_requested.connect(self.on_load_model)
        self.predict_view.load_input_requested.connect(self.on_load_prediction_input)
        self.predict_view.predict_requested.connect(self.on_predict)
        self.predict_view.export_requested.connect(self.on_export_results)

    def log(self, message: str) -> None:
        self.log_panel.appendPlainText(message)
        self.model_view.append_log(message)
        self.statusBar().showMessage(message, 3000)

    def _open_data_dialog(self, title: str) -> str:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Data Files (*.csv *.txt *.tsv *.xlsx *.xls)",
        )
        return path

    def on_load_data(self) -> None:
        path = self._open_data_dialog("Load Dataset")
        if not path:
            return
        try:
            df = load_tabular_data(path)
            self.data_controller.set_data(df)
            self.data_view.set_dataframe(df)
            self.config.recent_files = [path] + [p for p in self.config.recent_files if p != path][:9]
            save_config(self.config)
            self.log(f"Loaded dataset: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def on_preprocess(self) -> None:
        try:
            feature_column = self.data_view.feature_column_box.currentText()
            target_column = self.data_view.target_column_box.currentText()
            processed, splits, summary = self.data_controller.preprocess(
                feature_columns=[feature_column],
                target_column=target_column,
                method=self.data_view.norm_box.currentText(),
                train_ratio=self.data_view.train_split.value() / 100.0,
                val_ratio=self.data_view.val_split.value() / 100.0,
                test_ratio=self.data_view.test_split.value() / 100.0,
            )
            self.data_view.set_dataframe(processed)
            self.data_view.show_summary(summary)
            self.log(
                "Preprocessing completed. "
                f"Train={len(splits['x_train'])}, Val={len(splits['x_val'])}, Test={len(splits['x_test'])}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Preprocessing Error", str(exc))

    def on_train_model(self) -> None:
        if not self.data_controller.splits:
            QMessageBox.warning(self, "Missing Data", "Run preprocessing before training.")
            return

        try:
            config = TrainingConfig(
                hidden_layers=self.model_controller.parse_hidden_layers(self.model_view.hidden_layers_edit.text()),
                activation=self.model_view.activation_box.currentText(),
                optimizer=self.model_view.optimizer_box.currentText(),
                learning_rate=self.model_view.learning_rate_spin.value(),
                epochs=self.model_view.epochs_spin.value(),
                batch_size=self.model_view.batch_size_spin.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Training Config Error", str(exc))
            return

        self.model_view.progress.setRange(0, 0)
        self.log("Training started...")

        self.thread = QThread(self)
        self.worker = TrainingWorker(self.model_controller, self.data_controller.splits, config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_training_finished)
        self.worker.failed.connect(self._on_training_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)

        self.thread.start()

    def _on_training_finished(self, _model, history) -> None:
        self.model_view.progress.setRange(0, 100)
        self.model_view.progress.setValue(100)
        self.model_view.history_plot.plot_training_history(history["train_loss"], history["val_loss"])
        self.log("Training completed.")

    def _on_training_failed(self, error: str) -> None:
        self.model_view.progress.setRange(0, 100)
        self.model_view.progress.setValue(0)
        QMessageBox.critical(self, "Training Error", error)
        self.log(f"Training failed: {error}")

    def on_save_model(self) -> None:
        if self.model_controller.trained_model is None:
            QMessageBox.warning(self, "No Model", "No trained model available.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.pt", "PyTorch Model (*.pt)")
        if not path:
            return

        try:
            val_losses = (self.model_controller.training_history or {}).get("val_loss", [0.0])
            self.model_controller.save_model(path, min(val_losses))
            self.config.last_model_path = path
            save_config(self.config)
            self.log(f"Model saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def on_load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", self.config.last_model_path or "", "PyTorch Model (*.pt)")
        if not path:
            return
        try:
            model = load_ann_model(path)
            self.model_controller.trained_model = model
            self.config.last_model_path = path
            save_config(self.config)
            self.log(f"Loaded model from {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    def on_load_prediction_input(self) -> None:
        path = self._open_data_dialog("Load Prediction Input")
        if not path:
            return
        try:
            df = load_tabular_data(path)
            self.predict_controller.set_input_data(df)
            self.predict_view.set_input_dataframe(df)
            self.log(f"Loaded prediction input: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Input Error", str(exc))

    def on_predict(self) -> None:
        if self.model_controller.trained_model is None:
            QMessageBox.warning(self, "No Model", "Load or train a model first.")
            return
        if self.predict_controller.input_data is None:
            QMessageBox.warning(self, "No Input", "Load prediction input data first.")
            return

        feature_column = self.data_view.feature_column_box.currentText()
        target_column = self.data_view.target_column_box.currentText() or None

        try:
            results, metrics = self.predict_controller.run_prediction(
                self.model_controller.trained_model,
                feature_columns=[feature_column],
                target_column=target_column,
            )
            self.predict_view.set_result_dataframe(results)
            metric_text = ", ".join(f"{k.upper()}: {v:.6f}" for k, v in metrics.items()) or "N/A"
            self.predict_view.metrics_label.setText(f"Metrics: {metric_text}")
            if metrics and target_column and target_column in results.columns:
                self.predict_view.result_plot.plot_predictions(results[target_column], results["prediction"])
            self.log("Prediction complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", str(exc))

    def on_export_results(self) -> None:
        if self.predict_controller.results is None:
            QMessageBox.warning(self, "No Results", "No predictions available to export.")
            return

        default_path = str(Path.cwd() / "prediction_results.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export Results", default_path, "CSV Files (*.csv)")
        if not path:
            return
        try:
            save_dataframe_csv(self.predict_controller.results, path)
            self.log(f"Exported results to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_about(self) -> None:
        QMessageBox.information(
            self,
            "About ANNSIMS",
            "ANNSIMS Desktop\n"
            "Python desktop app for SIMS ANN training, inference, and visualization.",
        )
