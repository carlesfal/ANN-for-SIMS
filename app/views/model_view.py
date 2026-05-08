from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.views.plot_view import PlotView


class ModelView(QWidget):
    train_requested = pyqtSignal()
    save_model_requested = pyqtSignal()
    load_model_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.hidden_layers_edit = QLineEdit("64,32")
        self.activation_box = QComboBox()
        self.activation_box.addItems(["relu", "sigmoid", "tanh"])

        self.optimizer_box = QComboBox()
        self.optimizer_box.addItems(["adam", "sgd"])

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(1e-6, 1.0)
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setValue(0.001)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(32)

        self.train_button = QPushButton("Train Model")
        self.save_button = QPushButton("Save Model")
        self.load_button = QPushButton("Load Model")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.history_plot = PlotView()

        form = QFormLayout()
        form.addRow("Hidden layers", self.hidden_layers_edit)
        form.addRow("Activation", self.activation_box)
        form.addRow("Optimizer", self.optimizer_box)
        form.addRow("Learning rate", self.learning_rate_spin)
        form.addRow("Epochs", self.epochs_spin)
        form.addRow("Batch size", self.batch_size_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self.train_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.load_button)

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Training Progress"))
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Training Loss"))
        layout.addWidget(self.history_plot)
        layout.addWidget(QLabel("Model Log"))
        layout.addWidget(self.log)
        self.setLayout(layout)

        self.train_button.clicked.connect(self.train_requested.emit)
        self.save_button.clicked.connect(self.save_model_requested.emit)
        self.load_button.clicked.connect(self.load_model_requested.emit)

    def append_log(self, message: str) -> None:
        self.log.append(message)
