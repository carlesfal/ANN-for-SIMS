from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.views.data_view import PandasModel
from app.views.plot_view import PlotView


class PredictView(QWidget):
    load_input_requested = pyqtSignal()
    predict_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.load_input_button = QPushButton("Load Prediction Input")
        self.predict_button = QPushButton("Run Prediction")
        self.export_button = QPushButton("Export Results")
        self.metrics_label = QLabel("Metrics: N/A")

        self.input_table = QTableView()
        self.results_table = QTableView()
        self.result_plot = PlotView()

        buttons = QHBoxLayout()
        buttons.addWidget(self.load_input_button)
        buttons.addWidget(self.predict_button)
        buttons.addWidget(self.export_button)

        layout = QVBoxLayout()
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Prediction Input"))
        layout.addWidget(self.input_table)
        layout.addWidget(QLabel("Prediction Results"))
        layout.addWidget(self.results_table)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.result_plot)
        self.setLayout(layout)

        self.load_input_button.clicked.connect(self.load_input_requested.emit)
        self.predict_button.clicked.connect(self.predict_requested.emit)
        self.export_button.clicked.connect(self.export_requested.emit)

    def set_input_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.input_table.setModel(PandasModel(dataframe))

    def set_result_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.results_table.setModel(PandasModel(dataframe))
