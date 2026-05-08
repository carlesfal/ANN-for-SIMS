from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.views.plot_view import PlotView


class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame) -> None:
        super().__init__()
        self._df = dataframe

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        return str(self._df.iat[index.row(), index.column()])

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class DataView(QWidget):
    load_requested = pyqtSignal()
    preprocess_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.load_button = QPushButton("Load Data")
        self.preprocess_button = QPushButton("Apply Preprocessing")
        self.feature_column_box = QComboBox()
        self.target_column_box = QComboBox()
        self.norm_box = QComboBox()
        self.norm_box.addItems(["none", "min-max", "z-score"])

        self.train_split = QSpinBox()
        self.val_split = QSpinBox()
        self.test_split = QSpinBox()
        for box, value in ((self.train_split, 70), (self.val_split, 15), (self.test_split, 15)):
            box.setRange(0, 100)
            box.setValue(value)
            box.setSuffix(" %")

        self.table = QTableView()
        self.plot = PlotView()
        self.summary_box = QTextEdit()
        self.summary_box.setReadOnly(True)

        top_row = QHBoxLayout()
        top_row.addWidget(self.load_button)
        top_row.addWidget(self.preprocess_button)

        mapping_group = QGroupBox("Column Mapping & Preprocessing")
        mapping_form = QFormLayout()
        mapping_form.addRow("Feature column", self.feature_column_box)
        mapping_form.addRow("Target column", self.target_column_box)
        mapping_form.addRow("Normalization", self.norm_box)
        mapping_form.addRow("Train split", self.train_split)
        mapping_form.addRow("Validation split", self.val_split)
        mapping_form.addRow("Test split", self.test_split)
        mapping_group.setLayout(mapping_form)

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(mapping_group)
        layout.addWidget(QLabel("Data Preview"))
        layout.addWidget(self.table)
        layout.addWidget(QLabel("Data Plot"))
        layout.addWidget(self.plot)
        layout.addWidget(QLabel("Summary Statistics"))
        layout.addWidget(self.summary_box)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.load_requested.emit)
        self.preprocess_button.clicked.connect(self.preprocess_requested.emit)

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.table.setModel(PandasModel(dataframe))
        columns = [str(c) for c in dataframe.columns]
        self.feature_column_box.clear()
        self.target_column_box.clear()
        self.feature_column_box.addItems(columns)
        self.target_column_box.addItems(columns)
        if len(columns) > 1:
            self.target_column_box.setCurrentIndex(1)
        self.plot_first_two_columns(dataframe)

    def plot_first_two_columns(self, dataframe: pd.DataFrame) -> None:
        if dataframe.shape[1] < 2:
            self.plot.clear_plot("Need at least 2 columns for preview")
            return
        x = dataframe.iloc[:, 0].to_numpy()
        y = dataframe.iloc[:, 1].to_numpy()
        self.plot.plot_series(x, y, "Raw Data Preview", str(dataframe.columns[0]), str(dataframe.columns[1]))

    def show_summary(self, summary: pd.DataFrame) -> None:
        self.summary_box.setText(summary.to_string())
