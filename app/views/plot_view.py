from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotView(FigureCanvasQTAgg):
    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear_plot(self, title: str = "") -> None:
        self.axes.clear()
        self.axes.set_title(title)
        self.draw()

    def plot_series(self, x, y, title: str, xlabel: str, ylabel: str) -> None:
        self.axes.clear()
        self.axes.plot(x, y)
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.draw()

    def plot_training_history(self, train_loss, val_loss) -> None:
        self.axes.clear()
        self.axes.plot(train_loss, label="Train Loss")
        self.axes.plot(val_loss, label="Val Loss")
        self.axes.set_title("Training History")
        self.axes.set_xlabel("Epoch")
        self.axes.set_ylabel("Loss")
        self.axes.legend()
        self.draw()

    def plot_predictions(self, y_true, y_pred) -> None:
        self.axes.clear()
        self.axes.scatter(y_true, y_pred, alpha=0.7)
        min_val = min(float(min(y_true)), float(min(y_pred)))
        max_val = max(float(max(y_true)), float(max(y_pred)))
        self.axes.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        self.axes.set_title("Predicted vs Actual")
        self.axes.set_xlabel("Actual")
        self.axes.set_ylabel("Predicted")
        self.draw()
