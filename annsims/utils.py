"""Environment detection and display helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_notebook() -> bool:
    """Return *True* when running inside an IPython/Jupyter kernel."""
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def detect_colab() -> bool:
    """Return *True* when running inside Google Colab."""
    try:
        from google.colab import files as _  # type: ignore[import-untyped] # noqa: F401
        return True
    except ImportError:
        return False


def show(obj: Any, n: int = 5) -> None:
    """Display *obj* using the best available method."""
    if detect_notebook():
        try:
            display(obj)  # type: ignore[name-defined]
        except NameError:
            print(obj)
    else:
        if isinstance(obj, pd.DataFrame):
            print(obj.head(n))
        else:
            print(obj)
