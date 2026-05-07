#!/usr/bin/env python3
"""Entry point for the ANNSIMS Desktop Application."""

import sys
import os

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from annsims.app import ANNSIMSApp


def main():
    app = ANNSIMSApp()
    app.run()


if __name__ == "__main__":
    main()
