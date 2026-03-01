#!/usr/bin/env python3
"""
Finnish/Swedish Name Generator using Markov Chains.

This module now serves as a thin façade that exposes the public API
and command-line entry point, while the implementation is split into
smaller, focused modules:

- ``markov_generator``: core Markov chain name generation logic.
- ``data_loader``: CSV reading and name/prevalence loading utilities.
- ``cli``: command-line interface and application orchestration.
"""

from data_loader import load_names_by_language, load_names_from_csv
from markov_generator import MarkovNameGenerator
from cli import main

__all__ = [
    "MarkovNameGenerator",
    "load_names_from_csv",
    "load_names_by_language",
    "main",
]


if __name__ == "__main__":
    main()

