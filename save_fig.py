#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import matplotlib.pyplot as plt

# Function


def save_fig(name: str) -> None:
    """
    Save figure.

    Input:
        name: str
            Figure name.
    """
    plt.savefig(
        fname=f"figures/{name}.pdf",
        format="pdf",
        transparent=True,
        bbox_inches="tight",
    )
