"""Shared pytest configuration for fast architecture and notebook-interface checks."""

from __future__ import annotations

import os


# Several public modules resolve Location Configuration at import time. Use the
# Reference Study Location so import smoke tests can collect without running notebooks.
os.environ.setdefault("FLOOD_RM_LOCATION", "marshfield")
