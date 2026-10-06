"""vox: local transcription and speech for Apple Silicon Macs and Linux."""

import os

__version__ = "0.2.1"

# vox never sends telemetry. Model downloads are the only network traffic.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
# Plain HTTP downloads write steadily to disk, which gives a smooth progress
# bar and simple resume. Xet downloads stall the bar while chunks are staged.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
