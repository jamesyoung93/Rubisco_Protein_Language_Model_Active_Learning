#!/usr/bin/env bash
set -euo pipefail

# Example: rebuild from sources (requires providing --lab_docx and optionally --yu2015_pdf)
# python build_cyano_doubling_time_dataset.py --download_yu2015 --lab_docx "../Cyano doubling time-122225.docx" --include_pcc11801

# Default: analyze shipped outputs
python analyze_cyano_doubling_times.py
