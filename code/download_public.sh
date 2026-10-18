#!/usr/bin/env bash
# Downloads the public inputs used by public_check.py into data/raw/.
# The CMS download URLs below are the 2024 data-year releases listed in
# https://data.cms.gov/data.json (datasets "Medicare Part D Spending by Drug"
# and "Medicare Part B Spending by Drug"); openFDA Drugs@FDA is a nightly export.
set -euo pipefail
mkdir -p "$(dirname "$0")/../data/raw"; cd "$(dirname "$0")/../data/raw"
curl -L -o partd_2024.csv "https://data.cms.gov/sites/default/files/2026-06/98218f98-166c-4723-8438-c344a4ef96a6/DSD_PTD_RY26_P04_V10_DY24_BGM.csv"
curl -L -o partb_2024.csv "https://data.cms.gov/sites/default/files/2026-06/6616fc91-eb70-404c-9f27-1448e47a3a73/DSD_PTB_RY26_P06_V10_DYT24_HCPCS-%20260602.csv"
curl -L -o drugsfda.json.zip "https://download.open.fda.gov/drug/drugsfda/drug-drugsfda-0001-of-0001.json.zip"
ls -la
