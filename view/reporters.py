from __future__ import annotations
import csv
import os
from typing import Dict

def write_csv_row(path: str, data: Dict[str, float], header_if_new: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if header_if_new and not file_exists:
            writer.writeheader()
        writer.writerow(data)
