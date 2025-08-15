#!/usr/bin/env python3
"""
rename_channels.py

Replace channel labels "ch1"–"ch16" in the *header row* (or optionally, the first data row)
of CSV / Excel files using a mapping dictionary that you can customize.

Default mapping (10-20 system):
  ch1: Fp1, ch2: F3, ch3: F7, ch4: C3, ch5: T3, ch6: P3, ch7: T5, ch8: O1,
  ch9: Fp2, ch10: F4, ch11: F8, ch12: C4, ch13: T4, ch14: P4, ch15: T6, ch16: O2

USAGE EXAMPLES
--------------
1) Single file (CSV or XLSX), rename header labels in-place:
   python rename_channels.py data/eeg.csv --inplace

2) Save to a new file with suffix:
   python rename_channels.py data/eeg.xlsx --suffix _renamed

3) Override some labels from CLI:
   python rename_channels.py data/eeg.csv --map ch1=Fz ch4=Cz

4) Use a JSON mapping file (keys like "ch1": "Fz"):
   python rename_channels.py data/*.csv --mapping-file mapping.json --inplace

5) Directory, recurse into subfolders:
   python rename_channels.py data_dir --recursive --suffix _hdr

NOTES
-----
- By default we rename the *column headers*. If your file has "ch1..ch16"
  in the *first data row* instead of the header, pass: --replace-first-row
- Excel writing requires `openpyxl` to be installed.
"""

#python rename_channels.py /path/to/20250815_144519_eeg_with_markers.csv --suffix _renamed
#D:/user/Files_without_backup/MSc_Project/20250815_144519_eeg_with_markers.csv --suffix _renamed

from __future__ import annotations

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_MAPPING: Dict[str, str] = {
    "ch1": "Fp1", "ch2": "F3", "ch3": "F7", "ch4": "C3",
    "ch5": "T3", "ch6": "P3", "ch7": "T5", "ch8": "O1",
    "ch9": "Fp2", "ch10": "F4", "ch11": "F8", "ch12": "C4",
    "ch13": "T4", "ch14": "P4", "ch15": "T6", "ch16": "O2",
}


def parse_kv_overrides(pairs: Iterable[str]) -> Dict[str, str]:
    """Parse CLI items like 'ch1=Fz' into {'ch1': 'Fz'} (case-insensitive keys)."""
    out = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value form")
        k, v = item.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if not re.fullmatch(r"ch([1-9]|1[0-6])", k):
            raise ValueError(f"Key '{k}' must be 'ch1'..'ch16'")
        if not v:
            raise ValueError(f"Value for '{k}' is empty")
        out[k] = v
    return out


def load_mapping(mapping_file: str | None, cli_pairs: Iterable[str] | None) -> Dict[str, str]:
    mapping = DEFAULT_MAPPING.copy()
    if mapping_file:
        with open(mapping_file, "r", encoding="utf-8") as f:
            file_map = json.load(f)
        # normalize keys to lower-case 'chN'
        file_map_norm = {str(k).strip().lower(): str(v).strip() for k, v in file_map.items()}
        mapping.update(file_map_norm)
    if cli_pairs:
        mapping.update(parse_kv_overrides(cli_pairs))
    return mapping


def find_files(paths: Iterable[str], recursive: bool) -> List[Path]:
    exts = {".csv", ".xlsx"}
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            if recursive:
                out += [f for f in path.rglob("*") if f.suffix.lower() in exts]
            else:
                out += [f for f in path.glob("*") if f.suffix.lower() in exts]
        else:
            # Allow globs
            if any(ch in str(path) for ch in "*?[]"):
                out += [f for f in Path().glob(str(path)) if f.suffix.lower() in exts]
            elif path.suffix.lower() in exts and path.exists():
                out.append(path)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for f in out:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def ch_keys_in(sequence: Iterable) -> bool:
    """Return True if any 'ch1'..'ch16' appears in a sequence of labels (case-insensitive)."""
    keys = {f"ch{i}" for i in range(1, 17)}
    for val in sequence:
        s = str(val).strip().lower()
        if s in keys:
            return True
    return False


def rename_header(df: pd.DataFrame, mapping: Dict[str, str], verbose: bool=False) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Rename DataFrame headers using mapping keys 'chN' (case-insensitive)."""
    # Build a case-insensitive renamer dict for existing columns
    current = list(df.columns)
    renamer = {}
    applied = []
    for col in current:
        key = str(col).strip().lower()
        if key in mapping:
            new = mapping[key]
            renamer[col] = new
            applied.append((col, new))
    if renamer:
        if verbose:
            print(f"  Renaming columns: {applied}")
        df = df.rename(columns=renamer)
    else:
        if verbose:
            print("  No 'chN' columns found in header")
    return df, applied


def replace_first_row(df: pd.DataFrame, mapping: Dict[str, str], verbose: bool=False) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Replace occurrences of 'chN' in the first row (row index 0) with mapped names."""
    if df.empty:
        return df, []
    applied = []
    row0 = df.iloc[0].copy()
    for idx, val in row0.items():
        key = str(val).strip().lower()
        if key in mapping:
            new = mapping[key]
            if verbose:
                print(f"  Row0: {val} -> {new} (column '{idx}')")
            row0[idx] = new
            applied.append((val, new))
    if applied:
        df.iloc[0] = row0
    else:
        if verbose:
            print("  No 'chN' labels found in the first row")
    return df, applied


def process_csv(path: Path, mapping: Dict[str, str], out_path: Path, replace_row0: bool, verbose: bool=False) -> bool:
    if verbose:
        print(f"- CSV: {path}")
    # Try header mode first
    df = pd.read_csv(path)
    changed = False
    if ch_keys_in(df.columns):
        df2, applied = rename_header(df, mapping, verbose)
        changed = changed or bool(applied)
    elif replace_row0 or ch_keys_in(df.iloc[0].tolist()):
        df2, applied = replace_first_row(df, mapping, verbose)
        changed = changed or bool(applied)
    else:
        df2 = df
        if verbose:
            print("  No changes made.")
    if changed:
        df2.to_csv(out_path, index=False)
        if verbose:
            print(f"  → Saved: {out_path}")
    else:
        # Still save if output path differs (to keep behavior consistent)
        if out_path != path:
            df2.to_csv(out_path, index=False)
            if verbose:
                print(f"  → Copied without changes: {out_path}")
    return changed


def process_xlsx(path: Path, mapping: Dict[str, str], out_path: Path, replace_row0: bool, verbose: bool=False) -> bool:
    if verbose:
        print(f"- XLSX: {path}")
    xls = pd.ExcelFile(path)
    changed_any = False
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            if verbose:
                print(f"  Sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
            changed = False
            if ch_keys_in(df.columns):
                df2, applied = rename_header(df, mapping, verbose)
                changed = changed or bool(applied)
            else:
                # Try without header to inspect first row for 'chN'
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                if replace_row0 or (not ch_keys_in(df.columns) and ch_keys_in(df_raw.iloc[0].tolist())):
                    df_raw2, applied = replace_first_row(df_raw, mapping, verbose)
                    df2 = df_raw2
                    changed = changed or bool(applied)
                else:
                    df2 = df
                    if verbose:
                        print("  No changes made.")
            df2.to_excel(writer, sheet_name=sheet_name, index=False, header=(df2.columns is not None))
            changed_any = changed_any or changed
        writer.close()
    if verbose:
        print(f"  → Saved: {out_path}")
    return changed_any


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Replace 'ch1'..'ch16' with custom labels in CSV/XLSX files.")
    p.add_argument("paths", nargs="+", help="Input files, globs, or directories.")
    p.add_argument("--mapping-file", help="JSON file with a dict like {'ch1': 'Fz', ...}.")
    p.add_argument("--map", dest="map_items", nargs="*", help="Override pairs like ch1=Fz ch4=Cz")
    p.add_argument("--replace-first-row", action="store_true",
                   help="Replace labels in the first *data* row instead of the header, if needed.")
    p.add_argument("--inplace", action="store_true", help="Overwrite files in place.")
    p.add_argument("--suffix", default="_renamed", help="Suffix for output files when not using --inplace (default: _renamed)")
    p.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = p.parse_args(argv)

    try:
        mapping = load_mapping(args.mapping_file, args.map_items)
    except Exception as e:
        print(f"Error loading mapping: {e}", file=sys.stderr)
        return 2

    files = find_files(args.paths, recursive=args.recursive)
    if not files:
        print("No CSV/XLSX files found.", file=sys.stderr)
        return 1

    for path in files:
        out_path = path if args.inplace else path.with_name(path.stem + args.suffix + path.suffix)
        try:
            if path.suffix.lower() == ".csv":
                process_csv(path, mapping, out_path, args.replace_first_row, args.verbose)
            elif path.suffix.lower() == ".xlsx":
                process_xlsx(path, mapping, out_path, args.replace_first_row, args.verbose)
            else:
                if args.verbose:
                    print(f"Skipping unsupported file type: {path}")
        except Exception as e:
            print(f"Failed to process {path}: {e}", file=sys.stderr)
            continue

    if args.verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
