#!/usr/bin/env python3
"""
clean_and_rename_channels.py

Resiliently read CSV/XLSX files, optionally drop corrupted “garbage” rows,
rename channel headers ch1..ch16 to your mapping, and save results to a
date-named folder (YYYY-MM-DD) if desired.

Examples (Windows paths ok):
  python clean_and_rename_channels.py "D:\path\file.csv" --drop-garbage --inplace --verbose
  python clean_and_rename_channels.py "D:\path\*.csv" --drop-garbage --date-dir --suffix _renamed
  python clean_and_rename_channels.py data_dir --recursive --map ch1=Fz ch4=Cz

Options:
  --map ch1=Fz ch4=Cz          Inline mapping overrides
  --mapping-file mapping.json  Load mapping from JSON (keys "ch1".."ch16")
  --drop-garbage               Drop rows with non-printables/� or bad timestamp
  --timestamp-col timestamp    Column expected numeric (default: timestamp)
  --replace-first-row          If channels live in the 1st data row, replace there
  --date-dir                   Save outputs inside YYYY-MM-DD subfolder (Europe/London)
  --inplace                    Overwrite input files; otherwise write with suffix
  --suffix _renamed            Suffix when not using --inplace (default: _renamed)
  --recursive                  Recurse into directories
  -v/--verbose                 Verbose output

Requires: pandas (and openpyxl for .xlsx writing/reading)
"""

from __future__ import annotations

import argparse
import io
import json
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
            # allow globs like *.csv
            if any(ch in str(path) for ch in "*?[]"):
                out += [f for f in Path().glob(str(path)) if f.suffix.lower() in exts]
            elif path.suffix.lower() in exts and path.exists():
                out.append(path)
    # de-dup preserve order
    seen = set()
    deduped = []
    for f in out:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def read_csv_resilient(path: Path, verbose: bool=False) -> pd.DataFrame:
    """Try multiple encodings; last resort wrap with TextIOWrapper(errors='replace')."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python")
            if verbose:
                print(f"  Read OK with encoding='{enc}'")
            return df
        except UnicodeDecodeError:
            continue
    if verbose:
        print("  Falling back to TextIOWrapper(errors='replace')")
    with open(path, "rb") as fb:
        text = io.TextIOWrapper(fb, encoding="utf-8", errors="replace")
        df = pd.read_csv(text, engine="python")
    return df


_NONPRINT_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")  # not tab/lf/cr/space..~


def drop_garbage_rows(df: pd.DataFrame, timestamp_col: str | None, verbose: bool=False) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    # rows containing the Unicode replacement char U+FFFD or other nonprintables
    has_repl = df.astype(str).apply(lambda c: c.str.contains("\uFFFD", na=False))
    has_nonprint = df.astype(str).apply(lambda c: c.str.contains(_NONPRINT_RE, na=False))
    mask_bad = has_repl.any(axis=1) | has_nonprint.any(axis=1)
    # optionally enforce numeric timestamp
    if timestamp_col and timestamp_col in df.columns:
        ts_ok = pd.to_numeric(df[timestamp_col], errors="coerce").notna()
        mask_bad = mask_bad | (~ts_ok)
    n_bad = int(mask_bad.sum())
    if verbose and n_bad:
        print(f"  Dropping {n_bad} suspicious row(s) (nonprintables/�/bad timestamp)")
    return df.loc[~mask_bad].copy(), n_bad


def ch_keys_in(sequence: Iterable) -> bool:
    keys = {f"ch{i}" for i in range(1, 17)}
    for val in sequence:
        s = str(val).strip().lower()
        if s in keys:
            return True
    return False


def rename_header(df: pd.DataFrame, mapping: Dict[str, str], verbose: bool=False) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    renamer = {}
    applied = []
    for col in list(df.columns):
        key = str(col).strip().lower()
        if key in mapping:
            renamer[col] = mapping[key]
            applied.append((col, mapping[key]))
    if renamer:
        if verbose:
            print(f"  Renaming columns: {applied}")
        df = df.rename(columns=renamer)
    else:
        if verbose:
            print("  No 'chN' columns found in header")
    return df, applied


def replace_first_row(df: pd.DataFrame, mapping: Dict[str, str], verbose: bool=False) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    if df.empty:
        return df, []
    applied = []
    row0 = df.iloc[0].copy()
    for idx, val in row0.items():
        key = str(val).strip().lower()
        if key in mapping:
            new = mapping[key]
            row0[idx] = new
            applied.append((val, new))
            if verbose:
                print(f"  Row0: {val} -> {new} (column '{idx}')")
    if applied:
        df.iloc[0] = row0
    else:
        if verbose:
            print("  No 'chN' labels found in first data row")
    return df, applied


def save_with_date_folder(out_path: Path, use_date_dir: bool) -> Path:
    if not use_date_dir:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    # Europe/London
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        date_str = datetime.now(ZoneInfo("Europe/London")).strftime("%Y-%m-%d")
    except Exception:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
    dated_dir = out_path.parent / date_str
    dated_dir.mkdir(parents=True, exist_ok=True)
    return dated_dir / out_path.name


def process_csv(path: Path, mapping: Dict[str, str], out_path: Path, drop_bad: bool,
                timestamp_col: str | None, replace_row0: bool, verbose: bool) -> None:
    if verbose:
        print(f"- CSV: {path}")
    df = read_csv_resilient(path, verbose=verbose)
    if drop_bad:
        df, _ = drop_garbage_rows(df, timestamp_col, verbose=verbose)
    changed = False
    if ch_keys_in(df.columns):
        df, applied = rename_header(df, mapping, verbose)
        changed = changed or bool(applied)
    elif replace_row0 or (not df.empty and ch_keys_in(df.iloc[0].tolist())):
        df, applied = replace_first_row(df, mapping, verbose)
        changed = changed or bool(applied)
    if not changed and verbose:
        print("  No changes made.")
    out_path = save_with_date_folder(out_path, use_date_dir=args.date_dir)
    df.to_csv(out_path, index=False, encoding="utf-8")
    if verbose:
        print(f"  → Saved: {out_path}")


def process_xlsx(path: Path, mapping: Dict[str, str], out_path: Path, drop_bad: bool,
                 timestamp_col: str | None, replace_row0: bool, verbose: bool) -> None:
    if verbose:
        print(f"- XLSX: {path}")
    xls = pd.ExcelFile(path)
    out_path = save_with_date_folder(out_path, use_date_dir=args.date_dir)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name in xls.sheet_names:
            if verbose:
                print(f"  Sheet: {sheet_name}")
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
            if drop_bad:
                df, _ = drop_garbage_rows(df, timestamp_col, verbose=verbose)
            changed = False
            if ch_keys_in(df.columns):
                df, applied = rename_header(df, mapping, verbose)
                changed = changed or bool(applied)
            else:
                # try first row replacement path if needed
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                if replace_row0 or ch_keys_in(df_raw.iloc[0].tolist()):
                    df_raw, applied = replace_first_row(df_raw, mapping, verbose)
                    df = df_raw
                    changed = changed or bool(applied)
            if not changed and verbose:
                print("  No changes made on this sheet.")
            # Write with header if present
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
    if verbose:
        print(f"  → Saved: {out_path}")


def main(argv: List[str] | None = None) -> int:
    global args  # used in helpers for date-dir
    p = argparse.ArgumentParser(description="Clean problematic CSV/XLSX files and rename channel labels.")
    p.add_argument("paths", nargs="+", help="Input files, globs, or directories.")
    p.add_argument("--mapping-file", help="JSON file with a dict like {'ch1': 'Fz', ...}.")
    p.add_argument("--map", dest="map_items", nargs="*", help="Override pairs like ch1=Fz ch4=Cz")
    p.add_argument("--drop-garbage", action="store_true", help="Drop rows with nonprintables/� or bad timestamp")
    p.add_argument("--timestamp-col", default="timestamp", help="Expected numeric column to validate (default: timestamp)")
    p.add_argument("--replace-first-row", action="store_true", help="Replace labels in the first *data* row if needed.")
    p.add_argument("--date-dir", action="store_true", help="Save outputs into YYYY-MM-DD subfolder (Europe/London)")
    p.add_argument("--inplace", action="store_true", help="Overwrite files in place.")
    p.add_argument("--suffix", default="_renamed", help="Suffix for output files when not using --inplace (default: _renamed)")
    p.add_argument("--recursive", action="store_true", help="Recurse into directories.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = p.parse_args(argv)

    try:
        mapping = load_mapping(args.mapping_file, args.map_items)
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return 2

    files = find_files(args.paths, recursive=args.recursive)
    if not files:
        print("No CSV/XLSX files found.")
        return 1

    for path in files:
        out_path = Path(path) if args.inplace else Path(path).with_name(Path(path).stem + args.suffix + Path(path).suffix)
        try:
            if Path(path).suffix.lower() == ".csv":
                process_csv(Path(path), mapping, out_path, args.drop_garbage, args.timestamp_col, args.replace_first_row, args.verbose)
            elif Path(path).suffix.lower() == ".xlsx":
                process_xlsx(Path(path), mapping, out_path, args.drop_garbage, args.timestamp_col, args.replace_first_row, args.verbose)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            continue

    if args.verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
