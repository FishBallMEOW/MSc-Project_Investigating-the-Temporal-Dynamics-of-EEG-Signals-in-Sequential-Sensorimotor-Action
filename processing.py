#!/usr/bin/env python3
"""
EEG Motor Imagery (MI) analysis pipeline that can run on:
  1) PhysioNet EEG Motor Movement/Imagery (EEGBCI) EDF files via MNE's fetcher
  2) Your own CSV recordings (EEG + markers)

What it does (end-to-end):
- Load data (EDF from EEGBCI OR CSV)
- Preprocess: notch (50 Hz), band-pass (0.5–40 Hz), (optional) resample, average ref
- Epoch around MI cues (default: -1 … 4 s)
- Time–frequency (Morlet) → % change (ERD/ERS)
- Extract per-trial ERD metrics (mu/beta @ C3/C4): peak ERD & onset latency
- Decoding (CSP + LDA and linear SVM) with stratified 5-fold CV on MI window
- Save key plots and CSV summaries under ./outputs

Dependencies (tested with MNE 1.6+):
    pip install mne numpy scipy pandas scikit-learn matplotlib
Optional: mne-connectivity, pyprep

Notes about EEGBCI runs (from MNE docs):
- Runs 4, 8, 12: Motor imagery (left vs right hand)
- Runs 6, 10, 14: Motor imagery (hands vs feet)
You can change these in the CONFIG below.

CSV expectation (configurable):
  * One file that interleaves EEG samples and marker rows, e.g.
      timestamp,type,ch1,...,chN,marker
      372762.7847,EEG, ...
      372762.7864,Marker,,,,, 100
  * "type" distinguishes EEG vs Marker rows
  * timestamps are in seconds (monotonic; small jitter ok)
  * channel columns named like ch1..ch16 or EEG1..EEG16 (auto-detected or set below)
  * marker column holds integer codes; map codes → conditions via CSV_EVENT_CODE_MAP

Author: (your name)
License: MIT
"""

#%% Load Libraries
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
mne.set_config('MNE_DATA', r'D:/data/mne_datasets', set_env=True)
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

#%% FUNCTIONS
# CONFIGURATION
@dataclass
class Config:
    # Source: 'eegbci' or 'csv'
    source: str = "eegbci"  # or "csv"

    # EEGBCI configuration
    eegbci_subject: int = 1
    # default motor imagery left-vs-right runs (per MNE docs)
    eegbci_runs: List[int] = None  # will default in __post_init__

    # CSV configuration
    csv_path: Optional[str] = None
    time_col: str = "timestamp"
    type_col: str = "type"           # values: 'EEG' or 'Marker'
    eeg_value: str = "EEG"
    marker_value: str = "Marker"
    marker_col: str = "marker"        # integer event codes
    channel_cols: Optional[List[str]] = None  # if None, auto-detect by regex (ch\d+|eeg\d+)
    csv_event_code_map: Dict[int, str] = None  # e.g., {100: 'left', 200: 'right'}
    csv_voltage_unit: str = "uV"       # 'uV' or 'V' (conversion to Volts is applied)

    # General preprocessing
    notch_freq: float = 50.0           # UK mains
    l_freq: float = 0.5
    h_freq: float = 40.0
    resample_sfreq: Optional[float] = 250.0   # None to keep native
    reref: str = "average"             # 'average' or a list of channel names

    # Epoching
    tmin: float = -1.0
    tmax: float = 4.0
    baseline: Tuple[Optional[float], Optional[float]] = (-1.0, 0.0)

    # Time-frequency
    freqs: np.ndarray = None           # will default to np.arange(8, 31)
    n_cycles: float = 5.0              # Morlet cycles
    ers_baseline_mode: str = 'percent' # 'percent' gives % change

    # ERD metrics
    mu_band: Tuple[float, float] = (8., 13.)
    beta_band: Tuple[float, float] = (13., 30.)
    erd_onset_threshold: float = -20.0 # % change threshold for onset (negative means drop)
    mi_window: Tuple[float, float] = (1.0, 4.0)  # window for ERD and decoding
    sensorimotor_chs: List[str] = None # defaults to ['C3','C4','Cz']

    # Decoding
    csp_components: int = 6
    cv_folds: int = 5
    random_state: int = 42

    # Output
    out_dir: str = "outputs"

    def __post_init__(self):
        if self.eegbci_runs is None:
            self.eegbci_runs = [4, 8, 12]  # MI left vs right hand
        if self.freqs is None:
            self.freqs = np.arange(8, 31)  # 8–30 Hz
        if self.sensorimotor_chs is None:
            self.sensorimotor_chs = ['C3', 'C4', 'Cz']
        if self.csv_event_code_map is None:
            # sensible default; adjust to your markers
            self.csv_event_code_map = {1: 'left', 2: 'right'}


CFG = Config()
Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)


# UTILITIES
def _to_volts(arr: np.ndarray, unit: str) -> np.ndarray:
    if unit.lower() in ["uv", "µv", "microvolt", "microvolts"]:
        return arr / 1e6
    elif unit.lower() in ["mv", "millivolt", "millivolts"]:
        return arr / 1e3
    elif unit.lower() in ["v", "volt", "volts"]:
        return arr
    else:
        raise ValueError(f"Unknown voltage unit: {unit}")


def _pick_existing(raw: mne.io.BaseRaw, names: List[str]) -> List[str]:
    return [ch for ch in names if ch in raw.ch_names]


# LOADERS
def load_from_eegbci(cfg: Config) -> Tuple[mne.io.BaseRaw, np.ndarray, Dict[str, int]]:
    """Download+load EEGBCI runs, concatenate, standardize names & montage, extract events.

    Returns: raw, events, event_id mapping {'left': int, 'right': int}
    """
    print(f"Fetching EEGBCI subject {cfg.eegbci_subject}, runs {cfg.eegbci_runs}…")
    from mne.datasets import eegbci

    file_paths = eegbci.load_data(cfg.eegbci_subject, cfg.eegbci_runs)
    raws = []
    for f in file_paths:
        r = mne.io.read_raw_edf(f, preload=True, verbose=False)
        # Standardize channel names like 'Fp1' → 'FP1', etc.
        eegbci.standardize(r)
        raws.append(r)
    raw = mne.concatenate_raws(raws)

    # Add a standard montage so C3/C4 etc are recognized spatially
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Extract events: EEGBCI annotates with 'T1' (left) and 'T2' (right)
    event_id_map = {'T1': 1, 'T2': 2}
    events, _ = mne.events_from_annotations(raw, event_id=event_id_map, verbose=False)
    # Make human-readable event_id
    event_id = {'left': 1, 'right': 2}

    return raw, events, event_id


def load_from_csv(cfg: Config) -> Tuple[mne.io.BaseRaw, np.ndarray, Dict[str, int]]:
    """Load custom CSV into an MNE Raw and events array.

    Expects: a column distinguishing EEG vs Marker rows, timestamps in seconds,
             channel columns for EEG rows, and an integer marker code column.
    """
    if cfg.csv_path is None:
        raise ValueError("csv_path must be set in Config when source='csv'")

    print(f"Loading CSV: {cfg.csv_path}")
    df = pd.read_csv(cfg.csv_path, encoding='latin1', sep=',', engine='python')

    # Split EEG vs Marker rows
    if cfg.type_col not in df.columns:
        raise ValueError(f"CSV missing type column '{cfg.type_col}'")
    eeg_df = df[df[cfg.type_col] == cfg.eeg_value].copy()
    mrk_df = df[df[cfg.type_col] == cfg.marker_value].copy()

    if cfg.time_col not in df.columns:
        raise ValueError(f"CSV missing time column '{cfg.time_col}'")
    if cfg.marker_col not in df.columns:
        raise ValueError(f"CSV missing marker column '{cfg.marker_col}'")

    # Auto-detect channel columns if needed
    ch_cols = cfg.channel_cols
    if ch_cols is None:
        ch_cols = [c for c in eeg_df.columns if pd.Series(c).str.contains(r'^(ch|eeg)\d+$', case=False, regex=True).any()]
        if len(ch_cols) == 0:
            # fallback: all numeric columns except timestamp
            numeric_cols = [c for c in eeg_df.columns if np.issubdtype(eeg_df[c].dtype, np.number)]
            ch_cols = [c for c in numeric_cols if c != cfg.time_col]
    ch_cols = list(ch_cols)

    if len(ch_cols) == 0:
        raise ValueError("No channel columns found. Set Config.channel_cols explicitly.")

    # Ensure sorted by timestamp and drop duplicates
    eeg_df = eeg_df.sort_values(cfg.time_col).drop_duplicates(subset=[cfg.time_col])

    # Estimate sampling frequency from median dt
    ts = eeg_df[cfg.time_col].to_numpy()
    dts = np.diff(ts)
    # debug
    print(f"Timestamp differences: {1/dts}")
    if len(dts) == 0:
        raise ValueError("Not enough EEG rows to estimate sampling rate")
    sfreq = float(np.round(1.0 / np.median(dts)))
    print(f"Estimated sampling rate: {sfreq:.1f} Hz")

    # Build MNE RawArray (channels x samples), convert to Volts
    data = eeg_df[ch_cols].to_numpy().T  # shape (n_channels, n_samples)
    data = _to_volts(data, cfg.csv_voltage_unit)

    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    # Add montage if channel names standard 10-10; ignore if not found
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
    except Exception:
        pass

    # Build events array from marker rows
    mrk_df = mrk_df.sort_values(cfg.time_col)
    code_to_name = cfg.csv_event_code_map  # e.g., {100:'left', 200:'right'}
    # Only keep codes we know
    mrk_df = mrk_df[mrk_df[cfg.marker_col].isin(list(code_to_name.keys()))]

    # Convert marker timestamps to sample indices
    mrk_ts = mrk_df[cfg.time_col].to_numpy()
    mrk_codes = mrk_df[cfg.marker_col].astype(int).to_numpy()

    # Build events: [sample, 0, event_id]
    events = []
    name_to_int: Dict[str, int] = {name: i + 1 for i, name in enumerate(sorted(set(code_to_name.values())))}
    # Inverse mapping
    code_to_int = {code: name_to_int[code_to_name[code]] for code in code_to_name}

    start_time = eeg_df[cfg.time_col].iloc[0]
    for t, code in zip(mrk_ts, mrk_codes):
        sample = int(np.round((t - start_time) * sfreq))
        if sample < 0 or sample >= raw.n_times:
            # Skip markers outside the EEG time span
            continue
        events.append([sample, 0, code_to_int[code]])
    events = np.array(events, dtype=int)

    # event_id mapping (names to ints)
    event_id = name_to_int

    return raw, events, event_id


# PREPROCESSING / EPOCHING
def preprocess_raw(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    print("Preprocessing: notch, band-pass, resample, re-reference…")
    raw = raw.copy()
    # Notch for mains interference
    if cfg.notch_freq:
        raw.notch_filter(freqs=[cfg.notch_freq], picks='eeg')
    # Band-pass
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks='eeg')
    # Resample (optional)
    if cfg.resample_sfreq is not None and cfg.resample_sfreq > 0:
        raw.resample(cfg.resample_sfreq)
    # Re-reference
    if cfg.reref == 'average':
        raw.set_eeg_reference('average', projection=False)
    elif isinstance(cfg.reref, (list, tuple)):
        raw.set_eeg_reference(cfg.reref, projection=False)
    return raw


def make_epochs(raw: mne.io.BaseRaw, events: np.ndarray, event_id: Dict[str, int], cfg: Config) -> mne.Epochs:
    print("Epoching around events…")
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=cfg.tmin, tmax=cfg.tmax,
                        baseline=cfg.baseline, picks=picks, preload=True, detrend=1)
    return epochs


# TIME–FREQUENCY & ERD METRICS
def compute_tfr(epochs: mne.Epochs, cfg: Config) -> mne.time_frequency.AverageTFR:
    print("Computing Morlet time–frequency power…")
    power = mne.time_frequency.tfr_morlet(
        epochs, freqs=cfg.freqs, n_cycles=cfg.n_cycles, use_fft=True,
        return_itc=False, average=True, decim=1, n_jobs=None
    )
    # Baseline as % change
    power.apply_baseline(cfg.baseline, mode=cfg.ers_baseline_mode)
    return power


def _band_indices(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return np.where((freqs >= lo) & (freqs <= hi))[0]


def extract_erd_metrics(epochs: mne.Epochs, power: mne.time_frequency.AverageTFR, cfg: Config) -> pd.DataFrame:
    """Compute per-trial ERD metrics in mu/beta over sensorimotor channels.

    Returns a tidy DataFrame with columns:
        ['trial', 'condition', 'channel', 'band', 'peak_erd_%', 'onset_s']
    """
    print("Extracting ERD metrics…")
    # For per-trial metrics we need single-trial TFR; recompute with average=False
    st_power = mne.time_frequency.tfr_morlet(
        epochs, freqs=cfg.freqs, n_cycles=cfg.n_cycles, use_fft=True,
        return_itc=False, average=False, decim=1, n_jobs=None
    )
    st_power.apply_baseline(cfg.baseline, mode=cfg.ers_baseline_mode)

    # Select channels
    chs = _pick_existing(epochs._raw, cfg.sensorimotor_chs)
    if not chs:
        chs = epochs.ch_names  # fallback
    ch_idx = [epochs.ch_names.index(c) for c in chs]

    mu_idx = _band_indices(cfg.freqs, cfg.mu_band)
    beta_idx = _band_indices(cfg.freqs, cfg.beta_band)
    mi_sample_lo = int(np.round((cfg.mi_window[0] - epochs.tmin) * epochs.info['sfreq']))
    mi_sample_hi = int(np.round((cfg.mi_window[1] - epochs.tmin) * epochs.info['sfreq']))

    rows = []
    # st_power.data shape: (n_epochs, n_channels, n_freqs, n_times)
    for ei in range(len(epochs)):
        cond = epochs.events[ei, 2]
        # reverse-map cond int -> name
        cond_name = {v: k for k, v in epochs.event_id.items()}[cond]

        # average across selected channels within each band
        for band_name, band_idx in [("mu", mu_idx), ("beta", beta_idx)]:
            # mean across freqs in band
            band_data = st_power.data[ei, ch_idx][:, band_idx, :]  # (n_ch, n_band_freqs, n_times)
            band_ts = band_data.mean(axis=1).mean(axis=0)  # (n_times,)

            # Peak ERD: minimum % in MI window
            mi_seg = band_ts[mi_sample_lo:mi_sample_hi]
            peak_erd = float(np.min(mi_seg))

            # Onset latency: first time ERD drops below threshold and stays 200 ms
            thr = cfg.erd_onset_threshold
            below = mi_seg < thr
            onset = np.nan
            if below.any():
                # require 0.2 s of sustained drop
                sustain = int(np.round(0.2 * epochs.info['sfreq']))
                idxs = np.where(below)[0]
                for idx in idxs:
                    end = idx + sustain
                    if end <= len(mi_seg) and np.all(mi_seg[idx:end] < thr):
                        onset = epochs.times[mi_sample_lo + idx]
                        break
            rows.append({
                'trial': ei,
                'condition': cond_name,
                'channel': "+".join(chs),
                'band': band_name,
                'peak_erd_%': peak_erd,
                'onset_s': onset
            })
    df = pd.DataFrame(rows)
    return df


# DECODING (CSP + LDA / SVM)
def run_decoding(epochs: mne.Epochs, cfg: Config) -> pd.DataFrame:
    print("Running decoding (CSP + LDA / SVM)…")
    # Focus on MI window (copy epochs with shorter time range)
    ep = epochs.copy().crop(tmin=cfg.mi_window[0], tmax=cfg.mi_window[1])

    # Band-pass for decoding (8–30 Hz is typical)
    ep.load_data().filter(8., 30., picks='eeg')

    X = ep.get_data()  # (n_epochs, n_channels, n_times)
    y = ep.events[:, 2]

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    results = []
    for clf_name, clf in [
        ("LDA", LDA()),
        ("LinearSVM", SVC(kernel='linear', C=1.0, probability=False))
    ]:
        pipe = Pipeline([
            ('csp', CSP(n_components=cfg.csp_components, reg='oas', log=True, norm_trace=False)),
            ('clf', clf)
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=None)
        results.append({'classifier': clf_name, 'mean_accuracy': float(np.mean(scores)), 'std': float(np.std(scores))})
        print(f"  {clf_name}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    return pd.DataFrame(results)


# PLOTTING HELPERS
def plot_tfr(power: mne.time_frequency.AverageTFR, cfg: Config, fname: str):
    """Save TFR plots. mne.AverageTFR.plot returns a Figure *or* a list of Figures
    when multiple channels are picked. Handle both cases and save one PNG per channel
    if needed (e.g., tfr_sensorimotor_C3.png, tfr_sensorimotor_C4.png).
    """
    picks = _pick_existing(power, cfg.sensorimotor_chs) or None
    figs = power.plot(picks=picks, baseline=None, mode=None, show=False)

    # Determine base filename without extension
    base = Path(fname)
    stem = base.with_suffix("")

    if isinstance(figs, list):
        # Save each figure with channel suffix
        if picks is None:
            ch_names = power.ch_names
        else:
            ch_names = picks
        for fig, ch in zip(figs, ch_names):
            out_path = f"{stem}_{ch}.png"
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    else:
        figs.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(figs)


def plot_erd_summary(df: pd.DataFrame, fname: str):
    import seaborn as sns  # optional, for a nicer summary; fallback to matplotlib if missing
    try:
        plt.figure(figsize=(7,4))
        sns.barplot(data=df, x='band', y='peak_erd_%', hue='condition', errorbar='se')
        plt.title('Peak ERD (% change) by condition')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
    except Exception:
        # Basic Matplotlib fallback
        bands = sorted(df['band'].unique())
        conds = sorted(df['condition'].unique())
        means = np.array([[df[(df['band']==b)&(df['condition']==c)]['peak_erd_%'].mean() for c in conds] for b in bands])
        x = np.arange(len(bands))
        width = 0.35
        plt.figure(figsize=(7,4))
        for i, c in enumerate(conds):
            plt.bar(x + i*width, means[:, i], width, label=c)
        plt.xticks(x + width/2, bands)
        plt.ylabel('Peak ERD (%)')
        plt.legend()
        plt.title('Peak ERD by condition')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()


#%% MAIN PIPELINE
    # # 2) Preprocess
    # raw_prep = preprocess_raw(raw, cfg)

    # # 3) Epochs
    # epochs = make_epochs(raw_prep, events, event_id, cfg)

    # # Persist a copy of epochs metadata
    # meta_path = Path(cfg.out_dir) / 'epochs_metadata.json'
    # with open(meta_path, 'w') as f:
    #     json.dump({
    #         'n_epochs': int(len(epochs)),
    #         'event_id': event_id,
    #         'tmin': cfg.tmin,
    #         'tmax': cfg.tmax,
    #         'baseline': cfg.baseline,
    #         'sfreq': float(epochs.info['sfreq']),
    #         'ch_names': epochs.ch_names
    #     }, f, indent=2)
    # print(f"Saved epochs metadata → {meta_path}")

    # # 4) TFR & ERD metrics
    # power = compute_tfr(epochs, cfg)
    # power_fig = Path(cfg.out_dir) / 'tfr_sensorimotor.png'
    # plot_tfr(power, cfg, str(power_fig))
    # print(f"Saved TFR plot → {power_fig}")

    # erd_df = extract_erd_metrics(epochs, power, cfg)
    # erd_csv = Path(cfg.out_dir) / 'erd_metrics.csv'
    # erd_df.to_csv(erd_csv, index=False)
    # print(f"Saved ERD metrics → {erd_csv}")

    # erd_fig = Path(cfg.out_dir) / 'erd_summary.png'
    # plot_erd_summary(erd_df, str(erd_fig))
    # print(f"Saved ERD summary plot → {erd_fig}")

    # # 5) Decoding
    # dec_df = run_decoding(epochs, cfg)
    # dec_csv = Path(cfg.out_dir) / 'decoding_summary.csv'
    # dec_df.to_csv(dec_csv, index=False)
    # print(f"Saved decoding summary → {dec_csv}")


# Main Pipeline
#%% CONFIG
# A) EEGBCI MI left-vs-right for subject 1, runs 4/8/12
# CFG = Config(source='eegbci', eegbci_subject=1, eegbci_runs=[4,8,12])

# B) Your CSV (adjust csv_event_code_map if needed)
CFG = Config(source='csv', csv_path='D:/user/Files_without_backup/MSc_Project/20250811_093558_eeg_with_markers.csv', #"D:/user/Files_without_backup/MSc_Project/20250808_093023_eeg_with_markers.csv", 
                csv_event_code_map={21:'left', 22:'right'},
                channel_cols=[f'ch{i}' for i in range(1,17)], csv_voltage_unit='uV')
#%% Load
if CFG.source.lower() == 'eegbci':
    raw, events, event_id = load_from_eegbci(CFG)
elif CFG.source.lower() == 'csv':
    raw, events, event_id = load_from_csv(CFG)
else:
    raise ValueError("Config.source must be 'eegbci' or 'csv'")

# print(f"events: {events}, {raw.n_times} samples")
# print(raw.get_data())

