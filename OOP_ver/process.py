#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Motor Imagery (MI) analysis — notebook-style .py

This script mirrors a Jupyter notebook layout using `#%%` cell markers so you can
run it top-to-bottom, section by section. It removes most helper functions and keeps
logic inline inside each section.

Pipeline (end-to-end):
1) Configure paths & parameters
2) Load data  (EEGBCI OR your CSV)
3) Preprocess (notch, band-pass, resample, re-reference)
4) Quick QC plot (raw vs filtered)
5) Epoch around MI cues
6) Time–Frequency (Morlet) → % change (ERD/ERS) + save TFR plots
7) ERD metrics per trial (mu/beta): peak ERD & onset latency; save CSV + summary plot
8) Decoding (CSP + LDA / linear SVM), 5-fold CV; save CSV

Tested with: mne>=1.6, numpy, scipy, pandas, scikit-learn, matplotlib
"""

#%% 1) Imports & global setup
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
mne.set_config('MNE_DATA', r'D:/data/mne_datasets', set_env=True)  # change if needed
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# Small helpers kept for clarity

def _to_volts(arr: np.ndarray, unit: str) -> np.ndarray:
    unit = (unit or '').lower()
    if unit in {"uv", "µv", "microvolt", "microvolts"}:  return arr / 1e6
    if unit in {"mv", "millivolt", "millivolts"}:          return arr / 1e3
    if unit in {"v", "volt", "volts"}:                      return arr
    raise ValueError(f"Unknown voltage unit: {unit}")

#%% 2) Configuration (edit here)
@dataclass
class Config:
    # Source selector: 'eegbci' or 'csv'
    source: str = 'eegbci'

    # EEGBCI specifics
    eegbci_subject: int = 5
    eegbci_runs: List[int] = None   # default below

    # CSV specifics (only used when source='csv')
    csv_path: Optional[str] = None
    time_col: str = 'timestamp'
    type_col: str = 'type'          # values: 'EEG' or 'Marker'
    eeg_value: str = 'EEG'
    marker_value: str = 'Marker'
    marker_col: str = 'marker'      # integer codes
    channel_cols: Optional[List[str]] = None  # if None auto-detect ch1..chN / eeg1..eegN
    csv_event_code_map: Dict[int, str] = None # e.g., {21:'left', 22:'right'}
    csv_voltage_unit: str = 'uV'             # 'uV' | 'mV' | 'V'

    # Preprocessing
    notch_freq: float = 50.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    resample_sfreq: Optional[float] = 250.0  # None to keep native
    reref: str | List[str] = 'average'       # 'average' or list of channel names

    # Epoching
    tmin: float = -1.0
    tmax: float = 4.0
    baseline: Tuple[Optional[float], Optional[float]] = (-1.0, 0.0)

    # TFR
    freqs: np.ndarray = None                 # default below
    n_cycles: float = 7.0
    ers_baseline_mode: str = 'percent'       # % change

    # ERD metrics
    mu_band: Tuple[float, float] = (8., 13.)
    beta_band: Tuple[float, float] = (13., 30.)
    erd_onset_threshold: float = -20.0       # % change (negative = drop)
    mi_window: Tuple[float, float] = (1.0, 4.0)
    sensorimotor_chs: List[str] = None       # default below

    # Decoding
    csp_components: int = 6
    cv_folds: int = 5
    random_state: int = 42

    # Output
    out_dir: str = 'outputs'

    def __post_init__(self):
        if self.eegbci_runs is None:
            self.eegbci_runs = [4, 8, 12]  # MI left vs right hand
        if self.freqs is None:
            self.freqs = np.arange(8, 31)
        if self.sensorimotor_chs is None:
            self.sensorimotor_chs = ['C3', 'C4', 'Cz']
        if self.csv_event_code_map is None:
            self.csv_event_code_map = {1: 'left', 2: 'right'}

CFG = Config(
    source='eegbci',
    eegbci_subject=5,
    eegbci_runs=[4, 8, 12],
    # Example for CSV:
    # source='csv',
    # csv_path=r'D:/user/Files_without_backup/MSc_Project/20250813_142853_eeg_with_markers.csv',
    # csv_event_code_map={21:'left', 22:'right'},
    # channel_cols=[f'ch{i}' for i in range(1,17)],
    # csv_voltage_unit='uV',
    # baseline=(-4, 0), tmin=-4, tmax=10
)

OUT = Path(CFG.out_dir); OUT.mkdir(parents=True, exist_ok=True)

#%% 3) Load data (EEGBCI OR CSV)
print(f"Source = {CFG.source}")
if CFG.source.lower() == 'eegbci':
    from mne.datasets import eegbci
    print(f"Fetching EEGBCI subject {CFG.eegbci_subject}, runs {CFG.eegbci_runs}…")
    file_paths = eegbci.load_data(CFG.eegbci_subject, CFG.eegbci_runs)

    raws = []
    for f in file_paths:
        r = mne.io.read_raw_edf(f, preload=True, verbose=False)
        eegbci.standardize(r)  # standardize channel names
        raws.append(r)
    raw = mne.concatenate_raws(raws)

    # Add montage so C3/C4 exist spatially
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')

    # Events: EEGBCI uses annotations 'T1' (left) and 'T2' (right)
    _map = {'T1': 1, 'T2': 2}
    events, _ = mne.events_from_annotations(raw, event_id=_map, verbose=False)
    event_id = {'left': 1, 'right': 2}

elif CFG.source.lower() == 'csv':
    if CFG.csv_path is None:
        raise ValueError("Set CFG.csv_path for source='csv'")
    print(f"Loading CSV: {CFG.csv_path}")
    df = pd.read_csv(CFG.csv_path, encoding='latin1', sep=',', engine='python')

    # Split EEG vs Marker rows
    if CFG.type_col not in df.columns:
        raise ValueError(f"CSV missing type column '{CFG.type_col}'")
    eeg_df = df[df[CFG.type_col] == CFG.eeg_value].copy()
    mrk_df = df[df[CFG.type_col] == CFG.marker_value].copy()

    if CFG.time_col not in df.columns or CFG.marker_col not in df.columns:
        raise ValueError("CSV must contain time and marker columns")

    # Auto-detect channel columns if needed
    ch_cols = CFG.channel_cols
    if ch_cols is None:
        patt = df.columns.str.match(r'^(ch|eeg)\d+$', case=False)
        ch_cols = [c for c, m in zip(df.columns, patt) if m and c in eeg_df.columns]
        if len(ch_cols) == 0:
            # fallback: all numeric columns except timestamp
            numeric_cols = [c for c in eeg_df.columns if np.issubdtype(eeg_df[c].dtype, np.number)]
            ch_cols = [c for c in numeric_cols if c != CFG.time_col]
    ch_cols = list(ch_cols)
    if len(ch_cols) == 0:
        raise ValueError("No channel columns found; set CFG.channel_cols explicitly")

    # Ensure sorted by timestamp and drop dups
    eeg_df = eeg_df.sort_values(CFG.time_col).drop_duplicates(subset=[CFG.time_col])

    # Estimate sampling frequency from median dt
    ts = eeg_df[CFG.time_col].to_numpy()
    dts = np.diff(ts)
    if len(dts) == 0:
        raise ValueError("Not enough EEG rows to estimate sampling rate")
    sfreq = float(np.round(1.0 / np.median(dts)))
    print(f"Estimated sampling rate: {sfreq:.1f} Hz")

    # Build RawArray (channels x samples), convert to Volts
    data = eeg_df[ch_cols].to_numpy().T
    data = _to_volts(data, CFG.csv_voltage_unit)
    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
    except Exception:
        pass

    # Build events array from marker rows
    mrk_df = mrk_df.sort_values(CFG.time_col)
    code_to_name = CFG.csv_event_code_map
    mrk_df = mrk_df[mrk_df[CFG.marker_col].isin(list(code_to_name.keys()))]

    mrk_ts = mrk_df[CFG.time_col].to_numpy()
    mrk_codes = mrk_df[CFG.marker_col].astype(int).to_numpy()

    # map codes -> contiguous ints
    name_to_int: Dict[str, int] = {name: i + 1 for i, name in enumerate(sorted(set(code_to_name.values())))}
    code_to_int = {code: name_to_int[code_to_name[code]] for code in code_to_name}

    start_time = eeg_df[CFG.time_col].iloc[0]
    events_list = []
    for t, code in zip(mrk_ts, mrk_codes):
        sample = int(np.round((t - start_time) * sfreq))
        if 0 <= sample < raw.n_times:
            events_list.append([sample, 0, code_to_int[code]])
    events = np.array(events_list, dtype=int)
    event_id = name_to_int

else:
    raise ValueError("CFG.source must be 'eegbci' or 'csv'")

print(f"Loaded events: {len(events)}  event_id={event_id}")

#%% 4) Preprocess (notch, band-pass, resample, re-reference)
raw_prep = raw.copy()
if CFG.notch_freq:
    raw_prep.notch_filter(freqs=[CFG.notch_freq], picks='eeg')
raw_prep.filter(l_freq=CFG.l_freq, h_freq=CFG.h_freq, picks='eeg')
if CFG.resample_sfreq is not None and CFG.resample_sfreq > 0:
    raw_prep.resample(CFG.resample_sfreq)
if CFG.reref == 'average':
    raw_prep.set_eeg_reference('average', projection=False)
elif isinstance(CFG.reref, (list, tuple)):
    raw_prep.set_eeg_reference(CFG.reref, projection=False)

#%% 5) Quick QC plot — raw vs filtered (first 5 s; common channels)
# Choose up to N channels (prefer sensorimotor)
max_channels = 8
common = [ch for ch in (CFG.sensorimotor_chs or []) if ch in raw.ch_names and ch in raw_prep.ch_names]
if not common:
    common = [ch for ch in raw_prep.ch_names if ch in raw.ch_names][:max_channels]
else:
    common = common[:max_channels]

# Align sampling rates
r0 = raw.copy()
r1 = raw_prep.copy()
if abs(r0.info['sfreq'] - r1.info['sfreq']) > 1e-6:
    r0.resample(r1.info['sfreq'])

# Crop to 0–5 s
window = (0.0, min(5.0, r0.times[-1], r1.times[-1]))
r0.crop(*window); r1.crop(*window)

picks1 = mne.pick_channels(r1.ch_names, include=common)
picks0 = mne.pick_channels(r0.ch_names, include=[r1.ch_names[i] for i in picks1])

data_r = r0.get_data(picks=picks0) * 1e6
data_f = r1.get_data(picks=picks1) * 1e6

fig, axes = plt.subplots(len(common), 1, figsize=(10, max(3, len(common)*1.4)), sharex=True)
if len(common) == 1: axes = [axes]
for ax, ch, y0, y1 in zip(axes, common, data_r, data_f):
    ax.plot(r1.times, y0, label='Raw', alpha=0.6, linewidth=0.8)
    ax.plot(r1.times, y1, label='Filtered', linewidth=1.0)
    ax.set_ylabel(f"{ch}\n(µV)")
    ax.grid(True, linestyle='--', alpha=0.3)
axes[0].legend(loc='upper right', frameon=False)
axes[-1].set_xlabel('Time (s)')
fig.suptitle(f"EEG time series • {len(common)} ch • {window[0]:.2f}–{window[1]:.2f} s", y=0.995)
fig.tight_layout()
qc_png = OUT / 'eeg_timeseries_raw_vs_filtered.png'
fig.savefig(qc_png, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"Saved QC plot → {qc_png}")

#%% 6) Epoch around events
picks = mne.pick_types(raw_prep.info, eeg=True, exclude='bads')
epochs = mne.Epochs(raw_prep, events, event_id=event_id, tmin=CFG.tmin, tmax=CFG.tmax,
                    baseline=CFG.baseline, picks=picks, preload=True, detrend=1)

# Save minimal metadata
meta_path = OUT / 'epochs_metadata.json'
with open(meta_path, 'w') as f:
    json.dump({
        'n_epochs': int(len(epochs)),
        'event_id': event_id,
        'tmin': CFG.tmin,
        'tmax': CFG.tmax,
        'baseline': CFG.baseline,
        'sfreq': float(epochs.info['sfreq']),
        'ch_names': epochs.ch_names
    }, f, indent=2)
print(f"Saved epochs metadata → {meta_path}")

#%% 7) Time–Frequency (Morlet) → % change baseline; save TFR plots
print("Computing Morlet TFR…")
st_power = mne.time_frequency.tfr_morlet(
    epochs, freqs=CFG.freqs, n_cycles=CFG.n_cycles, use_fft=True,
    return_itc=False, average=False, decim=1, n_jobs=None
)
st_power.apply_baseline(CFG.baseline, mode=CFG.ers_baseline_mode)

# Average across trials for visualization
power = st_power.average()

# Plot TFR for sensorimotor channels if present; else all
picks_tfr = [ch for ch in (CFG.sensorimotor_chs or []) if ch in power.ch_names] or None
figs = power.plot(picks=picks_tfr, baseline=None, mode=None, show=False)
base = OUT / 'tfr_sensorimotor'
if isinstance(figs, list):
    chs = (picks_tfr or power.ch_names)
    for fig, ch in zip(figs, chs):
        out_path = f"{base}_{ch}.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(fig)
else:
    out_path = f"{base}.png"; figs.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close(figs)
print("Saved TFR figure(s)")

#%% 8) ERD metrics per trial (mu/beta); save CSV + summary plot
print("Extracting ERD metrics…")
# Helper: band indices
mu_lo, mu_hi = CFG.mu_band
be_lo, be_hi = CFG.beta_band
freqs = CFG.freqs
mu_idx = np.where((freqs >= mu_lo) & (freqs <= mu_hi))[0]
beta_idx = np.where((freqs >= be_lo) & (freqs <= be_hi))[0]

# Choose channels (sensorimotor if present)
chs = [c for c in (CFG.sensorimotor_chs or []) if c in epochs.ch_names]
if not chs: chs = epochs.ch_names
ch_idx = [epochs.ch_names.index(c) for c in chs]

# MI window indices
mi_lo = int(np.round((CFG.mi_window[0] - epochs.tmin) * epochs.info['sfreq']))
mi_hi = int(np.round((CFG.mi_window[1] - epochs.tmin) * epochs.info['sfreq']))

rows = []
for ei in range(len(epochs)):
    cond_int = epochs.events[ei, 2]
    cond_name = {v: k for k, v in epochs.event_id.items()}[cond_int]

    for band_name, b_idx in [("mu", mu_idx), ("beta", beta_idx)]:
        # average across chosen channels and freqs in the band
        band_data = st_power.data[ei, ch_idx][:, b_idx, :]   # (n_ch, n_band_freqs, n_times)
        band_ts = band_data.mean(axis=1).mean(axis=0)        # (n_times,)

        mi_seg = band_ts[mi_lo:mi_hi]
        peak_erd = float(np.min(mi_seg))  # minimum % (ERD is negative)

        # Onset: first time it drops below threshold and stays 200 ms
        thr = CFG.erd_onset_threshold
        sustain = int(np.round(0.2 * epochs.info['sfreq']))
        onset = np.nan
        below = mi_seg < thr
        if below.any():
            idxs = np.where(below)[0]
            for idx in idxs:
                end = idx + sustain
                if end <= len(mi_seg) and np.all(mi_seg[idx:end] < thr):
                    onset = epochs.times[mi_lo + idx]
                    break

        rows.append({
            'trial': ei,
            'condition': cond_name,
            'channel': "+".join(chs),
            'band': band_name,
            'peak_erd_%': peak_erd,
            'onset_s': onset
        })

erd_df = pd.DataFrame(rows)
erd_csv = OUT / 'erd_metrics.csv'
erd_df.to_csv(erd_csv, index=False)
print(f"Saved ERD metrics → {erd_csv}")

# Summary bar plot (per band × condition)
try:
    import seaborn as sns
    plt.figure(figsize=(7,4))
    sns.barplot(data=erd_df, x='band', y='peak_erd_%', hue='condition', errorbar='se')
    plt.title('Peak ERD (% change) by condition')
    plt.tight_layout()
    erd_fig = OUT / 'erd_summary.png'
    plt.savefig(erd_fig, dpi=150); plt.close()
    print(f"Saved ERD summary plot → {erd_fig}")
except Exception:
    bands = sorted(erd_df['band'].unique())
    conds = sorted(erd_df['condition'].unique())
    means = np.array([[erd_df[(erd_df['band']==b)&(erd_df['condition']==c)]['peak_erd_%'].mean() for c in conds] for b in bands])
    x = np.arange(len(bands)); width = 0.35
    plt.figure(figsize=(7,4))
    for i, c in enumerate(conds):
        plt.bar(x + i*width, means[:, i], width, label=c)
    plt.xticks(x + width/2, bands)
    plt.ylabel('Peak ERD (%)'); plt.legend(); plt.title('Peak ERD by condition')
    plt.tight_layout()
    erd_fig = OUT / 'erd_summary.png'
    plt.savefig(erd_fig, dpi=150); plt.close()
    print(f"Saved ERD summary plot → {erd_fig}")

#%% 9) Decoding (CSP + LDA / linear SVM); 5-fold CV; save CSV
print("Running decoding…")
# Focus on MI window
ep = epochs.copy().crop(tmin=CFG.mi_window[0], tmax=CFG.mi_window[1])
# Decoding band-pass (8–30 Hz typical)
ep.load_data().filter(8., 30., picks='eeg')

X = ep.get_data()              # (n_epochs, n_channels, n_times)
y = ep.events[:, 2]
cv = StratifiedKFold(n_splits=CFG.cv_folds, shuffle=True, random_state=CFG.random_state)

results = []
for clf_name, clf in [("LDA", LDA()), ("LinearSVM", SVC(kernel='linear', C=1.0))]:
    pipe = Pipeline([
        ('csp', CSP(n_components=CFG.csp_components, reg='oas', log=True, norm_trace=False)),
        ('clf', clf)
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    mean_acc, std_acc = float(np.mean(scores)), float(np.std(scores))
    results.append({'classifier': clf_name, 'mean_accuracy': mean_acc, 'std': std_acc})
    print(f"  {clf_name}: {mean_acc:.3f} ± {std_acc:.3f}")

dec_df = pd.DataFrame(results)
dec_csv = OUT / 'decoding_summary.csv'
dec_df.to_csv(dec_csv, index=False)
print(f"Saved decoding summary → {dec_csv}")

#%% Done
print("All done ✔")
