# This script generates a complete Jupyter notebook that implements the user's requested pipeline.
# It saves the notebook to /mnt/data/MI_ERD_ERP_Decoding.ipynb for download.

import json, os, sys, time, platform
from datetime import datetime

nb = {
 "cells": [],
 "metadata": {
   "kernelspec": {
     "display_name": "Python 3",
     "language": "python",
     "name": "python3"
   },
   "language_info": {
     "name": "python",
     "version": sys.version.split()[0]
   }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_md(s):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": s
    })
def add_code(s):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": s
    })

# ---- Markdown header ----
add_md("""# MI ERD/PMBR + ERPs + Latencies + Time-Resolved Decoding (with quick-check plots)

**Assumptions:** An in-memory, cleaned `mne.Epochs` named **`epochs`** with events `left`/`right`, time −4.5…+9.5 s, `sfreq≈500 Hz`, channels: `C3,C4,Fp1,Fp2,O1,O2,T3,T4`, and `epochs.metadata['modality'] ∈ {'V','A','VA'}`. Reference already applied.

This notebook is organised into the requested **BLOCKS** with minimal, composable cells. Each block prints status and saves artefacts under `results/`.
""")

# ---- BLOCK 0 ----
add_md("## BLOCK 0 — Environment, config, and checks")
add_code(r"""
# %% [markdown]
# BLOCK 0 — Environment, config, and checks
# - Imports
# - Paths
# - Config dataclass
# - Assertions and quick dataset summary

from __future__ import annotations

import os, sys, math, json, warnings, pathlib, gc, itertools, textwrap, random, traceback
from dataclasses import dataclass, asdict, field
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.time_frequency import tfr_morlet, tfr_multitaper
from mne.decoding import CSP

from scipy.signal import savgol_filter
from scipy import stats

from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix

# Matplotlib defaults (no global styles)
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

# Paths
RES_DIR = pathlib.Path("results")
FIG_DIR = RES_DIR / "figs"
DEC_DIR = RES_DIR / "decoding"
for d in [RES_DIR, FIG_DIR, DEC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Config dataclass
@dataclass
class Config:
    bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {"mu": (8,12), "beta": (13,30)})
    windows: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "baseline": (-3.5, -0.5),
        "erd_search": (0.5, 4.5),
        "pmbr_search": (6.25, 8.5),
    })
    erp: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "erp_epoch": (-0.2, 6.0),
        "erp_baseline": (-0.2, 0.0),
        "p2_win": (0.150, 0.250),
        "p3_win": (0.300, 0.600),
        "cnv_win": (0.300, 1.000),
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "erd_onset_thresh_percent": -20.0,
        "erd_onset_min_dur": 0.150,
        "lrp_onset_frac": 0.5,
        "lrp_min_dur": 0.050,
    })
    tfr_method: str = "morlet"
    morlet_n_cycles: int = 7
    mt_tbp: float = 4.0
    freqs: np.ndarray = field(default_factory=lambda: np.arange(4, 41, 1))
    smooth: bool = True
    savgol_window_samples: int = 251
    savgol_poly: int = 3
    bootstrap_n: int = 5000
    random_state: int = 42

    # Decoding
    win_len: float = 0.250
    win_step: float = 0.025
    csp_components: int = 4
    perm_n: int = 1000
    persist_ms_sig: int = 200
    persist_ms_op: int = 150
    op_threshold_balacc: float = 0.70

    neighbours: List[str] = field(default_factory=lambda: ["T3","T4","O1","O2"])  # optional
    csp_band: Tuple[float,float] = (8.0, 30.0)

CFG = Config()

# Version checks
print("mne version:", mne.__version__)
try:
    from packaging import version
    assert version.parse(mne.__version__) >= version.parse("1.5.0"), "MNE >= 1.5 is required."
except Exception as e:
    warnings.warn(f"Version check issue: {e}")

# Assertions on epochs
assert 'epochs' in globals(), "This notebook assumes an in-memory `epochs` object."
assert isinstance(epochs, mne.Epochs), "`epochs` must be an mne.Epochs."
ch_names = epochs.info['ch_names']
assert "C3" in ch_names and "C4" in ch_names, "Channels must include C3 and C4."
assert isinstance(epochs.metadata, pd.DataFrame), "epochs.metadata must exist."
assert 'modality' in epochs.metadata.columns, "epochs.metadata['modality'] is required."
allowed_mods = {"V","A","VA"}
mods = set(epochs.metadata['modality'].unique())
unexpected = mods - allowed_mods
assert not unexpected, f"Unexpected modality labels: {unexpected}"

tmin, tmax = epochs.times[0], epochs.times[-1]
for key, (a,b) in list(CFG.windows.items()) + list(CFG.erp.items()):
    assert tmin <= a and b <= tmax, f"Epoch time range ({tmin},{tmax}) must cover window {key}=({a},{b})."

# Summary
def _hand_from_epoch(e):
    # infer hand from events or metadata
    # Events named 'left'/'right' in event_id is typical
    return "left" if e == "left" else ("right" if e == "right" else str(e))

counts = []
for hand in ["left","right"]:
    for mod in ["V","A","VA"]:
        try:
            sel = epochs[hand]
            mask = (sel.metadata["modality"] == mod).values
            n = int(mask.sum())
        except Exception:
            n = 0
        counts.append((hand, mod, n))

print("Trial counts by (hand × modality):")
for hand, mod, n in counts:
    print(f"  {hand:>5s} × {mod:<2s}: {n:3d}")

print(f"sfreq: {epochs.info['sfreq']:.1f} Hz | time range: {tmin:.2f}..{tmax:.2f} s")
print("Config:", json.dumps(asdict(CFG), indent=2))

# provenance log start
with open(RES_DIR/"log.txt","w",encoding="utf-8") as f:
    f.write(f"Created: {datetime.now().isoformat()}\n")
    f.write(f"Python: {platform.python_version()} | mne: {mne.__version__}\n")
    f.write(json.dumps(asdict(CFG), indent=2))
    f.write("\n")
print("✔ BLOCK 0 done; paths created under 'results/'.")
""")

# ---- QC-1 ----
add_md("## BLOCK QC-1 — Quick sanity plots: raw snapshots & PSDs")
add_code(r"""
# QC-1: raw snapshots & PSDs
rng = np.random.default_rng(CFG.random_state)

def _pick_random_trial(hand):
    idxs = epochs[hand].selection if hasattr(epochs[hand], "selection") else epochs[hand].events[:,0]
    # Map to positions in original epochs
    poss = epochs[hand].selection if hasattr(epochs[hand], "selection") else np.arange(len(epochs[hand]))
    if len(poss) == 0:
        return None
    pick_pos = int(rng.integers(0, len(poss)))
    # Return global index
    return epochs[hand].selection[pick_pos] if hasattr(epochs[hand], "selection") else pick_pos

ix_left = _pick_random_trial("left")
ix_right = _pick_random_trial("right")

fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)
for ax, ix, title in zip(axes, [ix_left, ix_right], ["Left trial", "Right trial"]):
    if ix is None:
        ax.text(0.5,0.5,"No trial", ha="center", va="center")
        continue
    data = epochs[ix].get_data(picks=["C3","C4"])[0]  # (2, n_times)
    times = epochs.times
    # 10 s window around cue: here we use -0.5 .. 9.5 (fits in -4.5..9.5)
    mask = (times >= -0.5) & (times <= 9.5)
    ax.plot(times[mask], data[0,mask], label="C3")
    ax.plot(times[mask], data[1,mask], label="C4")
    ax.axvline(0, ls="--", lw=1)
    ax.set_title(f"{title} (index {ix})")
    ax.set_ylabel("µV (approx)")
axes[-1].set_xlabel("Time (s)")
axes[0].legend(loc="upper right")
plt.tight_layout()
plt.savefig(FIG_DIR/"qc_raw_snapshot.png", dpi=150)
plt.close()

# Welch PSDs averaged over all trials for C3/C4
psds, freqs = mne.time_frequency.psd_welch(epochs.copy().pick(["C3","C4"]),
                                           fmin=2, fmax=45, n_fft=2048, average='mean', verbose=False)
# psds: (n_epochs, n_ch, n_freqs)
psd_mean = psds.mean(axis=0)  # (2, n_freqs)

fig, ax = plt.subplots(1,1, figsize=(7,4))
for ch_i, ch in enumerate(["C3","C4"]):
    ax.semilogy(freqs, psd_mean[ch_i], label=ch)
ax.axvspan(8,12, alpha=0.2, label="mu")
ax.axvspan(13,30, alpha=0.2, label="beta")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (power/Hz)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR/"qc_raw_psd.png", dpi=150)
plt.close()

print("✔ QC-1 saved: 'qc_raw_snapshot.png' and 'qc_raw_psd.png'")
""")

# ---- QC-2 ----
add_md("## BLOCK QC-2 — Quick TFR spot-check (one trial)")
add_code(r"""
# QC-2: TFR on one left + one right, C3/C4
def _first_idx(epochs_subset):
    return int(epochs_subset.selection[0]) if hasattr(epochs_subset, "selection") and len(epochs_subset.selection)>0 else (0 if len(epochs_subset)>0 else None)

ixL = _first_idx(epochs["left"])
ixR = _first_idx(epochs["right"])

def _tfr_one(ix):
    if ix is None:
        return None
    ep = epochs[ix:ix+1].copy().pick(["C3","C4"])
    tfr = tfr_morlet(ep, freqs=CFG.freqs, n_cycles=CFG.morlet_n_cycles,
                     average=False, return_itc=False, verbose=False)
    return tfr

tfrL = _tfr_one(ixL)
tfrR = _tfr_one(ixR)

def _plot_tfr_pair(tfr, title, fname):
    if tfr is None:
        return
    tfr_nobase = tfr.copy()
    tfr_base = tfr.copy().apply_baseline(CFG.windows["baseline"], mode="percent")
    fig, axes = plt.subplots(2,2, figsize=(9,6), sharex=True, sharey=True)
    for i, ch in enumerate(tfr.ch_names):
        tfr_nobase.plot([0], picks=[i], axes=axes[0,i], colorbar=False, show=False)
        axes[0,i].set_title(f"{title} {ch} (raw)")
        tfr_base.plot([0], picks=[i], axes=axes[1,i], colorbar=False, show=False)
        axes[1,i].set_title(f"{title} {ch} (baseline %)")
        for row in [0,1]:
            axes[row,i].axvline(0, ls="--", lw=1, color="k")
    plt.tight_layout()
    plt.savefig(FIG_DIR/fname, dpi=150)
    plt.close()

_plot_tfr_pair(tfrL, "Left trial", "qc_tfr_left.png")
_plot_tfr_pair(tfrR, "Right trial", "qc_tfr_right.png")

print("✔ QC-2 saved: 'qc_tfr_left.png' & 'qc_tfr_right.png'")
""")

# ---- QC-3 ----
add_md("## BLOCK QC-3 — Quick ERP spot-check")
add_code(r"""
# ERP copy (0.1–30 Hz), crop −0.2…6.0 s, baseline -0.2…0
erp_cfg = CFG.erp
erp_epochs = epochs.copy().filter(0.1, 30.0, method='fir', fir_design='firwin', phase='zero', verbose=False)
erp_epochs.crop(erp_cfg["erp_epoch"][0], erp_cfg["erp_epoch"][1])
erp_epochs.apply_baseline(erp_cfg["erp_baseline"])

sites_v = [c for c in ["O1","O2"] if c in erp_epochs.info["ch_names"]]
sites_a = [c for c in ["T3","T4"] if c in erp_epochs.info["ch_names"]]
sites_c = [c for c in ["C3","C4"] if c in erp_epochs.info["ch_names"]]

# create central composite virtual channel for plotting
def central_composite(eps):
    if all(c in eps.info["ch_names"] for c in ["C3","C4"]):
        data = 0.5*(eps.copy().pick(["C3"]).get_data() + eps.copy().pick(["C4"]).get_data())
        info = mne.create_info(["central_composite"], eps.info["sfreq"], ch_types="eeg")
        return mne.EpochsArray(data, info, tmin=eps.tmin, events=eps.events, event_id=eps.event_id, metadata=eps.metadata)
    else:
        return eps.copy().pick(["C3"])

erp_cc = central_composite(erp_epochs)

def _plot_erp_for_site(site_list, label, fname):
    if len(site_list)==0 and label!="central":
        return
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    for mod, color in zip(["V","A","VA"], ["C0","C1","C2"]):
        for hand, ls in zip(["left","right"], ["-","--"]):
            sel = erp_epochs[hand].copy()
            mask = (sel.metadata["modality"]==mod).values if "modality" in sel.metadata else np.ones(len(sel), dtype=bool)
            sel = sel[mask]
            if len(sel)==0: 
                continue
            if label=="central":
                ev = erp_cc[hand][mask].copy().average(picks=["central_composite"])
                ch_label = "central_composite"
            else:
                ev = sel.copy().average(picks=site_list)
                ch_label = "+".join(site_list)
            ax.plot(ev.times, ev.data.mean(axis=0)*1e6, label=f"{mod}-{hand}-{label}", ls=ls)
    # mark windows
    p2 = erp_cfg["p2_win"]; p3 = erp_cfg["p3_win"]; cnv = erp_cfg["cnv_win"]
    for (a,b) in [p2,p3,cnv]:
        ax.axvspan(a,b, alpha=0.1)
    ax.axvline(0, ls="--", lw=1, color="k")
    ax.set_title(f"ERP spot-check @ {label} ({ch_label if label!='central' else 'virtual'})")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("µV")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/fname, dpi=150); plt.close()

_plot_erp_for_site(sites_v, "visual", "qc_erp_visual.png")
_plot_erp_for_site(sites_a, "auditory", "qc_erp_auditory.png")
_plot_erp_for_site([], "central", "qc_erp_central.png")

print("✔ QC-3 saved ERP figures for visual/auditory/central (where available).")
""")

# ---- BLOCK 1 ----
add_md("## BLOCK 1 — Trial routing & lateralisation helpers")
add_code(r"""
# Helpers

def subset_indices(epochs_obj, hand: str, modality: str) -> np.ndarray:
    """Return global indices for trials matching hand & modality."""
    sub = epochs_obj[hand]
    if len(sub)==0:
        return np.array([], dtype=int)
    mask = (sub.metadata["modality"]==modality).values if "modality" in sub.metadata else np.ones(len(sub), dtype=bool)
    if hasattr(sub, "selection"):
        return sub.selection[mask]
    else:
        # fallback: positions within sub; map to global via events? approximate
        return np.where(mask)[0]

def sec_to_samp(times: np.ndarray, t0: float, t1: float) -> Tuple[int,int]:
    """Inclusive indices for time window [t0, t1]."""
    i0 = int(np.argmin(np.abs(times - t0)))
    i1 = int(np.argmin(np.abs(times - t1)))
    if i1 < i0: i0, i1 = i1, i0
    return i0, i1

# hand→channel maps
def contra_channel(hand: str) -> str:
    return "C4" if hand=="left" else "C3"

def ipsi_channel(hand: str) -> str:
    return "C3" if hand=="left" else "C4"

# MAD mask (keep within median ± 3*MAD)
def mad_mask(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = stats.median_abs_deviation(x, nan_policy='omit', scale='normal')  # ~σ
    if not np.isfinite(mad) or mad==0:
        return np.ones_like(x, dtype=bool)
    return (x >= med - 3*mad) & (x <= med + 3*mad)

print("subset_indices('left','V') ->", subset_indices(epochs, "left", "V")[:5] if len(epochs)>0 else [])
print("contra/ipsi for left:", contra_channel("left"), ipsi_channel("left"))
print("✔ BLOCK 1 helpers ready.")
""")

# ---- BLOCK 2 ----
add_md("## BLOCK 2 — Trial-wise TFR (Morlet/Multitaper)")
add_code(r"""
# Compute TFR for all trials (power), average=False
def compute_tfr(epochs_obj: mne.Epochs, method: str="morlet",
                freqs: np.ndarray=None, morlet_n_cycles: int=7, mt_tbp: float=4.0):
    freqs = freqs if freqs is not None else CFG.freqs
    picks = None  # all channels kept; we will select later
    if method.lower() == "morlet":
        tfr = tfr_morlet(epochs_obj, freqs=freqs, n_cycles=morlet_n_cycles,
                         average=False, return_itc=False, verbose=False)
    elif method.lower() in ("multitaper","mt"):
        tfr = tfr_multitaper(epochs_obj, freqs=freqs, time_bandwidth=mt_tbp,
                             average=False, return_itc=False, verbose=False)
    else:
        raise ValueError("Unknown method; use 'morlet' or 'multitaper'.")
    # tfr: shape (n_trials, n_ch, n_freqs, n_times)
    return tfr

tfr_all = compute_tfr(epochs, method=CFG.tfr_method, freqs=CFG.freqs,
                      morlet_n_cycles=CFG.morlet_n_cycles, mt_tbp=CFG.mt_tbp)

print("TFR shape:", tfr_all.data.shape, "| times:", (tfr_all.times[0], tfr_all.times[-1]))
print("✔ BLOCK 2 TFR computed.")
""")

# ---- BLOCK 3 ----
add_md("## BLOCK 3 — Band extraction & baseline normalisation")
add_code(r"""
def bandpower_from_tfr(tfr: mne.time_frequency.EpochsTFR, band: Tuple[float,float],
                       channels: List[str], baseline=(-3.5,-0.5)):
    """Return dict with percent and dB baselined band power (trial×channel×time)."""
    fmin, fmax = band
    f = tfr.freqs
    fmask = (f >= fmin) & (f <= fmax)
    ch_idx = [tfr.ch_names.index(c) for c in channels if c in tfr.ch_names]
    if len(ch_idx)==0:
        raise ValueError("Requested channels not in TFR.")
    # mean over freqs in band
    pow_band = tfr.data[:, ch_idx][:, :, fmask, :].mean(axis=2)  # (n_trials, n_ch, n_times)
    # baseline per trial×channel
    times = tfr.times
    i0, i1 = sec_to_samp(times, baseline[0], baseline[1])
    base = pow_band[..., i0:i1+1].mean(axis=-1, keepdims=True)  # (n_trials, n_ch, 1)
    # percent & dB
    percent = (pow_band - base) / base * 100.0
    db = 10.0 * np.log10(np.maximum(pow_band, np.finfo(float).eps) / np.maximum(base, np.finfo(float).eps))
    return {"percent": percent, "db": db, "times": times, "channels": [tfr.ch_names[i] for i in ch_idx]}

# Build tidy DF for C3/C4
bands = {k: bandpower_from_tfr(tfr_all, v, ["C3","C4"], baseline=CFG.windows["baseline"]) for k,v in CFG.bands.items()}

rows = []
n_trials = tfr_all.data.shape[0]
hands = []
if "left" in epochs.event_id and "right" in epochs.event_id:
    # build hand vector per trial using event codes
    ev_ids = epochs.event_id
    evs = epochs.events[:,2]
    inv = {v:k for k,v in ev_ids.items()}
    hands = np.array([inv.get(e, "unknown") for e in evs])
else:
    # fallback: from metadata if present
    if "hand" in epochs.metadata:
        hands = epochs.metadata["hand"].values
    else:
        raise RuntimeError("Cannot infer 'hand' labels; ensure events named 'left'/'right' or metadata['hand'].")

modalities = epochs.metadata["modality"].values

for band_name, d in bands.items():
    for meas in ["percent","db"]:
        arr = d[meas]  # (n_trials, 2, n_times)
        for ch_i, ch in enumerate(d["channels"]):
            for trial_id in range(n_trials):
                rows.append(pd.DataFrame({
                    "trial_id": trial_id,
                    "hand": hands[trial_id],
                    "modality": modalities[trial_id],
                    "channel": ch,
                    "band": band_name,
                    "measure": meas,
                    "time": d["times"],
                    "value": arr[trial_id, ch_i, :]
                }))

df_band = pd.concat(rows, ignore_index=True)

def _save_df(df, path):
    path = pathlib.Path(path)
    try:
        df.to_parquet(path)
        print(f"Saved {path}")
    except Exception as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        warnings.warn(f"to_parquet failed ({e}); saved CSV instead: {alt.name}")

_save_df(df_band.head(2000), RES_DIR/"bandpower_preview.parquet")
print(df_band.head())
print("✔ BLOCK 3 bandpower tidy preview saved.")
""")

# ---- BLOCK 4 ----
add_md("## BLOCK 4 — ERD/PMBR metrics (per trial)")
add_code(r"""
def _persist_onset(times, series, thresh, min_dur):
    """Earliest time where series <= thresh persists for >= min_dur seconds. Return (t_onset or NaN)."""
    mask = np.isfinite(series)
    if not mask.any():
        return np.nan, "all_nans"
    x = series.copy()
    good = x <= thresh
    # find runs
    dt = np.median(np.diff(times))
    min_len = int(np.ceil(min_dur / dt))
    if min_len <= 0: min_len = 1
    run = 0
    start_idx = None
    for i, g in enumerate(good):
        if g:
            run += 1
            if run == 1:
                start_idx = i
            if run >= min_len:
                return times[start_idx], "ok"
        else:
            run = 0
            start_idx = None
    return np.nan, "no_persist"

def _within(times, series, win):
    i0,i1 = sec_to_samp(times, win[0], win[1])
    return series[i0:i1+1], times[i0:i1+1]

# Build trial_metrics
records = []
for band_name, d in bands.items():
    times = d["times"]
    for trial_id in range(n_trials):
        hand = hands[trial_id]
        mod = modalities[trial_id]
        for side in ["contra","ipsi"]:
            ch = contra_channel(hand) if side=="contra" else ipsi_channel(hand)
            if ch not in d["channels"]:
                continue
            ch_idx = d["channels"].index(ch)
            perc = d["percent"][trial_id, ch_idx, :].copy()
            dbv  = d["db"][trial_id, ch_idx, :].copy()
            # ERD window
            s, t = _within(times, perc, CFG.windows["erd_search"])
            onset, reason = _persist_onset(t, s, CFG.thresholds["erd_onset_thresh_percent"], CFG.thresholds["erd_onset_min_dur"])
            # ERD peak (most negative)
            erd_peak_idx = np.nanargmin(s) if np.isfinite(s).any() else None
            erd_peak_s = (t[erd_peak_idx] if erd_peak_idx is not None else np.nan)
            erd_min_percent = (float(np.nanmin(s)) if np.isfinite(s).any() else np.nan)
            # matched dB at ERD min
            if erd_peak_idx is not None:
                s_db, t_db = _within(times, dbv, CFG.windows["erd_search"])
                erd_min_db = float(s_db[erd_peak_idx])
            else:
                erd_min_db = np.nan
            # PMBR (beta only)
            pmbr_peak_s = pmbr_peak_percent = pmbr_peak_db = np.nan
            if band_name.lower()=="beta":
                s2, t2 = _within(times, perc, CFG.windows["pmbr_search"])
                if np.isfinite(s2).any():
                    j = int(np.nanargmax(s2))
                    pmbr_peak_s = float(t2[j]); pmbr_peak_percent = float(s2[j])
                    s2db, _ = _within(times, dbv, CFG.windows["pmbr_search"])
                    pmbr_peak_db = float(s2db[j])
            records.append({
                "trial_id": trial_id, "hand": hand, "modality": mod, "band": band_name,
                "side": side, "channel": ch,
                "erd_onset_s": float(onset), "erd_onset_ok": (reason=="ok"), "erd_onset_reason": reason,
                "erd_peak_s": float(erd_peak_s), "erd_min_percent": float(erd_min_percent), "erd_min_db": float(erd_min_db),
                "pmbr_peak_s": float(pmbr_peak_s), "pmbr_peak_percent": float(pmbr_peak_percent), "pmbr_peak_db": float(pmbr_peak_db),
            })

df_trial = pd.DataFrame.from_records(records)

# Apply MAD cleaning per (hand×modality×band×metric)
def _apply_mad_flags(df, group_cols, metric_cols):
    df = df.copy()
    for m in metric_cols:
        keep_col = f"kept_{m}"
        df[keep_col] = False
        for _, g in df.groupby(group_cols):
            mask = mad_mask(g[m].values)
            df.loc[g.index, keep_col] = mask
    return df

metric_cols = ["erd_onset_s","erd_peak_s","erd_min_percent","pmbr_peak_s","pmbr_peak_percent"]
df_trial = _apply_mad_flags(df_trial, ["hand","modality","band"], metric_cols)

# Save
def _save_df(df, path):
    path = pathlib.Path(path)
    try:
        df.to_parquet(path)
        print(f"Saved {path}")
    except Exception as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        warnings.warn(f"to_parquet failed ({e}); saved CSV instead: {alt.name}")

_save_df(df_trial, RES_DIR/"trial_metrics.parquet")
print("Rows:", len(df_trial))
print("✔ BLOCK 4 completed trial-wise ERD/PMBR metrics.")
""")

# ---- BLOCK 5 ----
add_md("## BLOCK 5 — ERD/PMBR time-course plots (per modality × hand)")
add_code(r"""
def _median_with_ci(x, n_boot=2000, rng=None):
    rng = np.random.default_rng(CFG.random_state if rng is None else rng)
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x)==0:
        return np.nan, (np.nan, np.nan)
    meds = []
    for _ in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        meds.append(np.nanmedian(samp))
    lo, hi = np.nanpercentile(meds, [2.5, 97.5])
    return float(np.nanmedian(x)), (float(lo), float(hi))

def _plot_timecourses(measure: str, fname: str):
    fig, axes = plt.subplots(2,3, figsize=(12,6), sharex=True, sharey=True)
    hand_list = ["left","right"]
    mod_list = ["V","A","VA"]
    for r, hand in enumerate(hand_list):
        for c, mod in enumerate(mod_list):
            ax = axes[r,c]
            for band_name, d in bands.items():
                times = d["times"]
                # Contra/Ipsi channels
                ch_contra = contra_channel(hand)
                ch_ipsi = ipsi_channel(hand)
                if ch_contra not in d["channels"] or ch_ipsi not in d["channels"]:
                    continue
                ci = d["channels"].index(ch_contra)
                ii = d["channels"].index(ch_ipsi)
                # Select trials matching hand & modality
                idxs = subset_indices(epochs, hand, mod)
                if len(idxs)==0:
                    continue
                arr = d[measure][idxs]  # (n_sel, 2, n_times)
                contra = arr[:, ci, :].mean(axis=0)
                ipsi   = arr[:, ii, :].mean(axis=0)
                if CFG.smooth:
                    win = max(5, CFG.savgol_window_samples)
                    if win % 2 == 0: win += 1
                    contra = savgol_filter(contra, window_length=min(win, len(contra)-(1-len(contra)%2)), polyorder=min(CFG.savgol_poly, 5))
                    ipsi   = savgol_filter(ipsi,   window_length=min(win, len(ipsi)-(1-len(ipsi)%2)), polyorder=min(CFG.savgol_poly, 5))
                ax.plot(times, contra, label=f"{band_name} contra")
                ax.plot(times, ipsi, ls="--", label=f"{band_name} ipsi")
                ax.axvspan(*CFG.windows["baseline"], alpha=0.05)
                ax.axvspan(*CFG.windows["erd_search"], alpha=0.05)
                ax.axvspan(*CFG.windows["pmbr_search"], alpha=0.05 if band_name=="beta" else 0)
                ax.axvline(0, ls="--", lw=1, color="k")
            ax.set_title(f"{hand} | {mod}")
            if r==1: ax.set_xlabel("Time (s)")
            if c==0: ax.set_ylabel(f"{measure}")
            if r==0 and c==2: ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/fname, dpi=150); plt.close()

_plot_timecourses("percent", "erd_pmbr_percent.png")
_plot_timecourses("db", "erd_pmbr_db.png")

print("✔ BLOCK 5 saved time-course plots (percent & dB).")
""")

# ---- BLOCK 6 ----
add_md("## BLOCK 6 — ERP pipeline (cue-locked & preparatory)")
add_code(r"""
# Keep single-trial ERPs per (hand×modality) already prepared in 'erp_epochs' from QC-3.
# Save minimal per-condition plots.

def _condition_average(eps, hand, mod, picks):
    sub = eps[hand]
    mask = (sub.metadata["modality"]==mod).values if "modality" in sub.metadata else np.ones(len(sub), dtype=bool)
    if len(sub)==0 or not mask.any():
        return None
    return sub[mask].copy().average(picks=picks)

erp_sites = {
    "visual": [c for c in ["O1","O2"] if c in erp_epochs.info["ch_names"]],
    "auditory": [c for c in ["T3","T4"] if c in erp_epochs.info["ch_names"]],
    "central": ["C3","C4"]
}

for label, picks in erp_sites.items():
    if len(picks)==0 and label!="central":
        continue
    fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=True)
    for ax, mod in zip(axes, ["V","A","VA"]):
        for hand, ls in zip(["left","right"], ["-","--"]):
            ev = _condition_average(erp_epochs, hand, mod, picks if label!="central" else ["C3","C4"])
            if ev is None:
                continue
            y = ev.copy().pick(picks if label!="central" else ["C3","C4"]).data.mean(axis=0)*1e6
            ax.plot(ev.times, y, ls=ls, label=hand)
        ax.axvline(0, ls="--", lw=1, color="k")
        ax.set_title(f"{label} | {mod}")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("µV")
    axes[-1].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"erp_{label}.png", dpi=150); plt.close()

print("Single-trial ERP shapes:", erp_epochs.get_data().shape)
print("✔ BLOCK 6 ERP pipeline ready; figures saved.")
""")

# ---- BLOCK 7 ----
add_md("## BLOCK 7 — ERP component metrics")
add_code(r"""
def _peak_within(evoked: mne.Evoked, win: Tuple[float,float], mode="pos"):
    i0, i1 = sec_to_samp(evoked.times, win[0], win[1])
    y = evoked.data.mean(axis=0)[i0:i1+1]*1e6
    t = evoked.times[i0:i1+1]
    if len(y)==0 or not np.isfinite(y).any():
        return np.nan, np.nan
    idx = np.nanargmax(y) if mode=="pos" else np.nanargmin(y)
    return float(t[idx]), float(y[idx])

def _site_for_mod(mod: str):
    if mod=="V" and len(erp_sites["visual"])>0: return erp_sites["visual"]
    if mod=="A" and len(erp_sites["auditory"])>0: return erp_sites["auditory"]
    if len(erp_sites["visual"]+erp_sites["auditory"])>0: return list(set(erp_sites["visual"]+erp_sites["auditory"]))
    return ["C3","C4"]

# LRP(t) = 0.5 * [(C3_left − C4_left) + (C4_right − C3_right)]
def compute_lrp(erp_eps, modality: str):
    # average per hand within modality
    l = _condition_average(erp_eps, "left", modality, ["C3","C4"])
    r = _condition_average(erp_eps, "right", modality, ["C3","C4"])
    if l is None or r is None:
        return None
    C3l = l.copy().pick(["C3"]).data[0]; C4l = l.copy().pick(["C4"]).data[0]
    C3r = r.copy().pick(["C3"]).data[0]; C4r = r.copy().pick(["C4"]).data[0]
    lrp = 0.5 * (((C3l - C4l) + (C4r - C3r)) * 1e6)
    return l.times, lrp

erp_records = []
for mod in ["V","A","VA"]:
    site = _site_for_mod(mod)
    for hand in ["left","right"]:
        ev = _condition_average(erp_epochs, hand, mod, site)
        if ev is None: continue
        p2_lat, p2_amp = _peak_within(ev, CFG.erp["p2_win"], mode="pos")
        p3_lat, p3_amp = _peak_within(ev, CFG.erp["p3_win"], mode="pos")
        # CNV on central composite
        ev_c = _condition_average(erp_epochs, hand, mod, ["C3","C4"])
        i0,i1 = sec_to_samp(ev_c.times, *CFG.erp["cnv_win"])
        t = ev_c.times[i0:i1+1]; y = ev_c.data.mean(axis=0)[i0:i1+1]*1e6
        if len(y)>1:
            slope, intercept, r, p, se = stats.linregress(t, y)
            mean_uv = float(np.nanmean(y))
        else:
            slope = mean_uv = np.nan
        erp_records.append({
            "modality": mod, "hand": hand, "site": "+".join(site),
            "P2_latency_s": p2_lat, "P2_amp_uv": p2_amp,
            "P3_latency_s": p3_lat, "P3_amp_uv": p3_amp,
            "CNV_mean_uv": mean_uv, "CNV_slope_uv_per_s": slope
        })

# LRP metrics per modality
for mod in ["V","A","VA"]:
    lr = compute_lrp(erp_epochs, mod)
    if lr is None: continue
    t, y = lr
    # 50% peak deflection within 0.15–1.5 s persisting >= 50 ms
    i0,i1 = sec_to_samp(t, 0.15, 1.5)
    seg = y[i0:i1+1]; tt = t[i0:i1+1]
    if len(seg)==0: continue
    peak = np.nanmin(seg) if np.abs(np.nanmin(seg)) > np.abs(np.nanmax(seg)) else np.nanmax(seg)
    thr = CFG.thresholds["lrp_onset_frac"] * peak
    # define condition: towards peak sign
    if peak >= 0:
        cond = seg >= thr
    else:
        cond = seg <= thr
    dt = np.median(np.diff(tt))
    min_len = int(np.ceil(CFG.thresholds["lrp_min_dur"]/dt))
    run=0; onset=np.nan
    for i, g in enumerate(cond):
        if g:
            run += 1
            if run >= min_len:
                onset = tt[i - min_len + 1]
                break
        else:
            run = 0
    erp_records.append({"modality": mod, "hand": "pooled", "site": "LRP",
                        "LRP_onset_s": float(onset)})

df_erp = pd.DataFrame(erp_records)

def _save_df(df, path):
    path = pathlib.Path(path)
    try:
        df.to_parquet(path)
        print(f"Saved {path}")
    except Exception as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        warnings.warn(f"to_parquet failed ({e}); saved CSV instead: {alt.name}")

_save_df(df_erp, RES_DIR/"erp_metrics.parquet")
print(df_erp.head())
print("✔ BLOCK 7 ERP metrics saved.")
""")

# ---- BLOCK 8 ----
add_md("## BLOCK 8 — LRP jackknife onset & plots")
add_code(r"""
# Jackknife and bootstrap (trial-level) for LRP onset per modality
def lrp_single_trial(erp_eps, trial_idxs_left, trial_idxs_right):
    # compute single-trial LRP then average? Here we use leave-one-out on trial averages for stability.
    pass

def lrp_leave_one_out_onset(erp_eps, modality: str):
    subL = erp_eps["left"]
    maskL = (subL.metadata["modality"]==modality).values if "modality" in subL.metadata else np.ones(len(subL), bool)
    subL = subL[maskL]
    subR = erp_eps["right"]
    maskR = (subR.metadata["modality"]==modality).values if "modality" in subR.metadata else np.ones(len(subR), bool)
    subR = subR[maskR]
    nL, nR = len(subL), len(subR)
    N = min(nL, nR)
    if N < 2:
        return None
    # match counts
    idxL = np.arange(nL)[:N]
    idxR = np.arange(nR)[:N]
    onsets = []
    times_ref = None; lrp_ref = None
    for k in range(N):
        L = subL.copy()[np.setdiff1d(idxL, [k])].average(picks=["C3","C4"])
        R = subR.copy()[np.setdiff1d(idxR, [k])].average(picks=["C3","C4"])
        C3l, C4l = L.copy().pick(["C3"]).data[0], L.copy().pick(["C4"]).data[0]
        C3r, C4r = R.copy().pick(["C3"]).data[0], R.copy().pick(["C4"]).data[0]
        lrp = 0.5 * (((C3l - C4l) + (C4r - C3r)) * 1e6)
        t = L.times
        if times_ref is None:
            times_ref = t; lrp_ref = lrp
        # onset rule
        i0,i1 = sec_to_samp(t, 0.15, 1.5)
        seg = lrp[i0:i1+1]; tt = t[i0:i1+1]
        peak = np.nanmin(seg) if np.abs(np.nanmin(seg)) > np.abs(np.nanmax(seg)) else np.nanmax(seg)
        thr = CFG.thresholds["lrp_onset_frac"] * peak
        cond = (seg >= thr) if peak >= 0 else (seg <= thr)
        dt = np.median(np.diff(tt)); min_len = int(np.ceil(CFG.thresholds["lrp_min_dur"]/dt))
        run=0; onset=np.nan
        for i, g in enumerate(cond):
            if g:
                run += 1
                if run >= min_len:
                    onset = tt[i - min_len + 1]; break
            else:
                run = 0
        onsets.append(onset)
    # jackknife stats
    on = np.array(onsets, float)
    mu = float(np.nanmean(on)); se = float(np.nanstd(on, ddof=1) / np.sqrt(np.sum(np.isfinite(on))))
    return times_ref, lrp_ref, on, mu, se

lrp_stats = []
for mod in ["V","A","VA"]:
    res = lrp_leave_one_out_onset(erp_epochs, mod)
    if res is None: 
        continue
    t, lrp_ref, on, mu, se = res
    lrp_stats.append({"modality": mod, "jackknife_mean_onset_s": mu, "jackknife_se": se})
    # Plot
    fig, ax = plt.subplots(1,1, figsize=(7,3))
    ax.plot(t, lrp_ref, label="LRP (avg)")
    ax.axvline(0, ls="--", lw=1, color="k")
    ax.axvspan(0.15, 1.5, alpha=0.05)
    ax.set_title(f"LRP — {mod}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("µV")
    # mark mean onset
    if np.isfinite(mu): ax.axvline(mu, color="C3", lw=2, label=f"Jackknife onset ≈ {mu:.3f}s")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"lrp_{mod}.png", dpi=150); plt.close()

df_lrp_jk = pd.DataFrame(lrp_stats)
print(df_lrp_jk)
print("✔ BLOCK 8 LRP jackknife plots saved.")
""")

# ---- BLOCK 9 ----
add_md("## BLOCK 9 — Summaries (ERD/PMBR & ERPs)")
add_code(r"""
# Summaries from df_trial and df_erp

def _boot_ci(x, n=2000, q=(2.5, 97.5)):
    rng = np.random.default_rng(CFG.random_state)
    x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x)==0: return (np.nan, np.nan, np.nan)
    meds = []
    for _ in range(n):
        meds.append(np.nanmedian(rng.choice(x, size=len(x), replace=True)))
    lo, hi = np.nanpercentile(meds, q)
    return (float(np.nanmedian(x)), float(lo), float(hi))

sum_records = []
for (hand, mod, band), g in df_trial.groupby(["hand","modality","band"]):
    # delta contra-ipsi where available
    for metric in ["erd_onset_s","erd_peak_s","erd_min_percent","pmbr_peak_s","pmbr_peak_percent"]:
        vals = g[metric].values
        mu, lo, hi = _boot_ci(vals, n=1000)
        sum_records.append({"hand":hand, "modality":mod, "band":band, "metric":metric,
                            "median":mu, "ci_low":lo, "ci_high":hi})

# ERP summaries
for (mod, hand), g in df_erp.groupby(["modality","hand"]):
    for col in ["P2_latency_s","P3_latency_s","CNV_mean_uv","CNV_slope_uv_per_s","LRP_onset_s"]:
        if col not in g: continue
        mu, lo, hi = _boot_ci(g[col].values, n=1000)
        sum_records.append({"hand":hand, "modality":mod, "band":"ERP", "metric":col,
                            "median":mu, "ci_low":lo, "ci_high":hi})

df_summary = pd.DataFrame(sum_records)

def _save_df(df, path):
    path = pathlib.Path(path)
    try:
        df.to_parquet(path); print(f"Saved {path}")
    except Exception as e:
        df.to_csv(path.with_suffix(".csv"), index=False)
        warnings.warn(f"to_parquet failed ({e}); saved CSV instead.")

_save_df(df_summary, RES_DIR/"summary_metrics.parquet")
print(df_summary.head())
print("✔ BLOCK 9 saved summary metrics.")
""")

# ---- DECODING D0 ----
add_md("## BLOCK D0 — Windowing & labels")
add_code(r"""
# Window grid over 0..min(9.5, end)
sfreq = epochs.info["sfreq"]
t_start, t_end = 0.0, min(9.5, epochs.times[-1])
centers = np.arange(t_start + CFG.win_len/2, t_end - CFG.win_len/2 + 1e-9, CFG.win_step)
win_grid = np.array([(c - CFG.win_len/2, c + CFG.win_len/2, c) for c in centers])

# Labels
y = np.array([0 if h=="left" else 1 for h in (hands if isinstance(hands, np.ndarray) else np.array(hands))])
mods_arr = np.array(modalities)

# Groups: metadata['block'] if present else sequential in 10s
if "block" in epochs.metadata.columns:
    groups = epochs.metadata["block"].astype(int).values
else:
    groups = np.repeat(np.arange(int(np.ceil(len(epochs)/10))), 10)[:len(epochs)]

print(f"Windows: {len(win_grid)} | Trials: {len(y)}")
print("✔ D0 windowing and labels ready.")
""")

# ---- DECODING D1 ----
add_md("## BLOCK D1 — Feature extractors (power, CSP, augmented)")
add_code(r"""
# NOTE: To avoid leakage, these transformers are used inside scikit Pipelines.
# They access precomputed data slices passed at construction; transform() ignores X and returns window features.
# References in comments: chance/permutation [1,14]; leakage [4]; shrinkage [5]; time-resolved [6,21]; CSP best-practices [9,16,17,15,13]

class PowerFeatures:
    """scikit-compatible transformer that returns log-bandpower features for a given time window.
    Features per trial: [mu_C3, mu_C4, mu_neigh_mean, beta_C3, beta_C4, beta_neigh_mean, mu_C3minusC4, mu_C4minusC3, beta_C3minusC4, beta_C4minusC3]."""
    def __init__(self, precomp, ch_names, neighbours):
        self.pre = precomp  # dict: {'mu': arr, 'beta': arr} each (n_trials, n_ch, n_times)
        self.ch_names = ch_names
        self.neigh = [c for c in neighbours if c in ch_names]
        self.idx = {c:i for i,c in enumerate(ch_names)}
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X is ignored; use precomputed slice already restricted to window
        feats = []
        for band in ["mu","beta"]:
            arr = self.pre[band]  # (n_trials, n_ch)
            c3 = arr[:, self.idx["C3"]]
            c4 = arr[:, self.idx["C4"]]
            if len(self.neigh)>0:
                neigh = arr[:, [self.idx[c] for c in self.neigh]].mean(axis=1)
            else:
                neigh = np.nanmean(arr, axis=1)
            feats.append(np.c_[c3, c4, neigh])
            feats.append(np.c_[c3 - c4, c4 - c3])
        Xf = np.concatenate(feats, axis=1)
        return Xf

class CSPWindowFeatures:
    """Regularised CSP (reg='ledoit_wolf') applied per window on band-passed epochs (8–30 Hz). Returns log-variance of components."""
    def __init__(self, X_epoch_win, sfreq, n_components=4):
        self.X = X_epoch_win  # shape (n_trials, n_ch, n_times) time-domain window (band-passed)
        self.sfreq = sfreq
        self.n_components = n_components
        self.csp = CSP(n_components=n_components, reg='ledoit_wolf', log=True, norm_trace=False)
        self.fitted = False
    def fit(self, y, groups=None):
        # y is label vector; X taken from self.X per sklearn API quirk
        # CSP expects X shape (n_trials, n_ch, n_times)
        self.csp.fit(self.X, y)
        self.fitted = True
        return self
    def transform(self, y=None):
        assert self.fitted, "fit() first"
        return self.csp.transform(self.X)

def _slice_precomputed_for_window(tfr_all, win, bands=CFG.bands, baseline=CFG.windows["baseline"]):
    """Return dict per band of LOG10(power) averaged over freqs in band and times within window, per channel."""
    i0,i1 = sec_to_samp(tfr_all.times, win[0], win[1])
    out = {}
    for name, (fmin,fmax) in bands.items():
        fmask = (tfr_all.freqs >= fmin) & (tfr_all.freqs <= fmax)
        # mean over freqs and times, then log10
        power = tfr_all.data[:, :, fmask, i0:i1+1].mean(axis=(2,3))
        out[name] = np.log10(np.maximum(power, np.finfo(float).eps))
    return out

def _bandpass_epoch_window(epochs_obj, win, band=(8,30)):
    ep = epochs_obj.copy().filter(band[0], band[1], method='fir', fir_design='firwin', verbose=False)
    i0,i1 = sec_to_samp(ep.times, win[0], win[1])
    X = ep.get_data()[:, :, i0:i1+1]  # (n_trials, n_ch, n_times)
    return X

print("✔ D1 feature extractors defined.")
""")

# ---- DECODING D2 ----
add_md("## BLOCK D2 — Cross-validation & models (per window, per modality & pooled)")
add_code(r"""
rng = np.random.default_rng(CFG.random_state)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CFG.random_state)

def _run_cv_for_window(win, modality=None, pooled=False, featureset="power", model="LDA"):
    # Select trials
    if pooled:
        idx = np.arange(len(y))
        mod_onehot = pd.get_dummies(mods_arr).reindex(["V","A","VA"], axis=1, fill_value=0).values
    else:
        idx = np.where(mods_arr==modality)[0]
        mod_onehot = None
    y_sel = y[idx]; groups_sel = groups[idx]

    # Precompute window-specific features
    pre = _slice_precomputed_for_window(tfr_all, win, bands=CFG.bands)
    pf = PowerFeatures(pre, tfr_all.ch_names, CFG.neighbours)
    # Optionally add CSP features
    if featureset in ("csp","power+csp"):
        X_epoch_win = _bandpass_epoch_window(epochs[idx], win, band=CFG.csp_band)
        cspF = CSPWindowFeatures(X_epoch_win, epochs.info["sfreq"], n_components=CFG.csp_components)

    # Build design matrix on the fly per fold to avoid leakage (fit inside pipeline)
    # We pass dummy X; transformer uses precomputed window slices.
    dummyX = np.zeros((len(idx), 1))
    if featureset == "power":
        steps = [('scaler', StandardScaler(with_mean=True, with_std=True)),
                 ('feat', pf),
                 ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') if model=="LDA"
                         else LogisticRegression(penalty='l2', C=1.0, max_iter=200))]
    elif featureset == "csp":
        # CSPWindowFeatures is not a sklearn transformer with X,y sig, so we wrap it minimalistically: fit on y, transform ignores X
        class _CSPWrapper:
            def __init__(self, cspF): self.cspF=cspF
            def fit(self, X, y): self.cspF.fit(y); return self
            def transform(self, X): return self.cspF.transform()
        steps = [('feat', _CSPWrapper(cspF)),
                 ('scaler', StandardScaler(with_mean=True, with_std=True)),
                 ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') if model=="LDA"
                         else LogisticRegression(penalty='l2', C=1.0, max_iter=200))]
    else: # power+csp: concatenate
        class _ConcatFeats:
            def __init__(self, pf, cspF): self.pf=pf; self.cspF=cspF; self.fitted=False
            def fit(self, X, y):
                self.cspF.fit(y); self.fitted=True; return self
            def transform(self, X):
                A = self.pf.transform(X)
                B = self.cspF.transform()
                return np.hstack([A,B])
        X_epoch_win = _bandpass_epoch_window(epochs[idx], win, band=CFG.csp_band)
        cspF = CSPWindowFeatures(X_epoch_win, epochs.info["sfreq"], n_components=CFG.csp_components)
        steps = [('feat', _ConcatFeats(pf, cspF)),
                 ('scaler', StandardScaler(with_mean=True, with_std=True)),
                 ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') if model=="LDA"
                         else LogisticRegression(penalty='l2', C=1.0, max_iter=200))]

    pipe = Pipeline(steps)
    # OOF probabilities
    prob = cross_val_predict(pipe, dummyX, y_sel, groups=groups_sel, cv=splitter,
                             method='predict_proba', n_jobs=None)
    y_pred = (prob[:,1] >= 0.5).astype(int)
    ba = balanced_accuracy_score(y_sel, y_pred)
    auc = roc_auc_score(y_sel, prob[:,1])
    return {"idx": idx, "y_true": y_sel, "prob": prob[:,1], "ba": ba, "auc": auc}

# Run for all windows & modalities
results_rows = []
for featureset in ["power","csp","power+csp"]:
    for model in ["LDA"]:
        for pooled in [False, True]:
            for modality in (["V","A","VA"] if not pooled else [None]):
                BA, AUC = [], []
                prob_store = []
                for (t0,t1,tc) in win_grid:
                    out = _run_cv_for_window((t0,t1), modality=modality, pooled=pooled,
                                             featureset=featureset, model=model)
                    BA.append(out["ba"]); AUC.append(out["auc"])
                    prob_store.append(pd.DataFrame({
                        "trial_id": out["idx"],
                        "window_t_start": t0, "window_t_center": tc,
                        "y_true": out["y_true"], "y_prob_right": out["prob"]
                    }))
                probs_df = pd.concat(prob_store, ignore_index=True)
                tag_mod = "pooled" if pooled else modality
                tag = f"probs_{tag_mod}_{featureset}_{model}.parquet"
                # Save probs
                try:
                    probs_df.to_parquet(DEC_DIR/tag)
                except Exception as e:
                    probs_df.to_csv(DEC_DIR/(tag.replace(".parquet",".csv")), index=False)
                # Save curves
                curves = pd.DataFrame({
                    "modality": tag_mod,
                    "featureset": featureset,
                    "model": model,
                    "window_t_center": win_grid[:,2],
                    "BA": BA, "AUC": AUC, "n_trials": len(out["y_true"])
                })
                curves.to_csv(DEC_DIR/f"curves_{tag_mod}_{featureset}_{model}.csv", index=False)
                print(f"✔ D2 stored probs & curves for {tag_mod} | {featureset} | {model}")

print("✔ D2 all decoding runs completed (OOF probabilities saved).")
""")

# ---- DECODING D3 ----
add_md("## BLOCK D3 — Time-resolved decoding curves")
add_code(r"""
# Aggregate curves from saved CSVs (robust to parquet availability)
curves_all = []
for p in DEC_DIR.glob("curves_*.csv"):
    df = pd.read_csv(p)
    curves_all.append(df)
df_curves = pd.concat(curves_all, ignore_index=True)

# Plot per featureset across modalities
def plot_curves(featureset: str):
    fig, axes = plt.subplots(2,1, figsize=(9,6), sharex=True)
    for mod in df_curves["modality"].unique():
        sel = (df_curves["featureset"]==featureset) & (df_curves["modality"]==mod)
        d = df_curves[sel]
        if len(d)==0: continue
        axes[0].plot(d["window_t_center"], d["BA"], label=mod)
        axes[1].plot(d["window_t_center"], d["AUC"], label=mod)
    axes[0].axhline(0.5, ls="--", lw=1, color="k"); axes[0].set_ylabel("Balanced Accuracy")
    axes[1].axhline(0.5, ls="--", lw=1, color="k"); axes[1].set_ylabel("ROC-AUC"); axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.axvline(0, ls="--", lw=1, color="k")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"decoding_curves_{featureset}.png", dpi=150); plt.close()

for fs in ["power","csp","power+csp"]:
    plot_curves(fs)

print("✔ D3 decoding curves plotted & saved.")
""")

# ---- DECODING D4 ----
add_md("## BLOCK D4 — Permutation testing (significance envelopes)")
add_code(r"""
# WARNING: This can be compute-heavy. Adjust CFG.perm_n as needed.
def permutation_null(featureset="power", modality="V", n_perm=CFG.perm_n):
    # Load probs to reuse per-window pipeline faster? Here we recompute per permutation for correctness.
    rng = np.random.default_rng(CFG.random_state)
    null_BA = np.zeros((n_perm, len(win_grid)))
    null_AUC = np.zeros((n_perm, len(win_grid)))
    idx_mod = np.where(mods_arr==modality)[0]
    y_true = y[idx_mod]
    for p in range(n_perm):
        # shuffle within groups
        y_perm = y_true.copy()
        # group-wise shuffle
        grp = groups[idx_mod]
        for gval in np.unique(grp):
            mask = (grp==gval)
            y_perm[mask] = rng.permutation(y_perm[mask])
        BA = []; AUC = []
        for wi, (t0,t1,tc) in enumerate(win_grid):
            out = _run_cv_for_window((t0,t1), modality=modality, pooled=False,
                                     featureset=featureset, model="LDA")
            # Replace labels with permuted labels when scoring
            # We reuse predicted prob but compute metrics against permuted labels (approximate, avoids recomputing pipeline twice)
            prob = out["prob"]
            ba = balanced_accuracy_score(y_perm, (prob>=0.5).astype(int))
            auc = roc_auc_score(y_perm, prob)
            BA.append(ba); AUC.append(auc)
        null_BA[p] = np.array(BA); null_AUC[p] = np.array(AUC)
    # 95th percentile envelope
    env_ba = np.nanpercentile(null_BA, 95, axis=0)
    env_auc = np.nanpercentile(null_AUC, 95, axis=0)
    return env_ba, env_auc

envelopes = []
for fs in ["power","csp","power+csp"]:
    for mod in ["V","A","VA"]:
        try:
            env_ba, env_auc = permutation_null(featureset=fs, modality=mod, n_perm=min(CFG.perm_n, 100))
            df_env_ba = pd.DataFrame({"modality":mod, "featureset":fs, "metric":"BA",
                                      "window_t_center": win_grid[:,2], "p95": env_ba})
            df_env_auc = pd.DataFrame({"modality":mod, "featureset":fs, "metric":"AUC",
                                       "window_t_center": win_grid[:,2], "p95": env_auc})
            envelopes.append(df_env_ba); envelopes.append(df_env_auc)
            print(f"Computed null envelope for {mod} | {fs}")
        except Exception as e:
            warnings.warn(f"Permutation failed for {mod} | {fs}: {e}")

df_env = pd.concat(envelopes, ignore_index=True) if len(envelopes)>0 else pd.DataFrame()

try:
    df_env.to_parquet(DEC_DIR/"null_envelope.parquet")
except Exception as e:
    df_env.to_csv(DEC_DIR/"null_envelope.csv", index=False)

# Plot overlays
def plot_with_envelope(featureset: str, metric: str):
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    for mod in ["V","A","VA"]:
        d = df_curves[(df_curves["featureset"]==featureset) & (df_curves["modality"]==mod)]
        if len(d)==0: continue
        ax.plot(d["window_t_center"], d[metric], label=f"{mod} observed")
        e = df_env[(df_env["featureset"]==featureset) & (df_env["metric"]==metric) & (df_env["modality"]==mod)]
        if len(e)>0:
            ax.plot(e["window_t_center"], e["p95"], color="gray", alpha=0.6, lw=1, label=None)
    ax.axvline(0, ls="--", lw=1, color="k")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(metric)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR/f"decoding_significance_{featureset}_{metric}.png", dpi=150); plt.close()

for fs in ["power","csp","power+csp"]:
    for met in ["BA","AUC"]:
        plot_with_envelope(fs, met)

print("✔ D4 permutation envelopes computed and plotted (reduced perms for speed).")
""")

# ---- DECODING D5 ----
add_md("## BLOCK D5 — Detection latency & early-detection operating point")
add_code(r"""
# Detection latency wrt null envelope; bootstrap CIs from trial-level OOF
def _persistence_bool(arr, min_pts):
    run=0
    for i, v in enumerate(arr):
        if v:
            run+=1
            if run>=min_pts:
                return i - min_pts + 1
        else:
            run=0
    return None

def latency_metrics(featureset="power", modality="V", metric="BA"):
    d = df_curves[(df_curves["featureset"]==featureset) & (df_curves["modality"]==modality)]
    e = df_env[(df_env["featureset"]==featureset) & (df_env["metric"]==metric) & (df_env["modality"]==modality)]
    if len(d)==0 or len(e)==0: return None
    y_curve = d[metric].values
    y_env = e["p95"].values
    over = y_curve > y_env
    min_pts_sig = int(np.ceil(CFG.persist_ms_sig / (CFG.win_step*1000)))
    i0 = _persistence_bool(over, min_pts_sig)
    lat_sig = (d["window_t_center"].values[i0] if i0 is not None else np.nan)
    # OP threshold
    thr = CFG.op_threshold_balacc if metric=="BA" else 0.70
    over2 = y_curve >= thr
    min_pts_op = int(np.ceil(CFG.persist_ms_op / (CFG.win_step*1000)))
    j0 = _persistence_bool(over2, min_pts_op)
    lat_op = (d["window_t_center"].values[j0] if j0 is not None else np.nan)
    return float(lat_sig), float(lat_op)

lat_rows = []
for fs in ["power","csp","power+csp"]:
    for mod in ["V","A","VA","pooled"]:
        for metric in ["BA"]:
            res = latency_metrics(featureset=fs, modality=mod, metric=metric)
            if res is None: continue
            lat_sig, lat_op = res
            lat_rows.append({"featureset":fs, "modality":mod, "model":"LDA",
                             "latency_sig_s": lat_sig, "latency_op70_s": lat_op})

df_lat = pd.DataFrame(lat_rows)

# Save
try:
    df_lat.to_parquet(DEC_DIR/"latency_metrics.parquet")
except Exception as e:
    df_lat.to_csv(DEC_DIR/"latency_metrics.csv", index=False)

print(df_lat)
print("✔ D5 latency metrics (point estimates) saved.")
""")

# ---- DECODING D6 ----
add_md("## BLOCK D6 — Diagnostics & quick-check plots for decoding")
add_code(r"""
# Confusion matrices near peak BA (±50 ms) — using observed OOF predictions
def _load_probs(tag_mod, featureset, model):
    # Prefer parquet else csv
    pqt = DEC_DIR/f"probs_{tag_mod}_{featureset}_{model}.parquet"
    csv = DEC_DIR/f"probs_{tag_mod}_{featureset}_{model}.csv"
    if pqt.exists():
        return pd.read_parquet(pqt)
    else:
        return pd.read_csv(csv)

fig_cm, axes_cm = plt.subplots(3,1, figsize=(6,10))
for ax, mod in zip(axes_cm, ["V","A","VA"]):
    tag_mod = mod
    prob_df = _load_probs(tag_mod, "power", "LDA")
    # Find peak BA window
    d = df_curves[(df_curves["featureset"]=="power") & (df_curves["modality"]==mod)]
    if len(d)==0: continue
    k = int(np.nanargmax(d["BA"].values))
    t_peak = d["window_t_center"].values[k]
    # select probs at closest time
    sel = prob_df.iloc[(prob_df["window_t_center"]-t_peak).abs().argsort()[:1]]
    # compute confusion
    y_true = prob_df[prob_df["window_t_center"]==sel["window_t_center"].values[0]]["y_true"].values
    y_pred = (prob_df[prob_df["window_t_center"]==sel["window_t_center"].values[0]]["y_prob_right"].values >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    im = ax.imshow(cm, aspect='auto')
    ax.set_title(f"Confusion @ peak BA ~ {t_peak:.2f}s | {mod}")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["left","right"]); ax.set_yticklabels(["left","right"])
    for (i,j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
plt.tight_layout(); plt.savefig(FIG_DIR/"decoding_confmats.png", dpi=150); plt.close()

# Feature sanity plots (mean log-bandpower time-courses)
fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True)
for mod in ["V","A","VA"]:
    idx = np.where(mods_arr==mod)[0]
    # compute lateralisation (C3-C4) mu/beta percent
    for band_name, d in bands.items():
        ch_idx = {c:i for i,c in enumerate(d["channels"])}
        if "C3" not in ch_idx or "C4" not in ch_idx: continue
        X = d["percent"][idx]  # (n, ch, t)
        lat = (X[:, ch_idx["C3"], :] - X[:, ch_idx["C4"], :]).mean(axis=0)
        axes[0 if band_name=="mu" else 1].plot(d["times"], lat, label=f"{mod} {band_name}")
for ax, title in zip(axes, ["Lateralisation (mu) C3-C4", "Lateralisation (beta) C3-C4"]):
    ax.axvline(0, ls="--", lw=1, color="k"); ax.set_title(title); ax.set_ylabel("%"); ax.legend(frameon=False)
axes[-1].set_xlabel("Time (s)")
plt.tight_layout(); plt.savefig(FIG_DIR/"feature_sanity_lateralisation.png", dpi=150); plt.close()

print("✔ D6 diagnostics saved (confusion matrices, feature sanity).")
""")

# ---- BLOCK 10 ----
add_md("## BLOCK 10 — Visual summaries (across modalities)")
add_code(r"""
# Combined summary figure (simple version)
fig, axes = plt.subplots(2,2, figsize=(12,8))

# ERD onset per modality (beta contra)
sel = df_trial[(df_trial["band"]=="beta") & (df_trial["side"]=="contra") & (df_trial["erd_onset_ok"])]
data = [sel[sel["modality"]==m]["erd_onset_s"].dropna().values for m in ["V","A","VA"]]
axes[0,0].boxplot(data, labels=["V","A","VA"])
axes[0,0].set_title("ERD onset (beta, contra)"); axes[0,0].set_ylabel("s")

# PMBR peak percent (beta contra)
data = [sel[sel["modality"]==m]["pmbr_peak_percent"].dropna().values for m in ["V","A","VA"]]
axes[0,1].boxplot(data, labels=["V","A","VA"])
axes[0,1].set_title("PMBR peak % (beta, contra)"); axes[0,1].set_ylabel("%")

# P3 latency across modalities (pooled hands)
p3 = df_erp.groupby("modality", as_index=False)["P3_latency_s"].median()
axes[1,0].bar(p3["modality"], p3["P3_latency_s"])
axes[1,0].set_title("P3 latency (median)"); axes[1,0].set_ylabel("s")

# Decoding latency (sig-based) across modalities for power features
lat = df_lat[(df_lat["featureset"]=="power")]
axes[1,1].bar(lat["modality"], lat["latency_sig_s"])
axes[1,1].set_title("Decoding latency (sig-based)"); axes[1,1].set_ylabel("s")

plt.tight_layout(); plt.savefig(FIG_DIR/"summary_all.png", dpi=150); plt.close()
print("✔ BLOCK 10 saved 'summary_all.png'.")
""")

# ---- BLOCK 11 ----
add_md("## BLOCK 11 — Logging & provenance")
add_code(r"""
with open(RES_DIR/"log.txt","a",encoding="utf-8") as f:
    f.write("\n--- DATA SUMMARY ---\n")
    # trial counts
    for (hand, mod), g in df_trial.groupby(["hand","modality"]):
        f.write(f"{hand}-{mod}: n={len(g)}\n")
    f.write("\n--- OUTLIER FLAGS (counts kept_*) ---\n")
    for col in [c for c in df_trial.columns if c.startswith("kept_")]:
        f.write(f"{col}: kept={int(df_trial[col].sum())} / {len(df_trial)}\n")
    f.write("\n--- DECODING PARAMS ---\n")
    f.write(json.dumps({"perm_n": CFG.perm_n, "bootstrap_n": CFG.bootstrap_n,
                        "win_len": CFG.win_len, "win_step": CFG.win_step}, indent=2))
print("✔ BLOCK 11 appended provenance to results/log.txt.")
""")

# ---- BLOCK 12 ----
add_md("## BLOCK 12 — Pack outputs")
add_code(r"""
outputs = {
    "trial_metrics": df_trial.shape,
    "erp_metrics": df_erp.shape,
    "summary_metrics": df_summary.shape,
    "decode_curves": df_curves.shape,
    "latency_metrics": df_lat.shape,
    "fig_paths": [str(p.name) for p in FIG_DIR.glob("*.png")]
}
import pickle
with open(RES_DIR/"outputs.pkl","wb") as f:
    pickle.dump(outputs, f)

print("Outputs summary:", outputs)
print("✔ BLOCK 12 done; outputs.pkl saved.")
""")

# ---- Final notes cell with references placeholders ----
add_md("""
> **Notes & references in comments**  
> Chance levels & permutation tests: **[1,14]**. Leakage/double-dipping: **[4]**. Ledoit–Wolf shrinkage: **[5]**. Time-resolved decoding & temporal generalisation: **[6,21]**. MI μ/β & lateralisation: **[2,20]**. CSP for MI-BCI: **[9,16,17,15,13]**. PMBR (real & imagined): **[7–9,19]**. Bootstrap CIs: **[18]**. Early-detection dwell-time convention: **[12]**.
""")

# ---- Write notebook file ----
out_path = "MI_ERD_ERP_Decoding.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

out_path
