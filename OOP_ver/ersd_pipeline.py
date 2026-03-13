# -*- coding: utf-8 -*-
"""
ERS/ERD unified pipeline (methods: Hilbert, Morlet, Multitaper, STFT/Welch)
- Compatible with the user's environment (MNE 1.10.0; numpy>=2; scipy>=1.15; statsmodels 0.14.5).
- Uses mne.time_frequency.compute_tfr (no legacy tfr_multitaper).
- Produces:
    * Figures: mean±SD time-courses for C3/C4 (both % and dB), with baseline shading and onset markers
    * Figures: ERD/ERS topomaps (unsmoothed by default; toggle-able temporal smoothing)
    * CSVs:
        - onsets_per_trial.csv (per trial onsets for detectors that support per-trial)
        - onsets_group.csv     (mean + SD across trials; FDR group onset + bootstrap SD)
- Does NOT save any time-series CSVs.
"""
import os
import json
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import mne

from scipy.signal import stft, get_window
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_1samp

from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt


# ==============================
# ---- Utility / formatting ----
# ==============================

def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs >= lo) & (freqs <= hi)


def _gaussian_sigma_samples(fwhm_sec: float, sfreq: float) -> float:
    if fwhm_sec is None or fwhm_sec <= 0:
        return 0.0
    # fwhm = 2*sqrt(2*ln2)*sigma  -> sigma = fwhm / 2.3548
    return (fwhm_sec / 2.354820045) * sfreq


def _baseline_indices(times: np.ndarray, baseline: Tuple[float, float]) -> np.ndarray:
    b0, b1 = baseline
    return (times >= b0) & (times <= b1)


def _baseline_norm_percent_db(power: np.ndarray, times: np.ndarray,
                              baseline: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Baseline-normalize a per-trial power array.
    Parameters
    ----------
    power : array, shape (n_trials, n_channels, n_times)  -- power (>=0)
    times : array, shape (n_times,)
    baseline : (t0, t1)

    Returns
    -------
    percent : same shape, % change vs baseline
    db      : same shape, dB change vs baseline (10*log10(P / Pbase))
    base_mean : array, shape (n_trials, n_channels, 1) -- baseline mean
    base_std  : array, shape (n_trials, n_channels, 1) -- baseline std (for z-scores)
    """
    bmask = _baseline_indices(times, baseline)
    # Avoid division by zero
    base = np.clip(power[..., bmask].mean(axis=-1, keepdims=True), 1e-12, np.inf)
    base_std = power[..., bmask].std(axis=-1, keepdims=True) + 1e-12

    percent = (power - base) / base * 100.0
    db = 10.0 * np.log10(np.maximum(power, 1e-24) / base)
    return percent, db, base, base_std


def _apply_temporal_smoothing(data: np.ndarray, sfreq: float, fwhm_sec: Optional[float]) -> np.ndarray:
    """
    Apply Gaussian smoothing along time-axis of a (trials, channels, times) array.
    If fwhm_sec is None or <= 0, returns data unchanged.
    """
    if fwhm_sec is None or fwhm_sec <= 0:
        return data
    sigma = _gaussian_sigma_samples(fwhm_sec, sfreq)
    if sigma <= 0:
        return data
    # Gaussian along the last axis (time)
    return gaussian_filter1d(data, sigma=sigma, axis=-1, mode="nearest")


def _find_sustained_runs(bool_vec: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """
    Find all runs of True in a boolean vector with length >= min_len.
    Returns list of (start_idx, end_idx_exclusive).
    """
    runs = []
    n = len(bool_vec)
    i = 0
    while i < n:
        if bool_vec[i]:
            j = i
            while j < n and bool_vec[j]:
                j += 1
            if (j - i) >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# ==================================
# ---- ERS/ERD computation paths ----
# ==================================

def compute_bandpower_hilbert(epochs: mne.Epochs,
                              band: Tuple[float, float],
                              picks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter + Hilbert envelope -> power per trial/channel/time.

    Returns
    -------
    P : array (n_trials, n_channels, n_times) power
    times : array (n_times,)
    """
    lo, hi = band
    ep = epochs.copy()
    if picks is not None:
        ep.pick(picks)
    ep.filter(lo, hi, method='fir', fir_design='firwin', phase='zero-double', verbose='ERROR')
    ep.apply_hilbert(envelope=True, n_jobs=None, verbose='ERROR')
    A = ep.get_data()  # trials x ch x t, amplitude
    P = A ** 2
    return P, ep.times


def compute_bandpower_tfr(epochs: mne.Epochs,
                          band: Tuple[float, float],
                          method: str,
                          freqs: np.ndarray,
                          n_cycles,
                          time_bandwidth: Optional[float] = None,
                          picks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Morlet/Multitaper TFR -> band-averaged power per trial/channel/time.

    method: 'morlet' or 'multitaper' (mne.Epochs.compute_tfr)
    """
    ep = epochs.copy()
    if picks is not None:
        ep.pick(picks)

    kwargs = dict(method=method,
                  freqs=freqs,
                  n_cycles=n_cycles,
                  use_fft=True,
                  return_itc=False,
                  average=False,
                  decim=1,
                  n_jobs=None,
                  verbose='ERROR')
    if method == 'multitaper' and time_bandwidth is not None:
        kwargs['time_bandwidth'] = time_bandwidth

    tfr = ep.compute_tfr(**kwargs)  # EpochsTFR, shape: (n_trials, n_channels, n_freqs, n_times)
    data = tfr.data  # power
    fmask = _band_mask(tfr.freqs, band)
    band_power = data[:, :, fmask, :].mean(axis=2)  # avg over selected freqs
    return band_power, tfr.times


def compute_bandpower_stft(epochs: mne.Epochs,
                           band: Tuple[float, float],
                           win_len_sec: float = 0.5,
                           overlap: float = 0.5,
                           picks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding STFT (Welch-like) -> band power per trial/channel/time interpolated to epochs.times.
    """
    ep = epochs.copy()
    if picks is not None:
        ep.pick(picks)
    X = ep.get_data()  # trials x channels x times
    sf = ep.info['sfreq']
    n_trials, n_channels, n_times = X.shape
    win_len = int(round(win_len_sec * sf))
    n_overlap = int(round(overlap * win_len))
    window = get_window('hann', win_len, fftbins=True)

    # STFT per trial/channel
    f_lo, f_hi = band
    all_power = np.zeros_like(X, dtype=float)  # will hold band power resampled to epoch times
    times_target = ep.times

    for tr in range(n_trials):
        for ch in range(n_channels):
            f, t, Z = stft(X[tr, ch, :], fs=sf, window=window, nperseg=win_len,
                           noverlap=n_overlap, nfft=None, detrend=False, boundary=None, padded=False)
            P = np.abs(Z) ** 2  # spectrogram power
            fmask = (f >= f_lo) & (f <= f_hi)
            if fmask.any():
                band_p = P[fmask, :].mean(axis=0)
            else:
                band_p = np.zeros_like(t)
            # STFT 't' is in seconds, relative to the signal start (epochs.times[0])
            # Map to epochs time axis via interpolation
            t_abs = t + times_target[0]
            all_power[tr, ch, :] = np.interp(times_target, t_abs, band_p, left=band_p[0], right=band_p[-1])
    return all_power, times_target


# ====================================
# ---- Onset detection algorithms ----
# ====================================

def _detector_thresh_sd(z: np.ndarray, times: np.ndarray, k: float,
                        min_dur: float, search_window: Tuple[float, float],
                        direction: str) -> float:
    """Return onset time or np.nan. z: (n_times,) z-score vs baseline."""
    t0, t1 = search_window
    mask = (times >= t0) & (times <= t1)
    thr = k if direction == 'ERS' else -k
    above = z >= thr if direction == 'ERS' else z <= thr
    # apply window mask
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.nan
    sub = above[idx]
    # sustain
    min_len = max(1, int(round(min_dur / (times[1] - times[0]))))
    runs = _find_sustained_runs(sub, min_len)
    if not runs:
        return np.nan
    onset_idx = idx[runs[0][0]]
    return float(times[onset_idx])


def _detector_changepoint_cusum(z: np.ndarray, times: np.ndarray,
                                h: float, min_dur: float,
                                search_window: Tuple[float, float],
                                direction: str) -> float:
    """
    Simple CUSUM-like detector on z-score.
    Accumulate deviations; trigger when cumulative sum crosses ±h, then require sustain.
    """
    t0, t1 = search_window
    mask = (times >= t0) & (times <= t1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.nan
    zsub = z[idx]
    # One-sided cusum depending on direction
    if direction == 'ERS':
        s = np.maximum.accumulate(np.cumsum(zsub))  # trend to positive
        trig = s >= h
    else:
        s = np.maximum.accumulate(np.cumsum(-zsub))  # trend to negative
        trig = s >= h
    min_len = max(1, int(round(min_dur / (times[1] - times[0]))))
    runs = _find_sustained_runs(trig, min_len)
    if not runs:
        return np.nan
    onset_idx = idx[runs[0][0]]
    return float(times[onset_idx])


def _detector_halfpeak(percent: np.ndarray, times: np.ndarray,
                       search_window: Tuple[float, float],
                       direction: str) -> float:
    """
    50%-of-peak latency within a search window on baseline-normalized % signal.
    ERD: find trough; ERS: find peak. Onset = first crossing of 50% of that magnitude.
    """
    t0, t1 = search_window
    mask = (times >= t0) & (times <= t1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.nan
    sub = percent[idx]
    if direction == 'ERS':
        peak_idx = int(np.nanargmax(sub))
        peak_val = sub[peak_idx]
        if not np.isfinite(peak_val) or peak_val <= 0:
            return np.nan
        thr = 0.5 * peak_val
        # earliest crossing
        cross = np.where(sub >= thr)[0]
    else:
        peak_idx = int(np.nanargmin(sub))
        peak_val = sub[peak_idx]
        if not np.isfinite(peak_val) or peak_val >= 0:
            return np.nan
        thr = 0.5 * peak_val  # negative number
        cross = np.where(sub <= thr)[0]
    if cross.size == 0:
        return np.nan
    onset_idx = idx[int(cross[0])]
    return float(times[onset_idx])


def group_detector_fdr(percent_trials: np.ndarray,  # shape (n_trials, n_times)
                       times: np.ndarray,
                       alpha: float,
                       min_dur: float,
                       search_window: Tuple[float, float],
                       n_boot: int = 200,
                       random_state: int = 7) -> Tuple[float, float]:
    """
    Across-trial t-test vs 0 at each time, FDR-corrected.
    Returns (onset_time, bootstrap_sd_seconds).
    If no onset found, returns (nan, nan).
    """
    rng = np.random.default_rng(random_state)
    t0, t1 = search_window
    mask = (times >= t0) & (times <= t1)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.nan, np.nan
    sub = percent_trials[:, idx]
    # one-sample t-test vs 0 (% baseline)
    tvals, pvals = ttest_1samp(sub, popmean=0.0, axis=0, nan_policy='omit')
    rej, p_corr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    min_len = max(1, int(round(min_dur / (times[1] - times[0]))))
    runs = _find_sustained_runs(rej, min_len)
    if not runs:
        return np.nan, np.nan
    onset_idx = idx[runs[0][0]]
    onset_time = float(times[onset_idx])

    # Bootstrap SD over trial resampling
    if n_boot and n_boot > 0 and sub.shape[0] > 3:
        boots = []
        for _ in range(n_boot):
            bs = sub[rng.integers(0, sub.shape[0], size=sub.shape[0]), :]
            tvals_b, pvals_b = ttest_1samp(bs, popmean=0.0, axis=0, nan_policy='omit')
            rej_b, _, _, _ = multipletests(pvals_b, alpha=alpha, method='fdr_bh')
            runs_b = _find_sustained_runs(rej_b, min_len)
            if runs_b:
                boots.append(times[idx[runs_b[0][0]]])
        if len(boots) >= 3:
            sd = float(np.nanstd(np.array(boots), ddof=1))
        else:
            sd = np.nan
    else:
        sd = np.nan
    return onset_time, sd


# =====================================
# ---- Plotting (time + topomaps)  ----
# =====================================

def _shade_baseline(ax, baseline: Tuple[float, float], ymin: float, ymax: float, color=(0.85, 0.85, 0.85)):
    ax.axvspan(baseline[0], baseline[1], color=color, alpha=0.5, zorder=0)
    ax.set_ylim(ymin, ymax)


def plot_timeseries_mean_sd(times: np.ndarray,
                            mean_: np.ndarray,  # shape (n_channels, n_times)
                            sd_: np.ndarray,    # same shape
                            channels: List[str],
                            title: str,
                            unit_label: str,
                            baseline: Tuple[float, float],
                            onset_lines: Optional[Dict[str, float]] = None,
                            outpath: Optional[str] = None):
    """
    Draw mean ± SD for selected channels (e.g., ['C3','C4']) in two subplots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, ch in enumerate(channels):
        ax = axes[i]
        y = mean_[i, :]
        s = sd_[i, :]
        ax.plot(times, y, lw=2, label=ch)
        ax.fill_between(times, y - s, y + s, alpha=0.25)
        # auto y-lims from data
        ymin = float(np.nanmin(y - s)) * 1.1 if np.isfinite(np.nanmin(y - s)) else -1
        ymax = float(np.nanmax(y + s)) * 1.1 if np.isfinite(np.nanmax(y + s)) else 1
        if ymin == ymax:
            ymin, ymax = ymin - 1, ymax + 1
        _shade_baseline(ax, baseline, ymin, ymax)
        if onset_lines:
            for label, t in onset_lines.items():
                if np.isfinite(t):
                    ax.axvline(t, ls='--', lw=1.5, label=label)
        ax.set_ylabel(unit_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_topomap_percent(epochs: mne.Epochs,
                         percent_trials: np.ndarray,  # (n_trials, n_channels, n_times) in EEG-picks order
                         times: np.ndarray,
                         topo_win: Tuple[float, float],
                         picks_eeg: np.ndarray,
                         title: str,
                         outpath: Optional[str] = None):
    """
    Plot average % change within a time window as a topomap using MNE's EvokedArray.
    Compatible with MNE 1.10: uses vlim=... (no vmin/vmax); units/scalings as dicts.
    NOTE: percent_trials is already restricted to EEG channels in the same order as picks_eeg.
    """
    mask = (times >= topo_win[0]) & (times <= topo_win[1])
    if not np.any(mask):
        return

    # average across trials, then across the time window
    # percent_trials shape: trials x ch(=len(picks_eeg)) x time
    data_ch_t = np.nanmean(percent_trials, axis=0)                    # ch x time
    data_win = np.nanmean(data_ch_t[:, mask], axis=1, keepdims=True)  # ch x 1

    # Build info for EEG channels only (same order as percent_trials)
    info_eeg = mne.pick_info(epochs.info, sel=picks_eeg)

    # IMPORTANT: do NOT re-index with picks_eeg again here; data_win already matches info_eeg order
    ev = mne.EvokedArray(data_win, info_eeg, tmin=0.0,
                         nave=percent_trials.shape[0], comment=title)

    # MNE 1.10: use vlim=(vmin, vmax); provide dicts for scalings/units
    fig = ev.plot_topomap(
        times=[0.0],
        ch_type='eeg',
        scalings={'eeg': 1.0},
        units={'eeg': '%'},
        vlim=(None, None),          # auto; change to (neg, pos) to lock scale
        time_unit='s',
        cmap='RdBu_r',
        show=False
    )
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ===================================
# ---- Driver / Orchestration     ----
# ===================================

def run_ersd_pipeline(
    epochs: mne.Epochs,
    event_id: Dict[str, int],
    outdir: str = "./ersd_outputs",
    # Analysis config
    baseline: Tuple[float, float] = (-2.0, -1.0),
    bands: Dict[str, Tuple[float, float]] = None,
    coi: List[str] = ("C3", "C4"),
    # TFR params
    tfr_freqs: np.ndarray = None,
    morlet_n_cycles: Optional[np.ndarray] = None,
    multitaper_time_bandwidth: float = 4.0,
    multitaper_window_len_sec: float = 0.5,
    # STFT params
    stft_win_len_sec: float = 0.5,
    stft_overlap: float = 0.5,
    # Smoothing
    smoothing_enabled: bool = True,
    smoothing_fwhm_sec: float = 0.1,
    topo_use_temporal_smoothing: bool = False,
    # Onset detection
    threshold_k: float = 2.0,
    sustain_sec: float = 0.1,
    erd_window: Tuple[float, float] = (0.3, 2.5),
    ers_window: Tuple[float, float] = (1.5, 5.0),
    fdr_alpha: float = 0.05,
    fdr_bootstrap: int = 200,
    random_state: int = 7
):
    """
    Run ERS/ERD pipeline with four computation methods and four detectors.
    Produces:
      - onsets_per_trial.csv
      - onsets_group.csv
      - figures for time-courses and topomaps
    """
    _ensure_outdir(outdir)

    if bands is None:
        bands = dict(mu=(8., 12.), beta=(13., 30.))

    if tfr_freqs is None:
        tfr_freqs = np.arange(4., 41., 1.0)

    if morlet_n_cycles is None:
        morlet_n_cycles = np.linspace(3.0, 10.0, len(tfr_freqs))

    print("=== ERS/ERD Pipeline Config ===")
    print(f"Baseline: {baseline} s")
    print(f"Bands: {bands}")
    print(f"Channels of interest: {coi}")
    print(f"Smoothing: {smoothing_enabled} (FWHM={smoothing_fwhm_sec}s)")
    print(f"Topomap temporal smoothing: {topo_use_temporal_smoothing}")
    print(f"ERD window: {erd_window} s, ERS window: {ers_window} s")
    print(f"Detectors: thresh_sd(k={threshold_k}), changepoint(h=5), halfpeak, fdr(alpha={fdr_alpha})")
    print(f"Output dir: {outdir}")

    # Picks: EEG channels only (for plotting & computation)
    picks_eeg = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, ecg=False, stim=False)
    ch_names = np.array(epochs.ch_names)[picks_eeg].tolist()

    # Map COI to indices in EEG picks
    coi_idx = []
    for ch in coi:
        if ch in epochs.ch_names:
            coi_idx.append(np.where(np.array(epochs.ch_names) == ch)[0][0])
        else:
            print(f"Warning: channel {ch} not found; skipping in plots.")
    coi_idx = np.array(coi_idx, dtype=int)

    # Split epochs by condition
    epochs_by_cond = {cond: epochs[cond].copy() for cond in event_id}
    for cond, ep in epochs_by_cond.items():
        print(f"Condition '{cond}': {len(ep)} trials")

    # Prepare CSV accumulators
    rows_per_trial = []
    rows_group = []

    sfreq = float(epochs.info['sfreq'])
    min_len_samples = max(1, int(round(sustain_sec * sfreq)))

    # Iterate conditions and methods
    for cond_name, ep in epochs_by_cond.items():
        # ========== Compute bandpower by each method ==========
        # Pre-compute method outputs per band
        method_power = {}  # method -> band_name -> power array (n_trials, n_channels, n_times)
        method_times = None

        # Hilbert
        for band_name, band_rng in bands.items():
            P_h, times = compute_bandpower_hilbert(ep.copy().pick(picks_eeg), band_rng)
            method_power.setdefault('hilbert', {})[band_name] = P_h
            method_times = times

        # Morlet
        for band_name, band_rng in bands.items():
            P_m, _ = compute_bandpower_tfr(ep.copy().pick(picks_eeg), band_rng, method='morlet',
                                           freqs=tfr_freqs, n_cycles=morlet_n_cycles)
            method_power.setdefault('morlet', {})[band_name] = P_m

        # Multitaper
        n_cycles_mt = tfr_freqs * multitaper_window_len_sec  # ~ constant window length
        for band_name, band_rng in bands.items():
            P_mt, _ = compute_bandpower_tfr(ep.copy().pick(picks_eeg), band_rng, method='multitaper',
                                            freqs=tfr_freqs, n_cycles=n_cycles_mt,
                                            time_bandwidth=multitaper_time_bandwidth)
            method_power.setdefault('multitaper', {})[band_name] = P_mt

        # STFT
        for band_name, band_rng in bands.items():
            P_s, _ = compute_bandpower_stft(ep.copy().pick(picks_eeg), band_rng,
                                            win_len_sec=stft_win_len_sec, overlap=stft_overlap)
            method_power.setdefault('stft', {})[band_name] = P_s

        # ========== For each method & band: baseline norm, smoothing, plots, onsets ==========
        for comp_method, band_dict in method_power.items():
            for band_name, P in band_dict.items():
                # Baseline-normalize
                percent, db, base_mean, base_std = _baseline_norm_percent_db(P, method_times, baseline)

                # Optional temporal smoothing (time-courses); keep unsmoothed copies for topomaps if needed
                percent_smooth = percent.copy()
                db_smooth = db.copy()
                if smoothing_enabled and smoothing_fwhm_sec > 0:
                    percent_smooth = _apply_temporal_smoothing(percent_smooth, sfreq, smoothing_fwhm_sec)
                    db_smooth = _apply_temporal_smoothing(db_smooth, sfreq, smoothing_fwhm_sec)

                # z-scores vs baseline for detectors that use SD
                z = (P - base_mean) / (base_std + 1e-12)
                z_smooth = z.copy()
                if smoothing_enabled and smoothing_fwhm_sec > 0:
                    z_smooth = _apply_temporal_smoothing(z_smooth, sfreq, smoothing_fwhm_sec)

                # ---- Plots: mean ± SD at C3/C4 for % and dB ----
                # extract COI planes in the order of coi list
                if coi_idx.size > 0:
                    # Map COI indices to picks_eeg-relative order
                    # coi_idx are indices in epochs.ch_names; we need to convert to local index in picks_eeg
                    coi_local = []
                    for idx in coi_idx:
                        # find where this global index appears in picks_eeg
                        local = np.where(picks_eeg == idx)[0]
                        if local.size:
                            coi_local.append(int(local[0]))
                    coi_local = np.array(coi_local, dtype=int)
                    if coi_local.size > 0:
                        # %
                        mean_pct = np.nanmean(percent_smooth[:, coi_local, :], axis=0)
                        sd_pct = np.nanstd(percent_smooth[:, coi_local, :], axis=0, ddof=1)
                        title_pct = f"{cond_name} | {comp_method} | {band_name} | %"
                        out_pct = os.path.join(outdir, f"times_{cond_name}_{comp_method}_{band_name}_percent.png")
                        plot_timeseries_mean_sd(method_times, mean_pct, sd_pct, [coi[i] for i in range(coi_local.size)],
                                                title_pct, "% change", baseline, onset_lines=None, outpath=out_pct)
                        # dB
                        mean_db = np.nanmean(db_smooth[:, coi_local, :], axis=0)
                        sd_db = np.nanstd(db_smooth[:, coi_local, :], axis=0, ddof=1)
                        title_db = f"{cond_name} | {comp_method} | {band_name} | dB"
                        out_db = os.path.join(outdir, f"times_{cond_name}_{comp_method}_{band_name}_dB.png")
                        plot_timeseries_mean_sd(method_times, mean_db, sd_db, [coi[i] for i in range(coi_local.size)],
                                                title_db, "dB", baseline, onset_lines=None, outpath=out_db)

                # ---- Onset detection per trial (thresh_sd, changepoint, halfpeak) ----
                # ERD
                erd_onsets_trials = dict(thresh_sd=[], changepoint=[], halfpeak=[])
                ers_onsets_trials = dict(thresh_sd=[], changepoint=[], halfpeak=[])

                # Choose which arrays to feed into detectors
                #   - thresh_sd & changepoint on z-scores
                #   - halfpeak on percent
                for tr in range(P.shape[0]):
                    zz = z_smooth[tr, :, :]
                    pp = percent_smooth[tr, :, :]

                    # We'll detect on the mean over COI (if present), else on the average across EEG channels
                    if coi_idx.size > 0:
                        # map to local picks as above
                        if coi_idx.size > 0:
                            coi_local = []
                            for idx in coi_idx:
                                local = np.where(picks_eeg == idx)[0]
                                if local.size:
                                    coi_local.append(int(local[0]))
                            coi_local = np.array(coi_local, dtype=int)
                        if coi_local.size > 0:
                            z_trial = np.nanmean(zz[coi_local, :], axis=0)
                            p_trial = np.nanmean(pp[coi_local, :], axis=0)
                        else:
                            z_trial = np.nanmean(zz, axis=0)
                            p_trial = np.nanmean(pp, axis=0)
                    else:
                        z_trial = np.nanmean(zz, axis=0)
                        p_trial = np.nanmean(pp, axis=0)

                    # ERD onset per trial
                    erd_t1 = _detector_thresh_sd(z_trial, method_times, k=threshold_k,
                                                 min_dur=sustain_sec, search_window=erd_window, direction='ERD')
                    erd_t2 = _detector_changepoint_cusum(z_trial, method_times, h=5.0,
                                                         min_dur=sustain_sec, search_window=erd_window, direction='ERD')
                    erd_t3 = _detector_halfpeak(p_trial, method_times, search_window=erd_window, direction='ERD')
                    erd_onsets_trials['thresh_sd'].append(erd_t1)
                    erd_onsets_trials['changepoint'].append(erd_t2)
                    erd_onsets_trials['halfpeak'].append(erd_t3)

                    # ERS onset per trial
                    ers_t1 = _detector_thresh_sd(z_trial, method_times, k=threshold_k,
                                                 min_dur=sustain_sec, search_window=ers_window, direction='ERS')
                    ers_t2 = _detector_changepoint_cusum(z_trial, method_times, h=5.0,
                                                         min_dur=sustain_sec, search_window=ers_window, direction='ERS')
                    ers_t3 = _detector_halfpeak(p_trial, method_times, search_window=ers_window, direction='ERS')
                    ers_onsets_trials['thresh_sd'].append(ers_t1)
                    ers_onsets_trials['changepoint'].append(ers_t2)
                    ers_onsets_trials['halfpeak'].append(ers_t3)

                # Save per-trial rows
                n_trials = P.shape[0]
                for det in ['thresh_sd', 'changepoint', 'halfpeak']:
                    for tr in range(n_trials):
                        rows_per_trial.append(dict(
                            comp_method=comp_method,
                            detector=det,
                            band=band_name,
                            channel="COI-avg",
                            condition=cond_name,
                            trial_id=tr,
                            erd_onset_s=float(erd_onsets_trials[det][tr]) if np.isfinite(erd_onsets_trials[det][tr]) else np.nan,
                            ers_onset_s=float(ers_onsets_trials[det][tr]) if np.isfinite(ers_onsets_trials[det][tr]) else np.nan
                        ))

                # ---- Group FDR detector on mean across COI (or all EEG) ----
                if coi_idx.size > 0:
                    # derive local indices again
                    coi_local = []
                    for idx in coi_idx:
                        local = np.where(picks_eeg == idx)[0]
                        if local.size:
                            coi_local.append(int(local[0]))
                    coi_local = np.array(coi_local, dtype=int)
                    if coi_local.size > 0:
                        percent_trials = np.nanmean(percent_smooth[:, coi_local, :], axis=1)  # (trials, times)
                    else:
                        percent_trials = np.nanmean(percent_smooth, axis=1)
                else:
                    percent_trials = np.nanmean(percent_smooth, axis=1)

                erd_onset_fdr, erd_sd_fdr = group_detector_fdr(percent_trials, method_times, alpha=fdr_alpha,
                                                               min_dur=sustain_sec, search_window=erd_window,
                                                               n_boot=fdr_bootstrap, random_state=random_state)
                ers_onset_fdr, ers_sd_fdr = group_detector_fdr(percent_trials, method_times, alpha=fdr_alpha,
                                                               min_dur=sustain_sec, search_window=ers_window,
                                                               n_boot=fdr_bootstrap, random_state=random_state)

                # ---- Aggregate group stats for per-trial detectors ----
                for det in ['thresh_sd', 'changepoint', 'halfpeak']:
                    erd_arr = np.array(erd_onsets_trials[det], dtype=float)
                    ers_arr = np.array(ers_onsets_trials[det], dtype=float)
                    erd_mean = float(np.nanmean(erd_arr)) if np.isfinite(erd_arr).any() else np.nan
                    ers_mean = float(np.nanmean(ers_arr)) if np.isfinite(ers_arr).any() else np.nan
                    erd_sd = float(np.nanstd(erd_arr, ddof=1)) if np.isfinite(erd_arr).sum() > 1 else np.nan
                    ers_sd = float(np.nanstd(ers_arr, ddof=1)) if np.isfinite(ers_arr).sum() > 1 else np.nan

                    rows_group.append(dict(
                        comp_method=comp_method,
                        detector=det,
                        band=band_name,
                        channel="COI-avg",
                        condition=cond_name,
                        n_trials=int(n_trials),
                        erd_onset_s=erd_mean,
                        ers_onset_s=ers_mean,
                        erd_onset_sd_s=erd_sd,
                        ers_onset_sd_s=ers_sd
                    ))

                # FDR group row
                rows_group.append(dict(
                    comp_method=comp_method,
                    detector='fdr',
                    band=band_name,
                    channel="COI-avg",
                    condition=cond_name,
                    n_trials=int(n_trials),
                    erd_onset_s=float(erd_onset_fdr) if np.isfinite(erd_onset_fdr) else np.nan,
                    ers_onset_s=float(ers_onset_fdr) if np.isfinite(ers_onset_fdr) else np.nan,
                    erd_onset_sd_s=float(erd_sd_fdr) if np.isfinite(erd_sd_fdr) else np.nan,
                    ers_onset_sd_s=float(ers_sd_fdr) if np.isfinite(ers_sd_fdr) else np.nan
                ))

                # ---- Topomaps (%), ERD and ERS windows ----
                # Apply or not temporal smoothing for topomaps
                percent_for_topo = percent_smooth if topo_use_temporal_smoothing else percent
                # ERD topo
                topo_title_erd = f"Topo % | {cond_name} | {comp_method} | {band_name} | ERD {erd_window[0]:.1f}-{erd_window[1]:.1f}s"
                topo_out_erd = os.path.join(outdir, f"topo_%_{cond_name}_{comp_method}_{band_name}_ERD.png")
                plot_topomap_percent(ep, percent_for_topo, method_times, topo_win=erd_window,
                                     picks_eeg=picks_eeg, title=topo_title_erd, outpath=topo_out_erd)
                # ERS topo
                topo_title_ers = f"Topo % | {cond_name} | {comp_method} | {band_name} | ERS {ers_window[0]:.1f}-{ers_window[1]:.1f}s"
                topo_out_ers = os.path.join(outdir, f"topo_%_{cond_name}_{comp_method}_{band_name}_ERS.png")
                plot_topomap_percent(ep, percent_for_topo, method_times, topo_win=ers_window,
                                     picks_eeg=picks_eeg, title=topo_title_ers, outpath=topo_out_ers)

    # ====== Write CSVs ======
    df_trials = pd.DataFrame(rows_per_trial)
    df_group = pd.DataFrame(rows_group)
    trials_path = os.path.join(outdir, "onsets_per_trial.csv")
    group_path = os.path.join(outdir, "onsets_group.csv")
    df_trials.to_csv(trials_path, index=False)
    df_group.to_csv(group_path, index=False)

    print(f"Saved: {trials_path}")
    print(f"Saved: {group_path}")
    print("Done.")
