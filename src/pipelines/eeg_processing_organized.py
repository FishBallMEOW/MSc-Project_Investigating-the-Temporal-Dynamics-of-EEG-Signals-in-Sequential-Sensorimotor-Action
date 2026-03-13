# =============================================================================
# eeg_processing_organized.py
# =============================================================================
# Modular EEG/ECoG processing pipeline built on MNE-Python.
# Covers: loading (EDF/CSV/EEGBCI/ECoG .mat), preprocessing, ASR/ICA, epoching,
# AutoReject, TFR ERDS (multitaper/Morlet) with optional cluster masking, and
# Hilbert-based ERD/ERS (dB and %).
# Usage: adjust CONFIG then run `python eeg_processing_organized.py`.
# =============================================================================

# %% Imports
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import mne
from mne.preprocessing import ICA
from mne.stats import permutation_cluster_1samp_test as pcluster_test

try:
    from mne_icalabel import label_components
except Exception:
    label_components = None

try:
    from autoreject import AutoReject, Ransac
except Exception:
    AutoReject, Ransac = None, None

from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'

# =============================================================================
# %% CONFIGURATION
# =============================================================================
@dataclass
class Config:
    source: str = 'edf'  # 'eegbci' | 'csv' | 'edf' | 'ecog_mat'

    # EEGBCI
    eegbci_subject: int = 2
    eegbci_runs: Sequence[int] = (4, 8, 12)

    # CSV
    csv_path: str = str(DATA_DIR / '_archive' / 'Simulated_data' / 'simulated_eeg_with_markers_10Hz_ERS_5s.csv')
    time_col: str = 'timestamp'
    type_col: str = 'type'
    eeg_value: str = 'EEG'
    marker_value: str = 'Marker'
    marker_col: str = 'marker'
    channel_cols: Optional[Sequence[str]] = ('Fp1','F3','F7','C3','T3','P3','T5','O1','Fp2','F4','F8','C4','T4','P4','T6','O2')
    event_code_map: Dict[int, str] = None
    csv_voltage_unit: str = 'uV'  # 'uV'|'mV'|'V'

    # ECoG (.mat)
    ecog_mat_path: str = str(DATA_DIR / '_archive' / 'data_miller' / 'bp_im_t_h.mat')
    ecog_event_id: Dict[str, int] = None
    ecog_sfreq: float = 1000.0
    ecog_scale_uv: float = 0.0298

    # EDF
    edf_path: str = str(DATA_DIR / 'NIC2' / '20250820144519_img_auditory' / '20250820144519_img_auditory_Experiment.edf')

    # Channel type + targets
    ch_type: str = 'eeg'
    target_chs: Sequence[str] = ('C3','C4')

    # Preprocessing
    notch_freq: float = 50.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    resample_sfreq: Optional[float] = None
    reref: object = 'average'  # 'average' or list of ref names

    # Epoching
    tmin: float = -4.0
    tmax: float = 6.0
    baseline: Tuple[float, float] = (-1.5, -0.5)

    # TFR
    tfr_method: str = 'multitaper'  # 'multitaper'|'morlet'
    tfr_freqs: np.ndarray = np.arange(2, 36)
    tfr_n_cycles: object = 'freq'  # 'freq' -> n_cycles=freq
    baseline_mode: str = 'logratio'  # 'percent'|'logratio'|'zscore'

    # ERD/ERS bands
    mu_band: Tuple[float, float] = (8., 13.)
    beta_band: Tuple[float, float] = (13., 30.)

    # ICA / ICLabel
    use_asr: bool = True
    asr_cutoff: float = 20.0
    ica_method: str = 'infomax'
    ica_variance: float = 0.99
    ica_decim: int = 3
    ica_seed: int = 97
    iclabel_thresh: Dict[str, float] = None

    # AutoReject / RANSAC
    use_autoreject: bool = True
    use_ransac: bool = False

    # Reporting
    make_report: bool = True
    report_dir: Path = REPO_ROOT / 'archive' / 'reports'
    out_dir: Path = REPO_ROOT / 'artifacts' / 'outputs'

    def __post_init__(self):
        if self.event_code_map is None:
            self.event_code_map = {21: 'left', 22: 'right'}
        if self.ecog_event_id is None:
            self.ecog_event_id = {'tongue': 11, 'hand': 12}
        if self.iclabel_thresh is None:
            self.iclabel_thresh = {'eye':0.90, 'muscle':0.90, 'line':0.90}

CONFIG = Config()

# =============================================================================
# %% HELPERS
# =============================================================================
def _to_volts(arr: np.ndarray, unit: str) -> np.ndarray:
    # Convert array to Volts given a unit string ('uV'|'mV'|'V')
    u = (unit or '').lower()
    if u in {'uv', 'µv'}: return arr / 1e6
    if u == 'mv':         return arr / 1e3
    if u == 'v':          return arr
    raise ValueError(f'Unknown voltage unit: {unit}')

CUSTOM_8CH_ORDER = ['C3','C4','Fp1','Fp2','O1','O2','T3','T4']

# =============================================================================
# %% LOADING
# =============================================================================
def load_ecog_mat_to_raw(mat_path: str, event_id: Optional[Dict[str,int]], sfreq: float, scale_uv: float):
    # Load imagery/movement ECoG .mat (keys: 'data','stim') into Raw + events
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    data = mat['data']            # time x channels
    stim = mat['stim'].ravel()    # time
    data_v = (data * scale_uv) / 1e6
    ch_names = [f'E{i+1}' for i in range(data_v.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='ecog')
    raw = mne.io.RawArray(data_v.T, info)
    # Events = 0->nonzero transitions
    stim_bin = (stim != 0).astype(int)
    onsets = (np.diff(stim_bin) == 1).nonzero()[0] + 1
    codes = stim[onsets].astype(int)
    if event_id is None:
        default_names = {11:'tongue', 12:'hand'}
        unique_codes = np.unique(codes)
        event_id = {default_names.get(int(c), f'ev{int(c)}'): int(c) for c in unique_codes}
    events = np.column_stack([onsets, np.zeros_like(onsets), [event_id.get(int(c), int(c)) for c in codes]]).astype(int)
    return raw, events, event_id

def _rename_temporal_T7T8_to_T3T4(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    # If channels are T7/T8, rename to T3/T4 to match physical placement
    renames = {}
    if 'T7' in raw.ch_names: renames['T7'] = 'T3'
    if 'T8' in raw.ch_names: renames['T8'] = 'T4'
    if renames:
        mne.rename_channels(raw.info, renames)
        print(f'[EDF] Renamed channels: {renames}')
    return raw

def _set_custom_8ch_montage(raw: mne.io.BaseRaw, only_set_for_known=True) -> mne.io.BaseRaw:
    # Create a montage where T3/T4 take coordinates of T7/T8 from standard_1020
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos']
    alias_map = {'C3':'C3','C4':'C4','Fp1':'Fp1','Fp2':'Fp2','O1':'O1','O2':'O2','T3':'T7','T4':'T8'}
    ch_pos = {}
    for name, alias in alias_map.items():
        if (not only_set_for_known) or (name in raw.ch_names):
            if alias in pos:
                ch_pos[name] = pos[alias]
    if not ch_pos:
        print('[EDF] Warning: no matching channels for custom montage; skipping set_montage.')
        return raw
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(dig, on_missing='ignore')
    return raw

def _build_events_keep_21_22(raw: mne.io.BaseRaw, code_map: Dict[int,str]) -> Tuple[np.ndarray, Dict[str,int]]:
    # Extract only codes 21/22 from annotations or stim channel; map to names
    names = sorted(set(code_map.values()))
    name_to_int = {name: i+1 for i, name in enumerate(names)}
    code_to_int = {code: name_to_int[name] for code, name in code_map.items()}
    events_list = []
    if getattr(raw, 'annotations', None) and len(raw.annotations) > 0:
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            m = re.search(r'(\d+)', str(desc))
            if m:
                code = int(m.group(1))
                if code in code_to_int:
                    sample = int(raw.time_as_index(onset)[0])
                    if 0 <= sample < raw.n_times:
                        events_list.append([sample, 0, code_to_int[code]])
    if not events_list:
        try:
            ev = mne.find_events(raw, shortest_event=1, initial_event=True)
            for sample, _, val in ev:
                if val in code_to_int:
                    events_list.append([sample, 0, code_to_int[val]])
        except Exception:
            pass
    events = np.array(events_list, dtype=int) if events_list else np.empty((0,3), dtype=int)
    event_id = name_to_int
    return events, event_id

def load_edf(path: str | Path, custom_8ch: bool, code_map: Dict[int,str]) -> Tuple[mne.io.BaseRaw, np.ndarray, Dict[str,int]]:
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    print(f'[EDF] Loaded: {path} with {len(raw.ch_names)} channels @ {raw.info["sfreq"]} Hz')
    to_drop = [c for c in ('X','Y','Z') if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
        print(f'[EDF] Dropped non-EEG channels: {to_drop}')
    raw = _rename_temporal_T7T8_to_T3T4(raw)
    raw = _set_custom_8ch_montage(raw, only_set_for_known=True)
    events, event_id = _build_events_keep_21_22(raw, code_map)
    if events.size:
        print(f'[EDF] Using {len(events)} events (codes 21/22 only) with event_id={event_id}')
    else:
        print('[EDF] No 21/22 events found in annotations or stim channel.')
    if custom_8ch:
        available = [ch for ch in CUSTOM_8CH_ORDER if ch in raw.ch_names]
        if available:
            raw.pick(available)
            raw.reorder_channels(available)
            print(f'[EDF] Picked & ordered channels: {available}')
        else:
            print('[EDF] custom_8ch=True but expected 8 channels not found; keeping original layout.')
    return raw, events, event_id

def load_data(cfg: Config) -> Tuple[mne.io.BaseRaw, np.ndarray, Dict[str,int]]:
    src = cfg.source.lower()
    if src == 'eegbci':
        from mne.datasets import eegbci
        print(f'Fetching EEGBCI s{cfg.eegbci_subject} runs {cfg.eegbci_runs} ...')
        file_paths = eegbci.load_data(cfg.eegbci_subject, list(cfg.eegbci_runs))
        raws = [mne.io.read_raw_edf(fp, preload=True) for fp in file_paths]
        for r in raws: eegbci.standardize(r)
        raw = mne.concatenate_raws(raws)
        raw.rename_channels(lambda x: x.strip('.'))
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
        events, _ = mne.events_from_annotations(raw, event_id={'T1':1, 'T2':2})
        event_id = {'left':1, 'right':2}
        return raw, events, event_id
    if src == 'csv':
        df = pd.read_csv(cfg.csv_path, encoding='latin1')
        eeg_df = df[df[cfg.type_col]==cfg.eeg_value].copy()
        mrk_df = df[df[cfg.type_col]==cfg.marker_value].copy()
        ch_cols = list(cfg.channel_cols) if cfg.channel_cols else None
        if ch_cols is None:
            patt = df.columns.str.match(r'^(ch|eeg)\d+$', case=False)
            ch_cols = [c for c, m in zip(df.columns, patt) if m and c in eeg_df.columns]
            if not ch_cols:
                numeric_cols = [c for c in eeg_df.columns if np.issubdtype(eeg_df[c].dtype, np.number)]
                ch_cols = [c for c in numeric_cols if c != cfg.time_col]
        eeg_df = eeg_df.sort_values(cfg.time_col).drop_duplicates(subset=[cfg.time_col])
        ts = eeg_df[cfg.time_col].to_numpy()
        sfreq = float(np.round(1.0 / np.median(np.diff(ts))))
        data = eeg_df[ch_cols].to_numpy().T
        # convert to Volts if needed
        unit = (cfg.csv_voltage_unit or '').lower()
        if unit in {'uv','µv'}: data = data / 1e6
        elif unit == 'mv': data = data / 1e3
        elif unit != 'v': raise ValueError('Unknown csv_voltage_unit')
        info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types=cfg.ch_type)
        raw = mne.io.RawArray(data, info)
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='ignore')
        except Exception:
            pass
        mrk_df = mrk_df.sort_values(cfg.time_col)
        names = sorted(set(cfg.event_code_map.values()))
        name_to_int = {name: i+1 for i, name in enumerate(names)}
        code_to_int = {code: name_to_int[name] for code, name in cfg.event_code_map.items()}
        start_time = eeg_df[cfg.time_col].iloc[0]
        events = []
        for t, code in zip(mrk_df[cfg.time_col].to_numpy(), mrk_df[cfg.marker_col].astype(int).to_numpy()):
            sample = int(np.round((t - start_time) * sfreq))
            if 0 <= sample < raw.n_times and code in code_to_int:
                events.append([sample, 0, code_to_int[code]])
        events = np.array(events, dtype=int)
        event_id = name_to_int
        return raw, events, event_id
    if src == 'ecog_mat':
        raw, events, event_id = load_ecog_mat_to_raw(cfg.ecog_mat_path, cfg.ecog_event_id, cfg.ecog_sfreq, cfg.ecog_scale_uv)
        return raw, events, event_id
    if src == 'edf':
        raw, events, event_id = load_edf(cfg.edf_path, custom_8ch=True, code_map=cfg.event_code_map)
        return raw, events, event_id
    raise ValueError('Unknown source. Use one of: eegbci | csv | ecog_mat | edf')

# =============================================================================
# %% PREPROCESSING + ASR/ICA
# =============================================================================
def basic_preprocess(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    # Apply notch, band-pass, resample (optional), and re-reference (avg/custom)
    out = raw.copy()
    if cfg.notch_freq:
        out.notch_filter(freqs=[cfg.notch_freq], picks=mne.pick_types(out.info, eeg=True, ecog=True))
    out.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, picks=mne.pick_types(out.info, eeg=True, ecog=True))
    if cfg.resample_sfreq:
        out.resample(cfg.resample_sfreq)
    has_eeg = any(t == 'eeg' for t in out.get_channel_types())
    if has_eeg:
        if cfg.reref == 'average':
            out.set_eeg_reference('average', projection=False)
        elif isinstance(cfg.reref, (list, tuple)):
            out.set_eeg_reference(cfg.reref, projection=False)
    return out

def try_asr(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    # Optionally apply ASR if asrpy is available
    if not cfg.use_asr:
        return raw
    try:
        import asrpy
        print('Applying ASR ...')
        asr = asrpy.ASR(sfreq=raw.info['sfreq'], cutoff=cfg.asr_cutoff)
        asr.fit(raw, picks='eeg')
        cleaned = asr.transform(raw, picks='eeg')
        return cleaned
    except Exception as e:
        print('ASR not available; skipping. Reason:', e)
        return raw

def run_ica_and_select(raw_for_ica: mne.io.BaseRaw, cfg: Config) -> Tuple[ICA, List[int]]:
    # Fit ICA and auto-select artifact components via ICLabel (if available) + fallbacks
    fit_params = {'extended': True} if cfg.ica_method == 'infomax' else None
    ica = ICA(method=cfg.ica_method, n_components=cfg.ica_variance, random_state=cfg.ica_seed, max_iter='auto', fit_params=fit_params)
    ica.fit(raw_for_ica, picks='eeg', decim=cfg.ica_decim, reject_by_annotation=True, verbose='WARNING')
    exclude: List[int] = []
    if label_components is not None:
        sfreq = float(raw_for_ica.info['sfreq'])
        nyq = sfreq / 2.0
        ic_hi = min(100.0, nyq - 1.0)
        raw_tmp = raw_for_ica.copy().filter(1.0, ic_hi, picks='eeg', verbose='ERROR')
        ic_labels = label_components(raw_tmp, ica, method='iclabel')
        probs = ic_labels['y_pred_proba']
        labels = ic_labels['labels']
        thr = cfg.iclabel_thresh
        for idx, lbl in enumerate(labels):
            p = probs[idx]
            if 'eye' in lbl and p[2] >= thr['eye']: exclude.append(idx)
            elif 'muscle' in lbl and p[1] >= thr['muscle']: exclude.append(idx)
            elif 'line' in lbl and p[4] >= thr['line']: exclude.append(idx)
    from mne.preprocessing import find_eog_events
    from scipy.stats import kurtosis
    name_map = {c.lower(): c for c in raw_for_ica.ch_names}
    proxies = [name_map[w] for w in ['fpz','fp1','fp2','afz','af7','af8','fz','f1','f2'] if w in name_map]
    detector = proxies[0] if proxies else None
    if detector is None:
        eeg_picks = mne.pick_types(raw_for_ica.info, eeg=True)
        tmp = raw_for_ica.copy().filter(1.0, 10.0, picks=eeg_picks, verbose='ERROR')
        X = tmp.get_data(picks=eeg_picks)
        k = kurtosis(X, axis=1, fisher=True, nan_policy='omit')
        detector = raw_for_ica.ch_names[eeg_picks[int(np.nanargmax(np.abs(k)))]]
        print(f'No standard frontal labels found; using kurtosis-based detector: {detector}')
    try:
        eog_events = find_eog_events(raw_for_ica, ch_name=detector)
        print(f'Found {len(eog_events)} blink-like events on {detector}')
        if len(eog_events):
            eog_epochs = mne.Epochs(raw_for_ica, eog_events, event_id=998, tmin=-0.5, tmax=0.5, baseline=(None,0), picks='eeg', preload=True, reject_by_annotation=True)
            bads_ev, _ = ica.find_bads_eog(eog_epochs, ch_name=detector, measure='correlation', l_freq=1., h_freq=10., reject_by_annotation=True)
            exclude.extend(bads_ev)
    except Exception as ee:
        print('Blink-event fallback failed:', ee)
    try:
        raw_corr = raw_for_ica.copy().pick_types(eeg=True).filter(1., 10., verbose='ERROR')
        src = ica.get_sources(raw_corr).get_data()
        det_sig = raw_corr.copy().pick_channels([detector]).get_data()[0]
        n = min(src.shape[1], det_sig.size)
        src, det_sig = src[:, :n], det_sig[:n]
        def pearson1d(a,b):
            a = a - a.mean(); b = b - b.mean()
            denom = np.sqrt((a*a).sum() * (b*b).sum())
            return 0.0 if denom == 0 else float((a*b).sum() / denom)
        cors = np.array([pearson1d(s, det_sig) for s in src])
        thr = 0.30
        bads_corr = list(np.flatnonzero(np.abs(cors) >= thr))
        exclude.extend(bads_corr)
    except Exception as ec:
        print('Correlation fallback failed:', ec)
    try:
        bads_ecg, _ = ica.find_bads_ecg(raw_for_ica, method='ctps', threshold='auto')
        exclude.extend(bads_ecg)
    except Exception:
        pass
    exclude = sorted(set(exclude))
    print('ICA exclude:', exclude)
    return ica, exclude

def apply_ica(raw_clean_base: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    # Prepare a copy (optional ASR), fit ICA, apply to base cleaned raw
    work = try_asr(raw_clean_base.copy(), cfg)
    ica, exclude = run_ica_and_select(work, cfg)
    ica.exclude = exclude
    out = raw_clean_base.copy()
    ica.apply(out)
    return out

# =============================================================================
# %% EPOCHING + AUTOREJECT
# =============================================================================
def epoch_data(raw: mne.io.BaseRaw, events: np.ndarray, event_id: Dict[str,int], cfg: Config) -> mne.Epochs:
    # Create epochs and optionally clean with AutoReject/RANSAC
    picks = mne.pick_types(raw.info, eeg=True, ecog=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=cfg.tmin-0.5, tmax=cfg.tmax+0.5, baseline=None, picks=picks, preload=True, detrend=1, reject_by_annotation=True)
    print(epochs)
    if cfg.source.lower() != 'ecog_mat' and cfg.use_autoreject and AutoReject is not None:
        if cfg.use_ransac and Ransac is not None:
            ransac = Ransac(random_state=97, n_jobs=-1, verbose='tqdm')
            epochs = ransac.fit_transform(epochs)
        ar = AutoReject(n_jobs=-1, cv=5, random_state=97, verbose='tqdm')
        epochs_clean = ar.fit_transform(epochs)
        print(f'Dropped {ar.get_reject_log(epochs).bad_epochs.sum()} / {len(epochs)} epochs')
        return epochs_clean
    return epochs

# =============================================================================
# %% TFR (MULTITAPER/MORLET) + CLUSTER MASKING
# =============================================================================
def compute_tfr(epochs: mne.Epochs, cfg: Config) -> mne.time_frequency.EpochsTFR:
    # Compute ERDS maps as TFR (average=False), crop and baseline
    n_cycles = cfg.tfr_freqs if cfg.tfr_n_cycles == 'freq' else float(cfg.tfr_n_cycles)
    tfr = epochs.compute_tfr(method=cfg.tfr_method, freqs=cfg.tfr_freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, average=False, decim=1)
    tfr.crop(cfg.tmin, cfg.tmax).apply_baseline(cfg.baseline, mode=cfg.baseline_mode)
    return tfr

def plot_tfr_unmasked_and_clustered(tfr: mne.time_frequency.EpochsTFR, epochs: mne.Epochs, target_chs: Sequence[str], event_id: Dict[str,int]):
    # Plot ERDS maps (unmasked + cluster-masked) per condition for target channels
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type='mask')
    cnorm = TwoSlopeNorm(vcenter=0)
    for event in event_id:
        tfr_ev = tfr[event]
        sel_chs = [ch for ch in target_chs if ch in epochs.ch_names] or epochs.ch_names[:3]
        ch_inds = mne.pick_channels(epochs.ch_names, include=sel_chs)
        tfr_ev_avg = tfr_ev.average()
        fig_raw, axes_raw = plt.subplots(1, len(ch_inds)+1, figsize=(24,4), gridspec_kw={'width_ratios':[10]*len(ch_inds)+[1]})
        for ch_idx, ax in zip(ch_inds, axes_raw[:-1]):
            tfr_ev_avg.plot([ch_idx], cmap='RdBu_r', cnorm=cnorm, axes=ax, colorbar=False, show=False)
            ax.set_title(epochs.ch_names[ch_idx], fontsize=10)
            ax.axvline(0, linewidth=1, color='black', linestyle=':')
            if ch_idx != ch_inds[0]:
                ax.set_ylabel(''); ax.set_yticklabels([])
        fig_raw.colorbar(axes_raw[0].images[-1], cax=axes_raw[-1]).ax.set_yscale('linear')
        fig_raw.suptitle(f'ERDS (unmasked) — {event}')
        plt.show()
        fig, axes = plt.subplots(1, len(ch_inds)+1, figsize=(24,4), gridspec_kw={'width_ratios':[10]*len(ch_inds)+[1]})
        for ch_idx, ax in zip(ch_inds, axes[:-1]):
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch_idx], tail=1, **kwargs)
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch_idx], tail=-1, **kwargs)
            c = np.stack(c1 + c2, axis=2) if (c1 or c2) else np.zeros((*tfr_ev.data.shape[1:], 1), dtype=bool)
            p = np.concatenate((p1, p2)) if (len(p1) or len(p2)) else np.array([1.0])
            mask = c[..., p <= 0.05].any(axis=-1)
            tfr_ev.average().plot([ch_idx], cmap='RdBu_r', cnorm=cnorm, axes=ax, colorbar=False, show=False, mask=mask, mask_style='mask')
            ax.set_title(epochs.ch_names[ch_idx], fontsize=10)
            ax.axvline(0, linewidth=1, color='black', linestyle=':')
            if ch_idx != ch_inds[0]:
                ax.set_ylabel(''); ax.set_yticklabels('')
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale('linear')
        fig.suptitle(f'ERDS (cluster-masked) — {event}')
        plt.show()

# =============================================================================
# %% HILBERT-BASED ERD/ERS (TOPOS + LINES)
# =============================================================================
def erds_topo_db_from_hilbert(epochs: mne.Epochs, band: Tuple[float,float], baseline: Tuple[float,float]):
    lo, hi = band
    ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)
    A = ep.get_data()
    A_db = 20*np.log10(np.maximum(A, 1e-12))
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    A_db -= A_db[..., bmask].mean(axis=-1, keepdims=True)
    band_ts = A_db.mean(axis=0)
    ev = mne.EvokedArray(band_ts, ep.info, tmin=t[0], comment=f'Hilbert {band} (amp-dB vs baseline)')
    return ev, ep

def erds_topo_percent_from_hilbert(epochs: mne.Epochs, band: Tuple[float,float], baseline: Tuple[float,float]):
    lo, hi = band
    ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)
    A = ep.get_data()
    P = A**2
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    B = P[..., bmask].mean(axis=-1, keepdims=True)
    B = np.maximum(B, 1e-20)
    ERDS_pct = 100.0 * (P - B) / B
    band_ts = ERDS_pct.mean(axis=0)
    ev = mne.EvokedArray(band_ts, ep.info, tmin=t[0], comment=f'Hilbert {band} (% power vs baseline)')
    return ev, ep

def erds_line_db_from_hilbert(epochs: mne.Epochs, band: Tuple[float,float], ch_name: str, baseline: Tuple[float,float]):
    lo, hi = band
    ep = epochs.copy().pick(ch_name).filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)
    A = ep.get_data()
    A = np.squeeze(A, axis=1)
    A_db = 20*np.log10(np.maximum(A, 1e-12))
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    A_db -= A_db[:, bmask].mean(axis=1, keepdims=True)
    mean_tc = A_db.mean(axis=0)
    sem_tc  = A_db.std(axis=0, ddof=1) / np.sqrt(A_db.shape[0])
    return t, mean_tc, sem_tc

def erds_line_percent_from_hilbert(epochs: mne.Epochs, band: Tuple[float,float], ch_name: str, baseline: Tuple[float,float]):
    lo, hi = band
    ep = epochs.copy().pick(ch_name).filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)
    A = ep.get_data()
    A = np.squeeze(A, axis=1)
    P = A**2
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    B = P[:, bmask].mean(axis=1, keepdims=True)
    B = np.maximum(B, 1e-20)
    ERDS_pct = 100.0 * (P - B) / B
    mean_tc = ERDS_pct.mean(axis=0)
    sem_tc  = ERDS_pct.std(axis=0, ddof=1) / np.sqrt(ERDS_pct.shape[0])
    return t, mean_tc, sem_tc

def plot_hilbert_suite(epochs: mne.Epochs, event_id: Dict[str,int], target_chs: Sequence[str], baseline: Tuple[float,float], mu_band=(8,13), beta_band=(13,30), sigma=50.0):
    times_topo = np.linspace(-1.0, 6.0, 11)
    for event in event_id:
        ev_mu_db, _   = erds_topo_db_from_hilbert(epochs[event], mu_band, baseline)
        ev_beta_db, _ = erds_topo_db_from_hilbert(epochs[event], beta_band, baseline)
        print(f'[{event}] Topos (dB) – μ and β')
        ev_mu_db.plot_topomap(times=times_topo, ch_type='eeg', units=dict(eeg='dB'), scalings=dict(eeg=1), time_unit='s')
        ev_beta_db.plot_topomap(times=times_topo, ch_type='eeg', units=dict(eeg='dB'), scalings=dict(eeg=1), time_unit='s')
        ev_mu_pct, _   = erds_topo_percent_from_hilbert(epochs[event], mu_band, baseline)
        ev_beta_pct, _ = erds_topo_percent_from_hilbert(epochs[event], beta_band, baseline)
        print(f'[{event}] Topos (% change) – μ and β')
        ev_mu_pct.plot_topomap(times=times_topo, ch_type='eeg', units=dict(eeg='%'), scalings=dict(eeg=1), time_unit='s')
        ev_beta_pct.plot_topomap(times=times_topo, ch_type='eeg', units=dict(eeg='%'), scalings=dict(eeg=1), time_unit='s')
        for ch in target_chs:
            t_mu, mu_mean, mu_sem = erds_line_db_from_hilbert(epochs[event], mu_band, ch, baseline)
            t_be, be_mean, be_sem = erds_line_db_from_hilbert(epochs[event], beta_band, ch, baseline)
            mu_mean = gaussian_filter1d(mu_mean, sigma=sigma)
            be_mean = gaussian_filter1d(be_mean, sigma=sigma)
            plt.figure()
            plt.title(f'{event} — {ch} (Hilbert, dB rel. baseline)')
            plt.plot(t_mu, mu_mean, lw=2, label='μ: 8–13 Hz')
            plt.fill_between(t_mu, mu_mean-mu_sem, mu_mean+mu_sem, alpha=0.2, linewidth=0)
            plt.plot(t_be, be_mean, lw=2, label='β: 13–30 Hz')
            plt.fill_between(t_be, be_mean-be_sem, be_mean+be_sem, alpha=0.2, linewidth=0)
            plt.axhline(0, ls='--', lw=0.8); plt.axvline(0, ls='--', lw=0.8)
            plt.xlabel('Time (s)'); plt.ylabel('ERD/ERS (dB vs baseline)'); plt.legend(); plt.tight_layout(); plt.show()
            t_mu_p, mu_mean_p, mu_sem_p = erds_line_percent_from_hilbert(epochs[event], mu_band,  ch, baseline)
            t_be_p, be_mean_p, be_sem_p = erds_line_percent_from_hilbert(epochs[event], beta_band, ch, baseline)
            mu_mean_p = gaussian_filter1d(mu_mean_p, sigma=sigma)
            be_mean_p = gaussian_filter1d(be_mean_p, sigma=sigma)
            plt.figure()
            plt.title(f'{event} — {ch} (Hilbert, % change vs baseline)')
            plt.plot(t_mu_p, mu_mean_p, lw=2, label='μ: 8–13 Hz')
            plt.fill_between(t_mu_p, mu_mean_p - mu_sem_p, mu_mean_p + mu_sem_p, alpha=0.2, linewidth=0)
            plt.plot(t_be_p, be_mean_p, lw=2, label='β: 13–30 Hz')
            plt.fill_between(t_be_p, be_mean_p - be_sem_p, be_mean_p + be_sem_p, alpha=0.2, linewidth=0)
            plt.axhline(0, ls='--', lw=0.8); plt.axvline(0, ls='--', lw=0.8)
            plt.xlabel('Time (s)'); plt.ylabel('ERD/ERS (% change)'); plt.legend(); plt.tight_layout(); plt.show()

# =============================================================================
# %% REPORT (OPTIONAL)
# =============================================================================
def build_report(raw_before: mne.io.BaseRaw, raw_after_pre: mne.io.BaseRaw, raw_after_ica: Optional[mne.io.BaseRaw], report_dir: Path, src_tag: str) -> Path:
    report = mne.Report(title='Quick Check Report')
    report.add_raw(raw_before, title='Step 0 – Raw (unfiltered)', psd=True)
    report.add_raw(raw_after_pre, title='Step 1 – After notch + 1–40 Hz + avg ref', psd=True)
    fig0 = raw_before.plot_psd(average=True, fmax=60, show=False); fig0.suptitle('PSD – Raw (before)')
    report.add_figure(fig0, title='PSD – Raw (before)'); plt.close(fig0)
    fig1 = raw_after_pre.plot_psd(average=True, fmax=60, show=False); fig1.suptitle('PSD – Filtered (after notch + 1–40)')
    report.add_figure(fig1, title='PSD – Filtered (after)'); plt.close(fig1)
    if raw_after_ica is not None:
        report.add_raw(raw_after_ica, title='Step 4 – After ICA component removal', psd=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f'ica-qc-report_{src_tag}.html'
    report.save(out_path, overwrite=True, open_browser=False)
    print(f'QC report saved: {out_path.resolve()}')
    return out_path

# =============================================================================
# %% MAIN PIPELINE
# =============================================================================
def run_pipeline(cfg: Config = CONFIG):
    print('=== CONFIG ===')
    for k, v in asdict(cfg).items():
        print(f'{k}: {v}')
    print('==============')
    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    raw, events, event_id = load_data(cfg)
    cfg.ch_type = 'ecog' if cfg.source.lower() == 'ecog_mat' else 'eeg'
    print(raw)
    print('Events:', len(events), 'event_id:', event_id)
    if events.size:
        print('First 5 events:\n', events[:5])
    raw_pre = basic_preprocess(raw, cfg)
    raw_ica = apply_ica(raw_pre, cfg)
    if cfg.make_report:
        src_tag = f"{Path(cfg.edf_path).stem}_{cfg.source.lower()}"
        try:
            _ = build_report(raw, raw_pre, raw_ica, cfg.report_dir, src_tag)
        except Exception as e:
            print('Report creation failed:', e)
    epochs = epoch_data(raw_ica, events, event_id, cfg)
    try:
        evokeds_eeg = {name: epochs[name].average(method='mean') for name in event_id}
        for pick in ['C3','C4','O1','O2','T3','T4','T7','T8']:
            if pick in epochs.ch_names:
                mne.viz.plot_compare_evokeds(evokeds_eeg, picks=pick, combine=None, time_unit='s')
        times = np.linspace(-0.08, 0.4, 7)
        for name in event_id.keys():
            if name in evokeds_eeg:
                evokeds_eeg[name].plot_topomap(times=times, time_unit='s', ch_type=cfg.ch_type)
    except Exception as e:
        print('ERP quick-look failed (non-critical):', e)
    tfr = compute_tfr(epochs, cfg)
    plot_tfr_unmasked_and_clustered(tfr, epochs, cfg.target_chs, event_id)
    plot_hilbert_suite(epochs, event_id, cfg.target_chs, cfg.baseline, mu_band=cfg.mu_band, beta_band=cfg.beta_band, sigma=50.0)
    print('Pipeline finished.')

if __name__ == '__main__':
    run_pipeline(CONFIG)
