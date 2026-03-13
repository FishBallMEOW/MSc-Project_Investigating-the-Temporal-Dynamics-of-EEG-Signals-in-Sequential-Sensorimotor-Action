
# %% Imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import mne, re
import seaborn as sns

from mne.decoding import CSP
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import warnings

from autoreject import AutoReject, Ransac

from scipy.ndimage import gaussian_filter1d
from scipy import stats, signal
from scipy.stats import ttest_1samp

# Make figures a bit larger
# plt.rcParams['figure.figsize'] = (9, 4)
# mne.set_log_level('WARNING')


# %% Config
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

source = "edf"   # "eegbci", "csv", "edf", or "ecog_mat"

if source.lower() == "eegbci":
    # EEGBCI settings
    subject = 2
    runs = [4, 8, 12]  # motor imagery, left vs right
elif source.lower() == "csv":
    # CSV settings (used if source == "ecog_mat")
    # 20250818_121855_1block/20250818_121855_eeg_with_markers_renamed.csv  Simulated_data/simulated_eeg_with_markers_10Hz_ERD_5s.csv
    csv_path = str(DATA_DIR / "_archive" / "Simulated_data" / "simulated_eeg_with_markers_10Hz_ERS_5s.csv")   # <- change me
    time_col = "timestamp"
    type_col = "type"          # values: "EEG" or "Marker"
    eeg_value = "EEG"
    marker_value = "Marker"
    marker_col = "marker"      # integer codes
    channel_cols = ["Fp1", "F3", "F7", "C3", "T3", "P3", "T5", "O1", "Fp2", "F4", "F8", "C4", "T4", "P4", "T6", "O2"]        # e.g. [f"ch{i}" for i in range(1,17)]
    event_code_map = {21: "left", 22: "right"}  # CSV marker code -> name
    csv_voltage_unit = "uV"    # "uV" | "mV" | "V"
elif source.lower() == "ecog_mat":
    # ECoG (.mat) settings (used if source == "ecog_mat")
    ecog_mat_path = str(DATA_DIR / "_archive" / "data_miller" / "bp_im_t_h.mat")  # <- change me
    ecog_event_id = {'tongue': 11, 'hand': 12}      # default codes in the dataset
    ecog_sfreq = 1000.0
    ecog_scale_uv = 0.0298   # μV per amplifier unit (per README)
elif source.lower() == "edf":
    # EDF settings (used if source == "edf")
    # 20250820151806_img_visual/20250820151806_img_visual_Experiment.edf
    # 20250820161249_img_multi/20250820161249_img_multi_Experiment.edf
    # 20250820144519_img_auditory/20250820144519_img_auditory_Experiment.edf
    edf_path = str(DATA_DIR / "NIC2" / "20250820144519_img_auditory" / "20250820144519_img_auditory_Experiment.edf")  # <- change me
    event_code_map = {21: "left", 22: "right"}  # codes -> names
else:
    raise ValueError("Unknown source")

if source.lower() == "ecog_mat":
    ch_type = "ecog" 
else:
    ch_type = "eeg" 

# Preprocessing
notch_freq = 50.0
l_freq, h_freq = 0.5, 40.0
resample_sfreq = None     # set None to keep native
reref = "average"          # "average" or list of ref channel names

# Epoching
tmin, tmax = -4.0, 6.0
baseline = (-1.5, -0.5)

# TFR (Morlet)
freqs = np.arange(2, 36)   # 2–35 Hz
n_cycles = 7.0
baseline_mode = "logratio"  # % change: percent, log: logratio, zscore

# ERD metrics
mu_band = (8., 13.)
beta_band = (13., 30.)
erd_onset_threshold = -20.0   # % drop (negative)
mi_window = (0.0, 4.0)        # seconds
if source.lower() == "ecog_mat":
    target_chs = ["E20", "E28"] #, "O1", "O2", "P3", "P4"]

else:
    target_chs = ["C3", "C4"] #, "O1", "O2", "P3", "P4"]

# Report
report = None

# Output directory
OUT = REPO_ROOT / "artifacts" / "outputs"; OUT.mkdir(exist_ok=True, parents=True)
print("Configured. Source =", source)

# %% ECoG loader
def load_ecog_mat_to_raw(mat_path, event_id=None, sfreq=1000.0, scale_uv=0.0298):
    """
    Load an imagery/movement ECoG .mat (with 'data' and 'stim') into MNE Raw + events.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file (e.g., 'AA_imagery_t_h.mat').
    event_id : dict or None
        Mapping of event names to codes, e.g., {'tongue': 11, 'hand': 12}.
        If None, the mapping is inferred from the codes present.
    sfreq : float
        Sampling frequency in Hz (dataset uses 1000 Hz).
    scale_uv : float
        Microvolts per amplifier unit in the dataset. Convert to Volts for MNE.

    Returns
    -------
    raw : mne.io.RawArray
        ECoG data (channels × samples), ch_type='ecog'.
    events : ndarray, shape (n_events, 3)
        [sample, 0, event_code] for cue onsets.
    event_id : dict
        Name→code mapping.
    """
    from scipy.io import loadmat
    import numpy as np
    import mne

    mat = loadmat(mat_path)
    data = mat['data']           # shape: time × channels
    stim = mat['stim'].ravel()   # shape: time

    # Convert μV to V for MNE
    data_v = (data * scale_uv) / 1e6  # μV → V

    ch_names = [f'E{i+1}' for i in range(data_v.shape[1])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='ecog')
    raw = mne.io.RawArray(data_v.T, info)

    # Event onsets = transitions from 0 → non-zero in 'stim'
    stim_bin = (stim != 0).astype(int)
    onsets = (np.diff(stim_bin) == 1).nonzero()[0] + 1
    codes = stim[onsets].astype(int)

    if event_id is None:
        # Infer mapping if not provided
        default_names = {11: 'tongue', 12: 'hand'}
        unique_codes = np.unique(codes)
        event_id = {default_names.get(int(c), f'ev{int(c)}'): int(c) for c in unique_codes}

    events = np.column_stack([onsets, np.zeros_like(onsets), [event_id.get(int(c), int(c)) for c in codes]]).astype(int)
    return raw, events, event_id

CUSTOM_8CH_ORDER = ["C3", "C4", "Fp1", "Fp2", "O1", "O2", "T3", "T4"]

def _rename_temporal_T7T8_to_T3T4(raw):
    """If channels are named T7/T8, rename to T3/T4 as per user's physical placement."""
    renames = {}
    if "T7" in raw.ch_names: renames["T7"] = "T3"
    if "T8" in raw.ch_names: renames["T8"] = "T4"
    if renames:
        mne.rename_channels(raw.info, renames)
        print(f"[EDF] Renamed channels: {renames}")
    return raw

def _set_custom_8ch_montage(raw, only_set_for_known=True):
    """
    Create a montage where T3/T4 take the coordinates of T7/T8 from standard_1020.
    This preserves accurate positions even after renaming.
    If only_set_for_known=True, we only map positions for channels present in raw.
    """
    std = mne.channels.make_standard_montage("standard_1020")
    pos = std.get_positions()["ch_pos"]
    # Build mapping for our 8 channels
    alias_map = {
        "C3": "C3",
        "C4": "C4",
        "Fp1": "Fp1",
        "Fp2": "Fp2",
        "O1": "O1",
        "O2": "O2",
        "T3": "T7",  # <- alias: T3 uses T7 coords
        "T4": "T8",  # <- alias: T4 uses T8 coords
    }
    ch_pos = {}
    for name, alias in alias_map.items():
        if (not only_set_for_known) or (name in raw.ch_names):
            if alias in pos:
                ch_pos[name] = pos[alias]
    if not ch_pos:
        print("[EDF] Warning: no matching channels for custom montage; skipping set_montage.")
        return raw
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(dig, on_missing="ignore")
    return raw

def _build_events_keep_21_22(raw):
    """
    Return (events, event_id) where:
      - events[:,2] are small integers (1..N) mapped from names
      - event_id maps names -> ints, e.g. {'left':1, 'right':2}
    Only codes 21 and 22 are kept.
    """
    # names -> ints (stable, sorted by name like CSV loader)
    names = sorted(set(event_code_map.values()))
    name_to_int = {name: i + 1 for i, name in enumerate(names)}
    code_to_int = {code: name_to_int[name] for code, name in event_code_map.items()}

    events_list = []

    # 1) Try annotations first (descriptions may be '21', '22', or contain digits)
    if getattr(raw, "annotations", None) and len(raw.annotations) > 0:
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            m = re.search(r"(\d+)", str(desc))
            if m:
                code = int(m.group(1))
                if code in code_to_int:
                    sample = int(raw.time_as_index(onset)[0])
                    if 0 <= sample < raw.n_times:
                        events_list.append([sample, 0, code_to_int[code]])

    # 2) If none from annotations, fall back to stim channel
    if not events_list:
        try:
            ev = mne.find_events(raw, shortest_event=1, initial_event=True)
            for sample, _, val in ev:
                if val in code_to_int:
                    events_list.append([sample, 0, code_to_int[val]])
        except Exception:
            pass

    events = np.array(events_list, dtype=int) if events_list else np.empty((0, 3), dtype=int)
    event_id = name_to_int  # names -> ints, same as CSV
    return events, event_id

def load_edf(edf_path: str | Path, preload: bool=True, verbose: str | bool='ERROR', custom_8ch=False):
    ...
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose=verbose)
    print(f"[EDF] Loaded: {edf_path} with {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")

    # --- drop non-EEG channels X/Y/Z if present ---
    to_drop = [c for c in ("X", "Y", "Z") if c in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
        print(f"[EDF] Dropped non-EEG channels: {to_drop}")

    # (keep your existing T7/T8 -> T3/T4 rename + custom montage here)
    raw = _rename_temporal_T7T8_to_T3T4(raw)
    raw = _set_custom_8ch_montage(raw, only_set_for_known=True)

    # --- build events exactly like CSV (keep only 21/22) ---
    events, event_id = _build_events_keep_21_22(raw)
    if events.size:
        print(f"[EDF] Using {len(events)} events (codes 21/22 only) with event_id={event_id}")
    else:
        print("[EDF] No 21/22 events found in annotations or stim channel.")
        # You can decide whether to proceed without events or raise here.

    # (optional) enforce your 8-channel order if requested
    if custom_8ch:
        available = [ch for ch in CUSTOM_8CH_ORDER if ch in raw.ch_names]
        if available:
            raw.pick(available)
            raw.reorder_channels(available)
            print(f"[EDF] Picked & ordered channels: {available}")
        else:
            print("[EDF] custom_8ch=True but none of the expected 8 channels found; keeping original layout.")

    return raw, events, event_id


# %% Load data
def _to_volts(arr, unit: str):
    u = (unit or '').lower()
    if u in {"uv", "µv"}: return arr / 1e6
    if u in {"mv"}:       return arr / 1e3
    if u in {"v"}:        return arr
    raise ValueError(f"Unknown voltage unit: {unit}")

if source.lower() == "eegbci":
    from mne.datasets import eegbci
    print(f"Fetching EEGBCI s{subject} runs {runs} ...")
    file_paths = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(fp, preload=True) for fp in file_paths]
    for r in raws: eegbci.standardize(r)
    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda x: x.strip("."))  # remove dots from channel names
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    events, _ = mne.events_from_annotations(raw, event_id={"T1":1, "T2":2})
    event_id = {"left":1, "right":2}

elif source.lower() == "csv":
    df = pd.read_csv(csv_path, encoding="latin1")
    eeg_df = df[df[type_col] == eeg_value].copy()
    mrk_df = df[df[type_col] == marker_value].copy()
    # channels
    ch_cols = channel_cols
    print(ch_cols)
    if ch_cols is None:
        patt = df.columns.str.match(r"^(ch|eeg)\d+$", case=False)
        ch_cols = [c for c, m in zip(df.columns, patt) if m and c in eeg_df.columns]
        if not ch_cols:
            numeric_cols = [c for c in eeg_df.columns if np.issubdtype(eeg_df[c].dtype, np.number)]
            ch_cols = [c for c in numeric_cols if c != time_col]
    # sampling rate
    eeg_df = eeg_df.sort_values(time_col).drop_duplicates(subset=[time_col])
    ts = eeg_df[time_col].to_numpy()
    sfreq = float(np.round(1.0 / np.median(np.diff(ts))))
    data = eeg_df[ch_cols].to_numpy().T
    data = _to_volts(data, csv_voltage_unit)
    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types=ch_type)
    raw = mne.io.RawArray(data, info)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")  # TODO:check later
        raw.set_montage(montage, on_missing="ignore")
    except Exception:
        pass
    # events
    mrk_df = mrk_df.sort_values(time_col)
    names = sorted(set(event_code_map.values()))
    name_to_int = {name: i+1 for i, name in enumerate(names)}
    code_to_int = {code: name_to_int[name] for code, name in event_code_map.items()}
    start_time = eeg_df[time_col].iloc[0]
    events = []
    for t, code in zip(mrk_df[time_col].to_numpy(), mrk_df[marker_col].astype(int).to_numpy()):
        sample = int(np.round((t - start_time) * sfreq))
        if 0 <= sample < raw.n_times and code in code_to_int:
            events.append([sample, 0, code_to_int[code]])
    events = np.array(events, dtype=int)
    event_id = name_to_int

elif source.lower() == "ecog_mat":
    print(f"Loading ECoG .mat: {ecog_mat_path}")
    raw, events, event_id = load_ecog_mat_to_raw(
        ecog_mat_path,
        event_id=ecog_event_id,
        sfreq=ecog_sfreq,
        scale_uv=ecog_scale_uv
    )
elif source.lower() == "edf":
    raw, events, event_id = load_edf(edf_path, custom_8ch=True)
else:
    raise ValueError("source must be 'eegbci' or 'csv' or 'ecog_mat'")

print(raw)
print("Events:", len(events), "event_id:", event_id)
print("First 5 events:\n", events[:5])

# Report
REPORT_DIR = REPO_ROOT / "archive" / "reports"

# Helper
def _maybe_show(fig):
    try: 
        plt.show(block=False)
    except: 
        pass

def _close(fig):
    try:
        plt.close(fig)
    except:
        pass
    
report = mne.Report(title=f"Quick Check Report")
report.add_raw(raw, title="Step 0 – Raw (unfiltered)", psd=True)


# %% Preprocess + QC
raw_f = raw.copy()
if notch_freq:
    raw_f.notch_filter(freqs=[notch_freq], picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
if l_freq is not None or h_freq is not None:
    raw_f.filter(l_freq=l_freq, h_freq=h_freq, picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
if resample_sfreq is not None and resample_sfreq > 0:
    raw_f.resample(resample_sfreq)
# Apply re-referencing only if there are EEG channels:
has_eeg = any(t == 'eeg' for t in raw_f.get_channel_types())
if has_eeg:
    if reref == "average":
        raw_f.set_eeg_reference("average", projection=False)
    elif isinstance(reref, (list, tuple)):
        raw_f.set_eeg_reference(reref, projection=False)

# QC overlay plot (first 5 s)
max_channels = 16
pref = [ch for ch in target_chs if ch in raw.ch_names and ch in raw_f.ch_names]
chs = (pref or raw_f.ch_names)[:max_channels]

r0 = raw.copy()
r1 = raw_f.copy()
if abs(r0.info['sfreq'] - r1.info['sfreq']) > 1e-6:
    r0.resample(r1.info['sfreq'])
win = (0.0, min(5.0, r0.times[-1], r1.times[-1]))
r0.crop(*win); r1.crop(*win)
p0 = mne.pick_channels(r0.ch_names, include=chs)
p1 = mne.pick_channels(r1.ch_names, include=chs)

t = r1.times
dat0 = (r0.get_data(picks=p0) * 1e6)
dat1 = (r1.get_data(picks=p1) * 1e6)

fig, axes = plt.subplots(len(chs), 1, sharex=True, figsize=(10, max(3, len(chs)*1.3)))
if len(chs) == 1: axes = [axes]
for ax, ch, y0, y1 in zip(axes, chs, dat0, dat1):
    ax.plot(t, y0, label="Raw", alpha=0.6, linewidth=0.8)
    ax.plot(t, y1, label="Filtered", linewidth=1.0)
    ax.set_ylabel(f"{ch} (µV)"); ax.grid(True, linestyle="--", alpha=0.3)
axes[0].legend(loc="upper right", frameon=False)
axes[-1].set_xlabel("Time (s)")
plt.suptitle(f"EEG overlay (first {win[1]-win[0]:.1f}s)"); plt.tight_layout()
plt.show()

# Report
report.add_raw(raw_f, title="Step 1 – After notch + 1–40 Hz + avg ref", psd=True)

# Optional quick PSD overlay (before vs after)
fig = raw.plot_psd(average=True, fmax=60, show=False)
fig.suptitle("PSD – Raw (before filtering)")
_maybe_show(fig)
report.add_figure(fig, title="PSD – Raw (before)")
_close(fig)

fig = raw_f.plot_psd(average=True, fmax=60, show=False)
fig.suptitle("PSD – Filtered (after notch + 1–40)")
_maybe_show(fig)
report.add_figure(fig, title="PSD – Filtered (after)")
_close(fig)

X = raw.get_data(picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
X_f = raw_f.get_data(picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
for i in range(1): #(X.shape[0]):
    plt.plot(X[i])
    # plt.plot(X_f[i])
    plt.title(f"Raw (not filtered): Channel {i+1} - {raw.ch_names[i]}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (V)")
    plt.show()
    # plt.plot(X[i])
    plt.plot(X_f[i])
    plt.title(f"Filtered: Channel {i+1} - {raw.ch_names[i]}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (V)")
    plt.show()

# ICA configs
ASR_CUTOFF = 20.0

ICA_METHOD = "infomax"                  # "infomax", "picard", or "fastica"
ICA_VARIANCE = 0.99                     # keep comps to explain 99% variance
DECIM = 3
RAND_SEED = 97

ICLABEL_THRESH = dict(eye=0.90, muscle=0.90, line=0.90)  # EEGLAB-like cutoffs

raw_ica = raw_f.copy()

# ---- Optional ASR (Clean Rawdata-like) ----
try:
    import asrpy  # ← use the package namespace
    print("Applying ASR ...")
    asr = asrpy.ASR(sfreq=raw_ica.info["sfreq"], cutoff=ASR_CUTOFF)

    # Calibrate on EEG channels, then transform the same Raw
    asr.fit(raw_ica, picks="eeg")
    raw_ica = asr.transform(raw_ica, picks="eeg")  # returns a cleaned Raw

    # Report
    report.add_raw(raw_ica, title=f"Step 2 – After ASR (cutoff={ASR_CUTOFF})", psd=True)

except Exception as e:
    print("ASR not available; skipping. Reason:", e)

print(f"Fitting ICA ({ICA_METHOD}) ...")

fit_params = {"extended": True} if ICA_METHOD == "infomax" else None  # EEGLAB-style extended Infomax

ica = ICA(
    method=ICA_METHOD,
    n_components=ICA_VARIANCE,
    random_state=RAND_SEED,
    max_iter="auto",
    fit_params=fit_params,          # << moved here
)

ica.fit(
    raw_ica,
    picks="eeg",
    decim=DECIM,
    reject_by_annotation=True,
    verbose="WARNING",
)
print(ica)

# Report
# Component maps and overview
figs = ica.plot_components(show=False)
if type(figs) is not list:
    figs = [figs]
for fig in figs:
    fig.suptitle("ICA – Component topographies")
    _maybe_show(fig)
    report.add_figure(fig, title="Step 3 – ICA component maps")
    _close(fig)

# Power spectra of sources (quick look)
fig = ica.plot_sources(raw_ica)
if type(figs) is not list:
    figs = [figs]
for fig in figs:
    _maybe_show(fig)
    report.add_figure(fig, title="ICA sources – example timeseries (10 s)")
    _close(fig)

fig = ica.plot_overlay(raw, exclude=[0], picks="eeg")
_maybe_show(fig)
report.add_figure(fig, title="ICA sources – Raw, global field power, and average across chs")
_close(fig)

ica.plot_properties(raw, picks=[0])

# Setup variables for ICLabel
exclude = []
used_iclabel = False

# ---- Select artifact ICs (ICLabel first, robust fallbacks) ----
# 1) Try ICLabel on a 1–100 Hz COPY (recommended by ICLabel)
from mne_icalabel import label_components

sfreq = float(raw_ica.info["sfreq"])
nyq = sfreq / 2.0
ic_hi = min(100.0, nyq - 1.0)   # e.g., 79.0 for EEGBCI (160 Hz)
raw_ica.filter(1.0, ic_hi, picks="eeg", verbose="ERROR")
ic_labels = label_components(raw_ica, ica, method="iclabel")
used_iclabel = True

# ICLabel order: ['brain','muscle','eye','heart','line_noise','channel_noise','other']
EYE_T, MUS_T, LINE_T = ICLABEL_THRESH["eye"], ICLABEL_THRESH["muscle"], ICLABEL_THRESH["line"]
labels = ic_labels['labels']
probs = ic_labels['y_pred_proba']

exclude_idx = [
    idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
]

print(probs, labels)
for i in exclude_idx:
    if labels[i] == "eye blink":
        if probs[i] >= EYE_T:
            exclude.append(i)
    elif labels[i] == "muscle artifact":
        if probs[i] >= MUS_T:
            exclude.append(i)
    elif labels[i] == "line_noise":
        if probs[i] >= LINE_T:
            exclude.append(i)
exclude = sorted(set(exclude))
print("ICLabel picked components:", exclude)

# ---- Select artifact ICs (ICLabel first, robust fallbacks) ----
import numpy as np
from mne.preprocessing import find_eog_events
from scipy.stats import kurtosis

# Case-insensitive channel name helper
name_map = {c.lower(): c for c in raw_ica.ch_names}
wanted = ["fpz","fp1","fp2","afz","af7","af8","fz","f1","f2"]
proxies = [name_map[w] for w in wanted if w in name_map]

# If none of the typical frontals exist, auto-pick a "blink detector"
# = EEG channel with the highest kurtosis in 1–10 Hz (blink-salient)
detector = None
if proxies:
    detector = proxies[0]
else:
    eeg_picks = mne.pick_types(raw_ica.info, eeg=True)
    tmp = raw_ica.copy().filter(1.0, 10.0, picks=eeg_picks, verbose="ERROR")
    X = tmp.get_data(picks=eeg_picks)
    k = kurtosis(X, axis=1, fisher=True, nan_policy="omit")
    detector = raw_ica.ch_names[eeg_picks[int(np.nanargmax(np.abs(k)))]]
    print(f"No standard frontal labels found; using kurtosis-based detector: {detector}")


# (A) Event-based blink epochs from the detector channel
try:
    eog_events = find_eog_events(raw_ica, ch_name=detector)
    print(f"Found {len(eog_events)} blink-like events on '{detector}'")
    if len(eog_events):
        eog_epochs = mne.Epochs(raw_ica, eog_events, event_id=998, tmin=-0.5, tmax=0.5, 
                                baseline=(None, 0), picks="eeg", preload=True, reject_by_annotation=True)
        print(eog_epochs)
        bads_ev, _ = ica.find_bads_eog(
            eog_epochs, ch_name=detector, measure="correlation",
            l_freq=1., h_freq=10., reject_by_annotation=True
        )
        exclude.extend(bads_ev)
        print("Event-based EOG picks:", sorted(set(bads_ev)))
except Exception as ee:
    print("Blink-event fallback failed:", ee)

# (B) Correlation fallback: IC sources vs detector signal
try:
    # Work from ONE common, band-limited copy
    raw_corr = raw_f.copy().load_data()
    raw_corr.pick_types(eeg=True)                      # match ICA training picks
    raw_corr.filter(1., 10., verbose="ERROR")          # blink band

    # Sources and detector from the SAME time base
    src = ica.get_sources(raw_corr).get_data()         # (n_comp, n_times)
    det_sig = raw_corr.copy().pick_channels([detector]).get_data()[0]  # (n_times,)

    # Align lengths just in case (e.g., tiny edge differences)
    n = min(src.shape[1], det_sig.size)
    if src.shape[1] != n or det_sig.size != n:
        print(f"Aligning lengths: src={src.shape[1]} det={det_sig.size} -> {n}")
    src = src[:, :n]
    det_sig = det_sig[:n]

    # Correlation (faster & safer than corrcoef in a loop)
    def pearson1d(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a*a).sum() * (b*b).sum())
        return 0.0 if denom == 0 else float((a*b).sum() / denom)

    cors = np.array([pearson1d(s, det_sig) for s in src])
    thr = 0.30
    bads_corr = list(np.flatnonzero(np.abs(cors) >= thr))
    exclude.extend(bads_corr)
    print(f"Correlation-based picks (|r| ≥ {thr:.2f}):", sorted(set(bads_corr)))
except Exception as ec:
    print("Correlation fallback failed:", ec)

# (C) Optional: ECG via CTPS (works without ECG channel sometimes)
try:
    bads_ecg, _ = ica.find_bads_ecg(raw_ica, method="ctps", threshold="auto")
    exclude.extend(bads_ecg)
    if bads_ecg:
        print("ECG-like picks (CTPS):", sorted(set(bads_ecg)))
except Exception:
    pass

exclude = sorted(set(exclude))
print("Fallback picked components:", exclude)

# Apply exclusion as usual
ica.exclude = exclude

raw_ica = raw_f.copy()
ica.apply(raw_ica)

# Report ===
try:
    import matplotlib.pyplot as plt
    classes = ['brain','muscle','eye','heart','line','chan_noise','other']
    S = np.array(probs)  # (n_comp, 7)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(S.T, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Component index")
    ax.set_title("ICLabel probabilities (per component)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _maybe_show(fig)
    report.add_figure(fig, title="ICLabel – probabilities heatmap")
    _close(fig)
except Exception:
    pass

# Component maps and overview
figs = ica.plot_components(show=False)
if type(figs) is not list:
    figs = [figs]
for fig in figs:
    fig.suptitle("ICA – Component topographies")
    _maybe_show(fig)
    report.add_figure(fig, title="Step 3 – ICA component maps")
    _close(fig)

# Power spectra of sources (quick look)
fig = ica.plot_sources(raw_f)
if type(figs) is not list:
    figs = [figs]
for fig in figs:
    _maybe_show(fig)
    report.add_figure(fig, title="ICA sources – example timeseries (10 s)")
    _close(fig)

report.add_raw(raw_ica, title="Step 4 – After ICA component removal", psd=True)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
file_name = source.lower()
new_file_name = 'ica-qc-report_20250820161249_img_multi_'+str(file_name)+'.html'
report_path = REPORT_DIR / new_file_name
report.save(report_path, overwrite=True, open_browser=False)
print(f"\nQC report saved: {report_path.resolve()}")
print("Open it in your browser for a step-by-step visual walkthrough.")

raw_f = raw_ica.copy()


# %% Epoching
picks = mne.pick_types(raw_f.info, eeg=True, ecog=True, exclude='bads')
epochs = mne.Epochs(raw_f, events, event_id=event_id, tmin=tmin - 0.5, tmax=tmax + 0.5,
                    baseline=None, picks=picks, preload=True, detrend=1)
print(epochs)
# epochs.plot(block=False)


list(epochs.event_id.keys())

if source.lower() != "ecog_mat":
    # (Optional) RANSAC first: finds consistently bad sensors and interpolates them within epochs
    # ransac = Ransac(random_state=97, n_jobs=-1, verbose='tqdm')  # defaults are fine
    # epochs_r = ransac.fit_transform(epochs)
    epochs_r = epochs.copy()  # use original epochs if not using RANSAC

    # AutoReject learns thresholds, interpolates bad channels, drops hopeless epochs ---
    ar = AutoReject(
        n_jobs=-1,
        cv=5,                         # cross-validation folds
        random_state=97,
        verbose='tqdm'
        # You can tune these if needed:
        # n_interpolate=[1, 2, 4, 8, 16],              # how many channels can be interpolated per epoch
        # consensus=np.linspace(0.1, 0.6, 6)           # fraction of sensors that must agree a sample is bad
    )
    # epochs_clean
    epochs_clean = ar.fit_transform(epochs_r)       # main step
    reject_log = ar.get_reject_log(epochs_r)        # what got rejected/interpolated

    print(reject_log.labels.shape, "epochs x channels")
    print(f"Dropped {reject_log.bad_epochs.sum()} / {len(epochs_r)} epochs")

    # Optional quick sanity checks:
    reject_log.plot()                 # heatmap of bad channels/epochs
    epochs_clean.plot()               # scroll a few trials
    epochs_clean.interpolate_bads()   # should already be clean, but harmless if called

if source.lower() != "ecog_mat":
    X = epochs.get_data(picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
    X_f = epochs_clean.get_data(picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
    for i in range(X.shape[0]):
        j = 5
        plt.plot(X[i, j, :])
        plt.title(f"Before AutoReject: Channel {j+1} - {raw.ch_names[j]}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (V)")
    plt.show()

    for i in range(X_f.shape[0]):
        j = 5
        plt.plot(X_f[i, j, :])
        plt.title(f"After AutoReject: Channel {j+1} - {raw.ch_names[j]}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude (V)")
    plt.show()

X = epochs.get_data(picks=mne.pick_types(raw_f.info, eeg=True, ecog=True))
for i in range(X.shape[0]//10):
    j = 0
    plt.plot(X[i, j, :])
    plt.title(f"Trial {i+1} - Channel {j+1} - {raw.ch_names[j]}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (V)")
    plt.show()

if source.lower() != "ecog_mat":
    epochs = epochs_clean.copy()  # use the cleaned epochs for further processing

evokeds_eeg = {name: epochs[name].average(method="mean") for name in event_id}
print(evokeds_eeg)

# # Quick look: single condition trace with GFP overlay & spatial coloring
# for name, ev in evokeds_eeg.items():
#     ev.plot(spatial_colors=True, gfp=True, time_unit="s", titles=f"ERP – {name}")

if source.lower() != "ecog_mat":
    # compare conditions on a specific channel (e.g., C3, C4)
    mne.viz.plot_compare_evokeds(evokeds_eeg, picks="C3", combine=None, time_unit="s")
    mne.viz.plot_compare_evokeds(evokeds_eeg, picks="C4", combine=None, time_unit="s")
    mne.viz.plot_compare_evokeds(evokeds_eeg, picks="O1", combine=None, time_unit="s")
    mne.viz.plot_compare_evokeds(evokeds_eeg, picks="O2", combine=None, time_unit="s")
    try:
        mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T3", combine=None, time_unit="s")
        mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T4", combine=None, time_unit="s")
    except Exception:
        pass
    try:
        mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T7", combine=None, time_unit="s")
        mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T8", combine=None, time_unit="s")
    except Exception:
        pass

    # topographies at key latencies (adjust to your paradigm, e.g., N100/P300 windows)
    times = np.linspace(-0.08, 0.4, 7)  # -80–400 ms
    for name in event_id.keys(): 
        if name in evokeds_eeg:
            print(name)
            evokeds_eeg[name].plot_topomap(times=times, time_unit="s", ch_type=ch_type)
else:
    for name in epochs.ch_names:
        if name in evokeds_eeg:
            print(name)

        # compare conditions on a specific channel (e.g., C3, C4)
        mne.viz.plot_compare_evokeds(evokeds_eeg, picks=name, combine=None, time_unit="s")
        # mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T3", combine=None, time_unit="s")
        # mne.viz.plot_compare_evokeds(evokeds_eeg, picks="T4", combine=None, time_unit="s")


# %% Multitaper/Morlet
method = "multitaper" # "multitaper" or "morlet"

freqs = np.arange(2, 36)  # frequencies from 2-35Hz
n_cycles = freqs  # fixed number of cycles
# n_cycles = np.interp(freqs, [6, 12, 30], [4, 5, 7])  # 0.33–0.67 s windows
time_bandwidth = 1.5  # W ≈ 1.5/T → ~±2–3 Hz around 10–20 Hz

# Compute TFR
tfr = epochs.compute_tfr(method=method, freqs=freqs,
    n_cycles=n_cycles, use_fft=True, return_itc=False,
    average=False, decim=1)
tfr.crop(tmin, tmax).apply_baseline(baseline, mode=baseline_mode)

kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test
# vmin, vmax = -0.5, 0.5  # set min and max ERDS values in plot
cnorm = TwoSlopeNorm(vcenter=0)  # min, center & max ERDS

# %% Unmasked Maps (no clustering)
for event in event_id:
    tfr_ev = tfr[event]

    # pick sensorimotor channels (fallback to first 3 if any are missing)
    sel_chs = [ch for ch in target_chs if ch in epochs.ch_names]
    if not sel_chs:
        sel_chs = epochs.ch_names[:3]
    ch_inds = mne.pick_channels(epochs.ch_names, include=sel_chs)

    # precompute condition-average once
    tfr_ev_avg = tfr_ev.average()

    # figure with one panel per selected channel + colorbar
    fig_raw, axes_raw = plt.subplots(
        1, len(ch_inds) + 1, figsize=(24, 4),
        gridspec_kw={"width_ratios": [10] * len(ch_inds) + [1]}
    )

    # plot each channel's full ERDS (no mask)
    for ch_idx, ax in zip(ch_inds, axes_raw[:-1]):
        tfr_ev_avg.plot(
            [ch_idx],
            cmap="RdBu_r",
            cnorm=cnorm,         # same symmetric color scale around 0
            axes=ax,
            colorbar=False,
            show=False,
        )
        ax.set_title(epochs.ch_names[ch_idx], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event line
        if ch_idx != ch_inds[0]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    # one shared colorbar
    fig_raw.colorbar(axes_raw[0].images[-1], cax=axes_raw[-1]).ax.set_yscale("linear")
    fig_raw.suptitle(f"ERDS (unmasked, no clustering) — {event}")
    plt.show()


# %% Cluster-based ERDS Maps
for event in event_id:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    
    # pick sensorimotor channels (fallback to first 3 if any are missing)
    sel_chs = [ch for ch in target_chs if ch in epochs.ch_names]

    if not sel_chs:
        sel_chs = epochs.ch_names[:3]
    ch_inds = mne.pick_channels(epochs.ch_names, include=sel_chs)

    # figure with one panel per selected channel + colorbar
    fig, axes = plt.subplots(
        1, len(ch_inds) + 1, figsize=(24, 4),
        gridspec_kw={"width_ratios": [10] * len(ch_inds) + [1]}
    )

    for ch_idx, ax in zip(ch_inds, axes[:-1]):
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch_idx], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch_idx], tail=-1, **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot(
            [ch_idx],
            cmap="RdBu_r",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(epochs.ch_names[ch_idx], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch_idx != ch_inds[0]:
            ax.set_ylabel("")
            ax.set_yticklabels("")

    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()

def erds_topo_percent_from_hilbert(epochs, band, baseline=(-1.0, 0.0)):
    """
    Same output shape as erds_topo_from_hilbert, but values are %ERD/ERS
    relative to baseline (negative = ERD, positive = ERS).

    Returns
    -------
    ev : mne.EvokedArray   # data shape: (n_channels, n_times), units: %
    ep : mne.Epochs        # filtered + Hilbert-applied copy
    """
    lo, hi = band
    ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)                 # amplitude envelope
    A = ep.get_data()                               # (n_epochs, n_ch, n_times)
    P = A ** 2                                      # power

    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])

    # per-trial, per-channel baseline power (mean over baseline window)
    B = P[..., bmask].mean(axis=-1, keepdims=True)  # (n_epochs, n_ch, 1)
    B = np.maximum(B, 1e-20)                        # avoid divide-by-zero

    # % change vs baseline
    ERDS_pct = 100.0 * (P - B) / B                  # (n_epochs, n_ch, n_times)

    # average over trials → ch × t
    band_ts = ERDS_pct.mean(axis=0)                 # (n_ch, n_times)

    ev = mne.EvokedArray(
        band_ts, ep.info, tmin=t[0],
        comment=f"Hilbert {band} (% power vs baseline)"
    )
    return ev, ep

def erds_topo_from_hilbert(epochs, band, baseline=(-1.0, 0.0)):
    lo, hi = band
    ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)                  # trials × ch × t (amplitude)
    A = ep.get_data()
    A_db = 20*np.log10(np.maximum(A, 1e-12))         # amplitude → dB (≈ power dB)
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    A_db -= A_db[..., bmask].mean(axis=-1, keepdims=True)   # per-trial baseline (dB)

    band_ts = A_db.mean(axis=0)                      # avg over trials → ch × t
    ev = mne.EvokedArray(band_ts, ep.info, tmin=t[0],
                         comment=f"Hilbert {band} (amp-dB)")
    return ev, ep                                    # keep as you had

# single-channel ERD/ERS time-course (same normalization as above)
def erds_line_from_hilbert(epochs, band, ch_name, baseline=(-1.0, 0.0)):
    lo, hi = band
    ep = epochs.copy().pick(ch_name).filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)                  # amplitude
    A = ep.get_data()                                # shape: (n_epochs, 1, n_times)
    A = np.squeeze(A, axis=1)                        # → (n_epochs, n_times)
    A_db = 20*np.log10(np.maximum(A, 1e-12))         # amplitude → dB
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])
    A_db -= A_db[:, bmask].mean(axis=1, keepdims=True)      # per-trial baseline (dB)

    mean_tc = A_db.mean(axis=0)
    sem_tc  = A_db.std(axis=0, ddof=1) / np.sqrt(A_db.shape[0])
    return t, mean_tc, sem_tc

def erds_line_percent_from_hilbert(epochs, band, ch_name, baseline=(-1.0, 0.0)):
    lo, hi = band
    ep = epochs.copy().pick(ch_name).filter(lo, hi, method='fir', fir_design='firwin')
    ep.apply_hilbert(envelope=True)          # amplitude envelope
    A = ep.get_data()                        # (n_epochs, 1, n_times)
    A = np.squeeze(A, axis=1)                # (n_epochs, n_times)

    P = A**2                                 # power from amplitude
    t = ep.times
    bmask = (t >= baseline[0]) & (t <= baseline[1])

    # per-trial baseline (mean power in baseline window)
    B = P[:, bmask].mean(axis=1, keepdims=True)
    B = np.maximum(B, 1e-20)                 # protect against divide-by-zero

    ERDS_pct = 100.0 * (P - B) / B           # percent change relative to baseline

    mean_tc = ERDS_pct.mean(axis=0)
    sem_tc  = ERDS_pct.std(axis=0, ddof=1) / np.sqrt(ERDS_pct.shape[0])
    return t, mean_tc, sem_tc

# ----------------- usage -----------------
times_topo = np.linspace(-1.0, 6.0, 11)  # for topomap snapshots
sigma = 50.0                             # for line smoothing

for event in event_id:                   # e.g., {'left':1, 'right':2}
    print(event)

    # Topomaps (dB)
    ev_mu,   ep_mu   = erds_topo_from_hilbert(epochs[event], band=(8, 13), baseline=baseline)
    ev_beta, ep_beta = erds_topo_from_hilbert(epochs[event], band=(13, 30), baseline=baseline)

    print('Mu Band')
    ev_mu.plot_topomap(times=times_topo, ch_type=ch_type,
                       units=dict(eeg='dB'), scalings=dict(eeg=1), time_unit='s')
    print('Beta Band')
    ev_beta.plot_topomap(times=times_topo, ch_type=ch_type,
                         units=dict(eeg='dB'), scalings=dict(eeg=1), time_unit='s')

    # Topomaps (% change)
    ev_mu_pct,   ep_mu_pct   = erds_topo_percent_from_hilbert(epochs[event], band=(8, 13), baseline=baseline)
    ev_beta_pct, ep_beta_pct = erds_topo_percent_from_hilbert(epochs[event], band=(13, 30), baseline=baseline)

    print('Mu Band')
    ev_mu_pct.plot_topomap(times=times_topo, ch_type=ch_type,
                       units=dict(eeg='%'), scalings=dict(eeg=1), time_unit='s')
    print('Beta Band')
    ev_beta_pct.plot_topomap(times=times_topo, ch_type=ch_type,
                         units=dict(eeg='%'), scalings=dict(eeg=1), time_unit='s')


    for ch in target_chs:
        # single-channel ERD/ERS line (mean ± SEM) for mu & beta
        t_mu,   mu_mean,   mu_sem   = erds_line_from_hilbert(epochs[event], (8, 13),  ch, baseline)
        t_be,   be_mean,   be_sem   = erds_line_from_hilbert(epochs[event], (13, 30), ch, baseline)

        mu_mean = gaussian_filter1d(mu_mean, sigma=sigma)
        be_mean = gaussian_filter1d(be_mean, sigma=sigma)
        plt.figure()
        plt.title(f"{event} — {ch} (Hilbert, dB rel. baseline)")
        plt.plot(t_mu, mu_mean,  lw=2, label='μ: 8–13 Hz')
        plt.fill_between(t_mu, mu_mean-mu_sem, mu_mean+mu_sem, alpha=0.2, linewidth=0)
        plt.plot(t_be, be_mean,  lw=2, label='β: 13–30 Hz')
        plt.fill_between(t_be, be_mean-be_sem, be_mean+be_sem, alpha=0.2, linewidth=0)
        plt.axhline(0, ls='--', lw=0.8)
        plt.axvline(0, ls='--', lw=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("ERD/ERS (dB rel. baseline)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Percent-change plot (μ and β on one figure)
        t_mu_p, mu_mean_p, mu_sem_p = erds_line_percent_from_hilbert(
            epochs[event], (8, 13), ch, baseline
        )
        t_be_p, be_mean_p, be_sem_p = erds_line_percent_from_hilbert(
            epochs[event], (13, 30), ch, baseline
        )

        mu_mean_p = gaussian_filter1d(mu_mean_p, sigma=sigma)
        be_mean_p = gaussian_filter1d(be_mean_p, sigma=sigma)

        plt.figure()
        plt.title(f"{event} — {ch} (Hilbert, % change rel. baseline)")
        plt.plot(t_mu_p, mu_mean_p, lw=2, label='μ: 8–13 Hz')
        plt.fill_between(t_mu_p, mu_mean_p - mu_sem_p, mu_mean_p + mu_sem_p, alpha=0.2, linewidth=0)
        plt.plot(t_be_p, be_mean_p, lw=2, label='β: 13–30 Hz')
        plt.fill_between(t_be_p, be_mean_p - be_sem_p, be_mean_p + be_sem_p, alpha=0.2, linewidth=0)
        plt.axhline(0, ls='--', lw=0.8)
        plt.axvline(0, ls='--', lw=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("ERD/ERS (% change)")
        plt.legend()
        plt.tight_layout()
        plt.show()

def calc_erds(epochs, channels=None, sigma=2, rest_period=(-1, 0)):

    # Get the sampling frequency
    sfreq = epochs.info['sfreq']

    # Get the channel indices
    if channels is not None:
        ch_idx = [epochs.ch_names.index(ch) for ch in channels]
    else:
        ch_idx = np.arange(len(epochs.ch_names))

    # Get the event names
    events = list(epochs.event_id.keys())

    # Initialize the dictionary for storing the ERD/ERS curves
    erds_dict = {}

    # Get the reference period limits
    rmin = rest_period[0] * sfreq
    rmax = rest_period[1] * sfreq

    # If the epoching window start from before the cue was shown
    if epochs.tmin < 0:
        # Shift both reference period limits accordingly
        rmin += -epochs.tmin * sfreq
        rmax += -epochs.tmin * sfreq

    # Convert the limits to integer for slicing
    rmin = int(rmin)
    rmax = int(rmax)

    # Iterate over the events
    for event in events:

        # Get the trials data for the relevant channels
        epochs_arr = epochs[event].copy().get_data()[:, ch_idx, :]

        # Initialize an empty array for the band-powers
        epochs_bp = np.zeros(epochs_arr.shape)

        # Iterate over the trials
        for i, trial in enumerate(epochs_arr):
            # Iterate over the channels
            for ch in range(len(ch_idx)):
                # Square the signal to get an estimate of the band-powers
                epochs_bp[i, ch, :] = trial[ch] ** 2

        # Average the band-powers over trials
        A = np.mean(epochs_bp, axis=0)

        # Get the reference period
        R = np.mean(A[:, rmin:rmax], axis=1).reshape(-1, 1)

        # Compute the ERD/ERS
        erds = (A - R) / R * 100


        # Smoothen the ERD/ERS curve
        erds = gaussian_filter1d(erds, sigma=sigma)

        # Append the curves to the corresponding events
        erds_dict[event] = erds

    return erds_dict


def plot_erds(
    erds_dict,
    epochs,
    channels=None,
    events=None,
    view='channel',
    title_suffix='',
):

    if view not in ['task', 'channel']:
        raise ValueError(
            "Please provide a valid view parameter. Valid view parameters are: 'task' and 'channel'."
        )

    if view == 'channel' and channels is None:
        raise ValueError("Please provide the channels for the 'channel' view.")

    # The values for plotting
    tmin = epochs.tmin
    tmax = epochs.tmax
    flow = float(epochs.info['highpass'])
    fhigh = float(epochs.info['lowpass'])
    events = list(epochs.event_id.keys())

    # Get the number of samples in the ERD/ERS curves
    n_chs = list(erds_dict.values())[0].shape[0]
    n_samples = list(erds_dict.values())[0].shape[1]

    x = np.linspace(tmin, tmax, n_samples)

    # Initialize the plot
    n_rows = n_chs if view == 'channel' else len(events)
    fig, axs = plt.subplots(n_rows, 1)
    axs = axs.ravel()

    if view == 'task':
        for ax, (event_name, erds_arr) in zip(axs, erds_dict.items()):

            if 'left' in event_name:
                event_name = 'Left\nMI'
            if 'right' in event_name:
                event_name = 'Right\nMI'

            if channels is not None:
                ax.plot(x, erds_arr.T, lw=2)
                ax.legend(channels)
                title = f'ERD/ERS Curves{title_suffix}\n({flow}-{fhigh} Hz BP)'
            else:
                ax.plot(x, np.mean(erds_arr, axis=0), lw=2, color='navy')
                title = f'ERD/ERS Curves Averaged Over Available Channels{title_suffix}\n({flow}-{fhigh} Hz BP)'

            if tmin <= 0:
                ax.axvline(0, color='gray', lw=2)
            ax.axhline(0, color='gray', ls='--')
            ax.set_xticks(np.arange(tmin, tmax + 0.1, 0.5))
            if ax != axs[-1]:
                ax.set_xticklabels([])
            ax.grid()
            ax_twin = ax.twinx()
            ax_twin.set_ylabel(event_name, rotation=0, labelpad=17)
            ax_twin.set_yticklabels([])

    elif view == 'channel':
        for i in range(len(events)):
            if 'left' in events[i]:
                events[i] = 'Left MI'
            if 'right' in events[i]:
                events[i] = 'Right MI'

        for i, ax in enumerate(axs):

            for erds_arr in erds_dict.values():

                if channels is not None:
                    ax.plot(x, erds_arr[i], lw=2)
                    title = f'ERD/ERS Curves{title_suffix}\n({flow}-{fhigh} Hz BP)'
                else:
                    ax.plot(x, np.mean(erds_arr, axis=0), lw=2, color='navy')
                    title = f'ERD/ERS Curves Averaged Over Available Channels{title_suffix}\n({flow}-{fhigh} Hz BP)'

            ax.legend(events)

            if tmin <= 0:
                ax.axvline(0, color='gray', lw=2)
            ax.axhline(0, color='gray', ls='--')
            ax.set_xticks(np.arange(tmin, tmax + 0.1, 0.5))
            if ax != axs[-1]:
                ax.set_xticklabels([])
            ax.grid()
            ax_twin = ax.twinx()
            ax_twin.set_ylabel(channels[i], rotation=0, labelpad=10)
            ax_twin.set_yticklabels([])

    ax = fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axes
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.tick_params(
        axis='x', which='both', top=False, bottom=False, left=False, right=False
    )
    ax.tick_params(
        axis='y', which='both', top=False, bottom=False, left=False, right=False
    )
    ax.grid(False)
    ax.set_xlabel('Time Relative to the Cue (in s)')
    ax.set_ylabel('Relative Band Power (in %)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Set the sigma for the Gaussian smoothing
sigma = 50
rest_period = (-1.5, -0.5)
channels = ['C3', 'C4']

lo, hi = (8, 13)  # Mu band
ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')

# Calculate and plot the ERD/ERS curves
erds_base = calc_erds(
    ep,
    channels=channels,
    sigma=sigma,
    rest_period=rest_period,
)
plot_erds(erds_base, ep, channels=channels, view='task', title_suffix=' (Mu Band)')

lo, hi = (13, 30)  # Beta band
ep = epochs.copy().filter(lo, hi, method='fir', fir_design='firwin')

# Calculate and plot the ERD/ERS curves
erds_base = calc_erds(
    ep,
    channels=channels,
    sigma=sigma,
    rest_period=rest_period,
)
plot_erds(erds_base, ep, channels=channels, view='task', title_suffix=' (Beta Band)')
