
# %% Imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from mne.stats import permutation_cluster_1samp_test as pcluster_test
from matplotlib.colors import TwoSlopeNorm

# Make figures a bit larger
plt.rcParams['figure.figsize'] = (9, 4)
mne.set_log_level('WARNING')


# %% Config
source = "eegbci"   # "eegbci" or "csv"

# EEGBCI settings
subject = 1
runs = [4, 8, 12] # [4, 8, 12]  # motor imagery, left vs right

# CSV settings (used if source == "csv")
csv_path = r"D:/path/to/your_file.csv"   # <- change me
time_col = "timestamp"
type_col = "type"          # values: "EEG" or "Marker"
eeg_value = "EEG"
marker_value = "Marker"
marker_col = "marker"      # integer codes
channel_cols = None        # e.g. [f"ch{i}" for i in range(1,17)]
event_code_map = {1: "left", 2: "right"}  # CSV marker code -> name
csv_voltage_unit = "uV"    # "uV" | "mV" | "V"

# Preprocessing
notch_freq = 50.0
l_freq, h_freq = 0.5, 40.0
resample_sfreq = 250.0     # set None to keep native
reref = "average"          # "average" or list of ref channel names

# Epoching
tmin, tmax = -1.0, 4.0
baseline = (-1.0, 0.0)

# TFR (Morlet)
freqs = np.arange(2, 36)   # 2–35 Hz
n_cycles = 5.0
baseline_mode = "percent"  # % change

# ERD metrics
mu_band = (8., 13.)
beta_band = (13., 30.)
erd_onset_threshold = -20.0   # % drop (negative)
mi_window = (1.0, 4.0)        # seconds
sensorimotor_chs = ["C3", "C4", "Cz"]

# Output directory
OUT = Path("outputs"); OUT.mkdir(exist_ok=True, parents=True)
print("Configured. Source =", source)


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
    raw.annotations.rename(dict(T1="left", T2="right"))
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    event_id = dict(left=1, right=2)  # map event IDs to tasks

elif source.lower() == "csv":
    df = pd.read_csv(csv_path, encoding="latin1")
    eeg_df = df[df[type_col] == eeg_value].copy()
    mrk_df = df[df[type_col] == marker_value].copy()
    # channels
    ch_cols = channel_cols
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
    info = mne.create_info(ch_names=ch_cols, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
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
else:
    raise ValueError("source must be 'eegbci' or 'csv'")

print(raw)
print("event_id:", event_id)


# %% Preprocess + QC
raw_f = raw.copy()

picks = mne.pick_types(raw_f.info, eeg=True, exclude='bads')
epochs = mne.Epochs(raw_f, event_id=event_id, tmin=tmin-0.5, tmax=tmax+0.5,
                    baseline=None, picks=picks, preload=True, detrend=1)

# epochs = mne.Epochs(
#     raw,
#     event_id=["left", "right"],
#     tmin=tmin - 0.5,
#     tmax=tmax + 0.5,
#     picks=("C3", "C4", "Cz"),
#     baseline=None,
#     preload=True,
# )

# %% Multitaper
freqs = np.arange(2, 36)  # frequencies from 2-35Hz
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = (-1, 0)  # baseline interval (in s)
cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

kwargs = dict(
    n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask"
)  # for cluster test

tfr = epochs.compute_tfr(method="multitaper", freqs=freqs,
    n_cycles=freqs, use_fft=True, return_itc=False,
    average=False, decim=2)
tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
for event in event_id:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    
    # pick sensorimotor channels (fallback to first 3 if any are missing)
    sel_chs = [ch for ch in sensorimotor_chs if ch in epochs.ch_names]
    if not sel_chs:
        sel_chs = epochs.ch_names[:3]
    ch_inds = mne.pick_channels(epochs.ch_names, include=sel_chs)

    # figure with one panel per selected channel + colorbar
    fig, axes = plt.subplots(
        1, len(ch_inds) + 1, figsize=(12, 4),
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
            cmap="RdBu",
            cnorm=cnorm,
            axes=ax,
            colorbar=False,
            show=False,
            mask=mask,
            mask_style="mask",
        )

        ax.set_title(epochs.ch_names[ch_idx], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if ch_idx != 0:
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
    fig.suptitle(f"ERDS ({event})")
    plt.show()