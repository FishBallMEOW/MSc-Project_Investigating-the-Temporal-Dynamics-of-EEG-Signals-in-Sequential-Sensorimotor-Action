#%% import library
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import welch
import numpy as np
import mne
import glob
import os

#%% Helper: Set montage
def set_montage(edf_raw):
    # Load the standard 10-10 montage for reference
    montage = mne.channels.make_standard_montage('standard_1020')
    montage_chs = montage.ch_names
    # Build a renaming dictionary so that "Fc5." → "FC5", "Fcz." → "FCz", etc.
    mapping = {}
    for old_name in edf_raw.info['ch_names']:
        stripped = old_name.replace('.', '')  # remove all dots, e.g. "Fcz." → "Fcz"
        match = None
        for m_ch in montage_chs:
            if m_ch.lower() == stripped.lower():
                match = m_ch
                break
        if match:
            mapping[old_name] = match
        else:
            # If no match is found, leave the name unchanged.
            mapping[old_name] = old_name
    # Apply the renaming so that channel names match exactly what the montage expects
    edf_raw.rename_channels(mapping)
    # Now attach the standard 10-10 montage
    edf_raw.set_montage(montage)
    return edf_raw

#%% Helper: Real-time data generator
def realtime_data_generator(raw: mne.io.Raw):
    """
    Generator that yields one sample per channel from the raw data,
    simulating real-time streaming.

    Yields:
    1D numpy array of shape (n_channels,) corresponding to each new sample.
    """
    data = raw.get_data()  # shape: n_channels x n_times
    n_times = data.shape[1]
    for t in range(n_times):
        yield data[:, t]

# a = realtime_data_generator(raw)
# print(a)
# for idx, sample in enumerate(a):
#     print(idx, sample)

# %% Helper: compute baseline and band power
def compute_bandpower(data_segment: np.ndarray, sfreq: float, band: tuple, window_samples: int) -> float:
    """
    Compute band power for a single-channel data segment using Welch's method.

    Parameters:
    -------------
    data_segment : 1D numpy array
        Time series data for one channel.
    sfreq : float
        Sampling frequency in Hz.
    band : tuple
        Frequency band (low_freq, high_freq) in Hz.
    window_samples : int
        Number of samples in the Welch window.

    Returns:
    --------
    float
        Integrated power within the specified band.
    """
    freqs, psd = welch(data_segment, fs=sfreq, nperseg=window_samples)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapezoid(psd[mask], freqs[mask])

def compute_baseline_power(baseline: mne.io.Raw, band: tuple, window_sec: float) -> np.ndarray:
    """
    Compute baseline bandpower per channel over the entire baseline recording.

    Parameters:
    baseline : mne.io.Raw
        Baseline data.
    band : tuple
        Frequency band (low, high) in Hz.
    window_sec : float
        Window length in seconds for power estimation.

    Returns:
    1D numpy array of shape (n_channels,) with baseline bandpower.
    """
    sfreq = baseline.info['sfreq']
    window_samples = int(window_sec * sfreq)
    data = baseline.get_data()
    n_ch = data.shape[0]
    baseline_power = np.zeros(n_ch)
    for ch in range(n_ch):
        baseline_power[ch] = compute_bandpower(data[ch], sfreq, band, window_samples)
    return baseline_power

#%% Helper: Compute ERS/ERD
def compute_ers_erd(data_seg: np.ndarray, baseline_power: np.ndarray, band: tuple, window_len: float = 2.0, sfreq: float = 160.0):
    current_power = np.array([
                compute_bandpower(data_seg[ch], sfreq, band, window_samples)
                for ch in range(data_seg.shape[0])
            ])
    percent_change = (current_power - baseline_power) / baseline_power * 100
    return percent_change

#%% Helper: Plot the topo map        
def plot_ers_erd_topomap(ers_erd_stream: np.ndarray,
                         raw: mne.io.Raw,
                         time_idx: int,
                         times: np.ndarray = None,
                         vmin: float = None,
                         vmax: float = None,
                         show: bool = True):
    """
    Plot a 2D topographic map of ERS/ERD values at a given time index.

    Parameters:
    -------------
    ers_erd_stream : 2D numpy array
        ERS/ERD values (n_times x n_channels).
    raw : mne.io.Raw
        Raw object with channel locations (montage) set.
    time_idx : int
        Index of the time point to plot.
    times : 1D array | None
        Optional array of time stamps corresponding to ERS/ERD rows.
    vmin, vmax : float | None
        Min/max percent-change for color scaling.
    show : bool
        Whether to display the plot immediately.
    """
    # select data to plot
    data = ers_erd_stream[time_idx]
    # determine title
    if times is not None:
        title = f"ERS/ERD Topomap at t={times[time_idx]:.2f}s"
    else:
        title = f"ERS/ERD Topomap at index {time_idx}"
    # plot
    fig, ax = plt.subplots()
    mne.viz.plot_topomap(data, raw.info, axes=ax, show=False, sensors=True)
    ax.set_title(title)
    if show:
        plt.show()
    return fig

#%% Main
# Define data directory
data_path = 'D:/user/Files_without_backup/MSc_Project/MSc-Project_Investigating-the-Temporal-Dynamics-of-EEG-Signals-in-Sequential-Sensorimotor-Action/Data_from_dataset'
edf_files = sorted(glob.glob(os.path.join(data_path, '*.edf')))

# Load & concatenate EDFs with annotations
baselines = []
raws = []
for edf_path in edf_files:
    if ("R01" in edf_path) or ("R02" in edf_path):
        baseline = mne.io.read_raw_edf(edf_path, preload=True)
        baselines.append(baseline)
    else:
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        raws.append(raw)
baseline = mne.concatenate_raws(baselines)
raw = mne.concatenate_raws(raws)

# Set montage
set_montage(baseline)
set_montage(raw)

# Preprocessing
# band-pass filter
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')
baseline.filter(l_freq=1., h_freq=40., fir_design='firwin')

# ERS/ERD
band = (8, 12)  # alpha band
window_len = 2.0
sfreq = raw.info['sfreq']
window_samples = int(window_len * sfreq)

buffer = np.zeros((raw.info['nchan'], window_samples))
ers_erd_stream = []

gen = realtime_data_generator(raw)
baseline_pw = compute_baseline_power(baseline, band, window_len)

for idx, sample in enumerate(gen):
    buffer = np.roll(buffer, -1, axis=1)
    buffer[:, -1] = sample
    if idx >= window_samples:    
        ers_erd_stream.append(compute_ers_erd(buffer, baseline_pw, band, window_len, sfreq))
    else: 
        ers_erd_stream.append(np.full(baseline_pw.shape, np.nan))

#%% Animation
# Convert list to array 
ers_erd_stream = np.array(ers_erd_stream)  # shape: (n_windows, n_channels)
n_windows = ers_erd_stream.shape[0]

ann     = raw.annotations
onsets  = np.array(ann.onset)            # in seconds
descs   = np.array(ann.description)      # e.g. 'T0','TASK1T2', etc.
tol     = 10.0 / sfreq                    # tolerance
current_event = ['']  # list so it’s mutable from inside update()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(1, 1, 1)

im, _ = mne.viz.plot_topomap(
        ers_erd_stream[0],
        raw.info,
        axes=ax,
        show=False,
        sensors=True
    )

# attach a single colorbar
cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                    fraction=0.046, pad=0.04)
cbar.set_label("ERS/ERD (% change)")

# Define update function: clear axes and replot topomap each time
def update(frame_idx):
    ax.clear()
    t = frame_idx / sfreq
    
    data = ers_erd_stream[frame_idx]
    im, _ = mne.viz.plot_topomap(
        data,
        raw.info,
        axes=ax,
        show=False,
        sensors=True
    )
    # find any annotations whose onset is within ±tol of t
    hits = np.abs(onsets - t) <= tol
    if hits.any():
        current_event[0] = descs[hits][0]        # first matching event
    
    if current_event[0]:
        ax.set_title(f"t = {t:.2f}s — {current_event[0]}")

    # return list of artists (collections) for FuncAnimation
    return ax.collections

# Create animation: step every 100 samples, 200ms interval, no blitting
ani = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, n_windows, 100),
    interval=200,
    blit=False
)

ani.save('S001_R04.gif', writer='PillowWriter', fps=20)

plt.show()
# %%
