#%% import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import mne
import glob
import os

#%% Define data directory
data_path = 'D:/user/Files_without_backup/MSc Project/Data/sourcedata/rawdata/S001'
edf_files = sorted(glob.glob(os.path.join(data_path, '*.edf')))

#%% Load data
# Load & concatenate raw EDFs with annotations
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

set_montage(baseline)
set_montage(raw)

#%% Preprocessing
# band-pass filter
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')
baseline.filter(l_freq=1., h_freq=40., fir_design='firwin')


#%% Helper: Real-time data generator
def realtime_data_generator(raw: mne.io.Raw) -> np.ndarray:
    """
    Generator that yields one sample per channel at the raw's sampling rate,
    simulating real-time data input.

    Yields:
    1D numpy array of shape (n_channels,) corresponding to each new sample.
    """
    sfreq = int(raw.info['sfreq'])
    data = raw.get_data()  # shape: n_channels x n_times
    n_times = data.shape[1]
    # Iterate through timepoints
    for t in range(n_times):
        yield data[:, t]

a = realtime_data_generator(raw)
print(a)