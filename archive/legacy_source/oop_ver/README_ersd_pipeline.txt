ERS/ERD pipeline (add-on after preprocessing)

Files:
  - ersd_pipeline.py           Reusable functions for 4 ERS/ERD methods + onsets + plotting
  - ERSD_after_preproc.ipynb   Notebook section to run after your own preprocessing/epoching cells

Assumptions:
  - You already produced an MNE `epochs` object covering -4 to 6 s around the cue (t=0).
  - epochs.event_id contains at least two event types (e.g., left/right). The code will auto-split.

What you get:
  - Per-trial and averaged ERS/ERD time-courses for Hilbert, Morlet, Multitaper, and STFT/Welch
  - Both percent and dB normalization (trial-wise baseline -2 to -1 s)
  - Onset detection with 4 methods (SD threshold, t-test+FDR, CUSUM-like, and 50% peak latency)
  - Identical plotting style across methods; baseline shaded; C3/C4 highlighted
  - ERD/ERS topomaps (percent) over user windows (defaults: ERD 0–2 s, ERS 2–4 s)
  - A kept multitaper TFR plot for your channels of interest

Typical parameters (editable in the notebook):
  - Morlet: freqs 5–40 Hz by 1 Hz; n_cycles linspace 3→7 (frequency-dependent)
  - Multitaper: n_cycles=5, time_bandwidth=4.0 (≈7 DPSS tapers)
  - STFT/Welch: 0.5 s Hamming window, 50 ms step (≈90% overlap at 1 kHz; scales with fs)
  - Smoothing: Gaussian FWHM 0.1 s (set to 0 to disable)

Outputs:
  - /mnt/data/ersd_timeseries.csv   long-form: method, unit, band, condition, channel, trial, time_s, value
  - /mnt/data/ersd_onsets.csv       onset table with all detectors per condition/band/channel

Notes:
  - Topomaps use Hilbert % by default (fast & robust); switch to TFR-derived if desired.
  - If you need heavier stats (e.g., cluster-based permutation), add it on top of the t-test/FDR scaffold.
  - For reproducibility, keep your epochs baseline=None during epoching and do baseline here per trial.
