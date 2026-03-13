#!/usr/bin/env python3
import csv
import signal
import sys
from collections import deque
from datetime import datetime
import time
from pylsl import StreamInlet, resolve_byprop, local_clock

# ----------------------
# CONFIGURATION
# ----------------------
EEG_STREAM_NAME    = 'EEG'
MARKER_STREAM_NAME = 'GameMarkers'

# Create timestamped filename, e.g. "20250803_142530_eeg_with_markers.csv"
now = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = f"{now}_eeg_with_markers.csv"

# How many EEG samples to pull at once
EEG_CHUNK_SIZE     = 32

# ----------------------
# GRACEFUL SHUTDOWN
# ----------------------
running = True
def handle_exit(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# ----------------------
# RESOLVE STREAMS
# ----------------------
print("Looking for EEG stream...")
eegs = resolve_byprop('name', EEG_STREAM_NAME, timeout=5)
if not eegs:
    print(f"ERROR: Couldn’t find an LSL stream named '{EEG_STREAM_NAME}'.")
    sys.exit(1)
eeg_inlet = StreamInlet(eegs[0], max_buflen=10)
time.sleep(1.0)  
eeg_offset = eeg_inlet.time_correction()
print("EEG offset:", eeg_offset) 

print("Looking for marker stream...")
marks = resolve_byprop('name', MARKER_STREAM_NAME, timeout=5)
if not marks:
    print(f"ERROR: Couldn’t find an LSL stream named '{MARKER_STREAM_NAME}'.")
    sys.exit(1)
marker_inlet = StreamInlet(marks[0], max_buflen=10)
time.sleep(1.0)  
marker_offset = marker_inlet.time_correction()
print("Marker offset:", marker_offset)

# ----------------------
# PREPARE OUTPUT FILE
# ----------------------
# Columns: timestamp, type, ch1…ch16, marker
header = ['timestamp', 'type'] + [f'ch{i}' for i in range(1, eeg_inlet.channel_count + 1)] + ['marker']
csvfile = open(OUTPUT_CSV, 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(header)

# ----------------------
# MAIN LOOP
# ----------------------
print(f"Recording… output will be saved to '{OUTPUT_CSV}'. Press Ctrl+C to stop.")

while running:
    # 1) Pull any new markers
    m_samples, m_ts = marker_inlet.pull_chunk(timeout=0.0, max_samples=10)  # non-blocking: timeout=0.0

    # adjust them into your local base:
    m_ts = [ts + marker_offset for ts in m_ts]

    for sample, ts in zip(m_samples, m_ts):
        val= sample[0]  # assuming marker is the first channel
        print(f"╞═ Received marker {val} @ {ts:.6f}")
        row = [f"{ts:.6f}", 'Marker'] \
            + [''] * eeg_inlet.channel_count \
            + [str(val)]
        writer.writerow(row)

    # 2) Pull EEG data in chunks
    eeg_samples, eeg_ts = eeg_inlet.pull_chunk(  
        timeout=0.0,
        max_samples=EEG_CHUNK_SIZE
    )  # timeout=0.0, the call never blocks: if no new samples are in LSL’s buffer, it returns ([], [])

    # adjust them into your local base:
    eeg_ts = [ts + eeg_offset for ts in eeg_ts]

    if eeg_samples:
        for sample, ts in zip(eeg_samples, eeg_ts):
            # Write EEG row
            row = [f"{ts:.6f}", 'EEG'] + [f"{v:.3f}" for v in sample] + ['']
            writer.writerow(row)

csvfile.close()
print(f"\nDone — data saved to '{OUTPUT_CSV}'")
