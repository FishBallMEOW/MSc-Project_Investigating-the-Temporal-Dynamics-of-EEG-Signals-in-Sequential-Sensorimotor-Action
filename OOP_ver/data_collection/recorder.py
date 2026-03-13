"""
Module defining a class to record EEG, marker, and user‐setup streams to CSV via LSL.
"""
import csv
import signal
import sys
import time
from datetime import datetime
from pylsl import StreamInlet, resolve_byprop
from pathlib import Path
import threading

class LSLDataRecorder:
    """
    Records EEG and marker streams from LSL into a timestamped CSV file.

    Usage:
        recorder = LSLDataRecorder(
            eeg_name='EEG',
            marker_name='GameMarkers',
            eeg_chunk_size=32
        )
        recorder.start()
        # ... later, to stop:
        recorder.stop()
    """
    def __init__(self,eeg_name='EEG',marker_name='GameMarkers',userinfo_name='UserInfo',usersettings_name='UserSettings',eeg_chunk_size=32):
        self.eeg_name = eeg_name
        self.marker_name = marker_name
        self.userinfo_name = userinfo_name
        self.usersettings_name = usersettings_name
        self.eeg_chunk_size = eeg_chunk_size
        self.running = False

        # for buffering & flushing
        self._buffer = []
        self._buffer_size = self.eeg_chunk_size * 10

        # for real-time plotting
        self._plot_buffer = []                # will hold (timestamp, sample) tuples
        self._plot_buffer_lock = threading.Lock()

        # metadata placeholders
        self.metadata_received = False
        self.settings_json = None
        self.userinfo_json = None
        self.us_ts = None
        self.ui_ts = None

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        # allow graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, sig, frame):
        self.running = False

    def _resolve_streams(self):
        # EEG
        print(f"Looking for EEG stream '{self.eeg_name}'…")
        eegs = resolve_byprop('name', self.eeg_name, timeout=5)
        if not eegs:
            print(f"ERROR: Couldn’t find EEG stream '{self.eeg_name}'")
            sys.exit(1)
        self.eeg_inlet = StreamInlet(eegs[0], max_buflen=10)
        time.sleep(0.1)
        self.eeg_offset = self.eeg_inlet.time_correction()
        print(f"EEG offset: {self.eeg_offset}")

        # User Settings
        print(f"Looking for UserSettings stream '{self.usersettings_name}'…")
        uss = resolve_byprop('name', self.usersettings_name, timeout=5)
        if not uss:
            print(f"ERROR: Could not find UserSettings stream '{self.usersettings_name}'")
            sys.exit(1)
        self.usersettings_inlet = StreamInlet(uss[0], max_buflen=5)
        time.sleep(0.1)
        self.usersettings_offset = self.usersettings_inlet.time_correction()
        print(f"UserSettings offset: {self.usersettings_offset}")

        # User Info
        print(f"Looking for UserInfo stream '{self.userinfo_name}'…")
        uis = resolve_byprop('name', self.userinfo_name, timeout=5)
        if not uis:
            print(f"ERROR: Could not find UserInfo stream '{self.userinfo_name}'")
            sys.exit(1)
        self.userinfo_inlet = StreamInlet(uis[0], max_buflen=5)
        time.sleep(0.1)
        self.userinfo_offset = self.userinfo_inlet.time_correction()
        print(f"UserInfo offset: {self.userinfo_offset}")

        # Marker
        print(f"Looking for marker stream '{self.marker_name}'…")
        marks = resolve_byprop('name', self.marker_name, timeout=5)
        if not marks:
            print(f"ERROR: Could not find marker stream '{self.marker_name}'")
            sys.exit(1)
        self.marker_inlet = StreamInlet(marks[0], max_buflen=10)
        time.sleep(0.1)
        self.marker_offset = self.marker_inlet.time_correction()
        print(f"Marker offset: {self.marker_offset}")

    def _wait_for_metadata(self):
        # Block until both UserSettings and UserInfo samples are received
        print("Waiting for UserSettings sample…")
        # Keep pulling until we get a valid timestamp
        while True:
            us_sample, us_ts = self.usersettings_inlet.pull_sample(timeout=1.0)
            if us_ts and us_sample:
                break
        us_ts += self.usersettings_offset
        self.settings_json = us_sample[0]
        self.us_ts = us_ts
        print(f"UserSettings received at {us_ts:.6f}: {self.settings_json}")

        print("Waiting for UserInfo sample…")
        while True:
            ui_sample, ui_ts = self.userinfo_inlet.pull_sample(timeout=1.0)
            if ui_ts and ui_sample:
                break
        ui_ts += self.userinfo_offset
        self.userinfo_json = ui_sample[0]
        self.ui_ts = ui_ts
        print(f"UserInfo received at {ui_ts:.6f}: {self.userinfo_json}")


    def _prepare_output(self):
        # 1) create file & writer
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = Path("Data") / now
        folder.mkdir(parents=True, exist_ok=True)
        self.output_csv = f"Data/{now}/{now}_eeg_with_markers.csv"
        # 'x' -> fail if file exists; pick a unique name, and write UTF-8
        self._csvfile = open(self.output_csv, 'x', newline='', encoding='utf-8')
        self._writer = csv.writer(self._csvfile)

        # 2) header: add two new columns for settings & info
        header = (
            ['timestamp', 'type']
            + [f'ch{i}' for i in range(1, self.eeg_inlet.channel_count + 1)]
            + ['marker', 'user_settings', 'user_info']
        )
        self._writer.writerow(header)

        # First row: metadata
        first_row = (
            [f"{self.us_ts:.6f}", 'Setup']
            + [''] * self.eeg_inlet.channel_count
            + ['']
            + [self.settings_json, self.userinfo_json]
        )
        self._writer.writerow(first_row)
        print(f"Recording… data will be saved to '{self.output_csv}'. Press Ctrl+C to stop.")

    def _collect_loop(self):
        # Continuously pull data and buffer for plotting/writing
        while self.running:
            # Pull markers
            m_samples, m_ts = self.marker_inlet.pull_chunk(timeout=0.0, max_samples=10)
            m_ts = [ts + self.marker_offset for ts in m_ts]

            # Pull EEG
            eeg_samples, eeg_ts = self.eeg_inlet.pull_chunk(
                timeout=0.0,
                max_samples=self.eeg_chunk_size
            )
            eeg_ts = [ts + self.eeg_offset for ts in eeg_ts]

            # Stash for plotting
            with self._plot_buffer_lock:
                for sample, ts in zip(eeg_samples, eeg_ts):
                    self._plot_buffer.append((ts, sample))

            # Buffer for CSV writing
            for (sample,), ts in zip(m_samples, m_ts):
                self._buffer.append((ts, 'Marker', sample))
            for sample, ts in zip(eeg_samples, eeg_ts):
                self._buffer.append((ts, 'EEG', sample))

            # Flush if enough and metadata is set
            if self.metadata_received and len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

            # Small pause to yield
            time.sleep(0.001)

    def start(self):
        try:
            self.running = True
            self._resolve_streams()

            # Start data collection thread immediately
            collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
            collect_thread.start()

            # Wait for user settings and info before writing to CSV
            self._wait_for_metadata()
            self._prepare_output()
            self.metadata_received = True
            
            # After metadata, flush any pre-buffered data
            if self._buffer:
                self._flush_buffer()

            # Keep running until stopped
            while self.running:
                time.sleep(0.1)

            # Ensure collection thread exits
            collect_thread.join()

            # Flush remaining data
            if self._buffer:
                self._flush_buffer()
        finally:
            self._cleanup()

    def stop(self):
        """Stop the recording loop gracefully."""
        self.running = False

    def _flush_buffer(self):
        # Sort by timestamp then write to CSV
        self._buffer.sort(key=lambda e: e[0])
        for ts, kind, payload in self._buffer:
            ts = float(ts)  # guarantees numeric
            if kind == 'Marker':
                row = (
                    [f"{ts:.6f}", 'Marker']
                    + [''] * self.eeg_inlet.channel_count
                    + [str(payload), '', '']
                )
            else:  # EEG
                row = (
                    [f"{ts:.6f}", 'EEG']
                    + [f"{v:.3f}" for v in payload]
                    + ['', '', '']
                )
            self._writer.writerow(row)
        self._buffer.clear()

    def get_eeg_buffer(self):
        """Return all newly-collected EEG samples for plotting."""
        with self._plot_buffer_lock:
            data = list(self._plot_buffer)
            self._plot_buffer.clear()
        return data

    def _cleanup(self):
        # Close CSV if open
        if hasattr(self, '_csvfile'):
            self._csvfile.close()
            print(f"\nDone — data saved to '{self.output_csv}'")
