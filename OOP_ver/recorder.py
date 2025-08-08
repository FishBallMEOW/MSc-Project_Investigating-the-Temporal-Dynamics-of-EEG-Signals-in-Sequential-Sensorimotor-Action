"""
Module defining a class to record EEG, marker, and user‐setup streams to CSV via LSL.
"""
import csv
import signal
import sys
import time
from datetime import datetime
from pylsl import StreamInlet, resolve_byprop
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

    def _prepare_output(self):
        # 1) create file & writer
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = f"{now}_eeg_with_markers.csv"
        self._csvfile = open(self.output_csv, 'w', newline='')
        self._writer = csv.writer(self._csvfile)

        # 2) header: add two new columns for settings & info
        header = (
            ['timestamp', 'type']
            + [f'ch{i}' for i in range(1, self.eeg_inlet.channel_count + 1)]
            + ['marker', 'user_settings', 'user_info']
        )
        self._writer.writerow(header)

        # 3) **block until we get both user‐metadata samples**
        print("Waiting for UserSettings sample…")
        us_sample, us_ts = self.usersettings_inlet.pull_sample(timeout=10.0)
        if us_sample is None:
            print("ERROR: timed out waiting for UserSettings")
            sys.exit(1)
        us_ts += self.usersettings_offset
        settings_json = us_sample[0]

        print("Waiting for UserInfo sample…")
        ui_sample, ui_ts = self.userinfo_inlet.pull_sample(timeout=10.0)
        if ui_sample is None:
            print("ERROR: timed out waiting for UserInfo")
            sys.exit(1)
        ui_ts += self.userinfo_offset
        userinfo_json = ui_sample[0]

        # 4) write them into the **first data row**
        #    here we combine both JSONs into the new columns
        first_row = (
            [f"{us_ts:.6f}", 'Setup']
            + [''] * self.eeg_inlet.channel_count
            + ['']  # marker col
            + [settings_json, userinfo_json]
        )
        self._writer.writerow(first_row)

        print(f"Recording… data will be saved to '{self.output_csv}'. Press Ctrl+C to stop.")

    def start(self):
        try:
            """Begin recording loop until stopped."""
            self.running = True
            self._resolve_streams()
            self._prepare_output()
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

                # ── also stash these for plotting ──
                with self._plot_buffer_lock:
                    for sample, ts in zip(eeg_samples, eeg_ts):
                        self._plot_buffer.append((ts, sample))

                # buffer them
                for (sample,), ts in zip(m_samples, m_ts):
                    self._buffer.append((ts, 'Marker', sample))
                for sample, ts in zip(eeg_samples, eeg_ts):
                    self._buffer.append((ts, 'EEG', sample))

                # flush when large enough
                if len(self._buffer) >= self._buffer_size:
                    self._flush_buffer()

        finally:
            # drain any leftovers, even if you Ctrl-C or an error is raised
            if self._buffer:
                self._flush_buffer()
            self._cleanup()

    def stop(self):
        """Stop the recording loop gracefully."""
        self.running = False

    def _flush_buffer(self):
        # sort by timestamp then write
        self._buffer.sort(key=lambda e: e[0])
        for ts, kind, payload in self._buffer:
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
        """Return all newly‐collected EEG samples as a list of (timestamp, sample),
           then clear them from the internal buffer."""
        with self._plot_buffer_lock:
            data = list(self._plot_buffer)
            self._plot_buffer.clear()
        return data

    def _cleanup(self):
        self._csvfile.close()
        print(f"\nDone — data saved to '{self.output_csv}'")
