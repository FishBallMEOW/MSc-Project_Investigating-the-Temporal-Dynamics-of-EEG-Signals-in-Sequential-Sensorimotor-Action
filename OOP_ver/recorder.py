"""
Module defining a class to record EEG and marker streams to CSV via LSL.
"""
import csv
import signal
import sys
from datetime import datetime
import time
from pylsl import StreamInlet, resolve_byprop, local_clock

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
    def __init__(self,eeg_name='EEG',marker_name='GameMarkers',userinfo_name='UserInfo',eeg_chunk_size=32):
        self.eeg_name = eeg_name
        self.marker_name = marker_name
        self.userinfo_name = userinfo_name
        self.eeg_chunk_size = eeg_chunk_size
        self.running = False
        self._setup_signal_handlers()
        self._buffer = []            # small in-memory event buffer
        self._buffer_size = 20       # flush every 20 events

    def _setup_signal_handlers(self):
        # allow graceful shutdown on SIGINT/SIGTERM
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, sig, frame):
        self.running = False

    def _resolve_streams(self):
        print(f"Looking for EEG stream '{self.eeg_name}'...")
        eegs = resolve_byprop('name', self.eeg_name, timeout=5)
        if not eegs:
            print(f"ERROR: Couldn’t find EEG stream '{self.eeg_name}'")
            sys.exit(1)
        self.eeg_inlet = StreamInlet(eegs[0], max_buflen=10)
        time.sleep(1.0)
        self.eeg_offset = self.eeg_inlet.time_correction()
        print(f"EEG offset: {self.eeg_offset}")

        print(f"Looking for userinfo stream '{self.userinfo_name}'...")
        ui_streams = resolve_byprop('name', self.userinfo_name, timeout=5)
        if not ui_streams:
            print(f"ERROR: Couldn’t find userinfo stream '{self.userinfo_name}'")
            sys.exit(1)
        self.userinfo_inlet = StreamInlet(ui_streams[0], max_buflen=360)
        time.sleep(1.0)
        self.userinfo_offset = self.userinfo_inlet.time_correction()
        print(f"UserInfo offset: {self.userinfo_offset}")

        print(f"Looking for marker stream '{self.marker_name}'...")
        marks = resolve_byprop('name', self.marker_name, timeout=5)
        if not marks:
            print(f"ERROR: Couldn’t find marker stream '{self.marker_name}'")
            sys.exit(1)
        self.marker_inlet = StreamInlet(marks[0], max_buflen=10)
        time.sleep(1.0)
        self.marker_offset = self.marker_inlet.time_correction()
        print(f"Marker offset: {self.marker_offset}")

    def _prepare_output(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = f"{now}_eeg_with_markers.csv"
        header = ['timestamp', 'type'] + \
            [f'ch{i}' for i in range(1, self.eeg_inlet.channel_count + 1)] + ['marker']
        self._csvfile = open(self.output_csv, 'w', newline='')
        self._writer = csv.writer(self._csvfile)
        self._writer.writerow(header)
        print(f"Recording… output will be saved to '{self.output_csv}'. Press Ctrl+C to stop.")

    def _flush_buffer(self):
        # sort by timestamp
        self._buffer.sort(key=lambda e: e[0])
        # write them all out
        for ts, kind, payload in self._buffer:
            if kind == 'Marker':
                row = [f"{ts:.6f}", 'Marker'] + [''] * self.eeg_inlet.channel_count + [str(payload)]
            else:  # EEG
                row = [f"{ts:.6f}", 'EEG'] + [f"{v:.3f}" for v in payload] + ['']
            self._writer.writerow(row)
        # reset buffer
        self._buffer.clear()

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
                
                # 3) Append into buffer
                for (sample,), ts in zip(m_samples, m_ts):
                    self._buffer.append((ts, 'Marker', sample))
                for sample, ts in zip(eeg_samples, eeg_ts):
                    self._buffer.append((ts, 'EEG', sample))

                # 4) When buffer big enough, sort & flush
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

    def _cleanup(self):
        self._csvfile.close()
        print(f"\nDone — data saved to '{self.output_csv}'")