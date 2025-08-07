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
    def __init__(self,eeg_name='EEG',marker_name='GameMarkers',eeg_chunk_size=32):
        self.eeg_name = eeg_name
        self.marker_name = marker_name
        self.eeg_chunk_size = eeg_chunk_size
        self.running = False
        self._setup_signal_handlers()

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

    def start(self):
        """Begin recording loop until stopped."""
        self.running = True
        self._resolve_streams()
        self._prepare_output()
        while self.running:
            # Pull markers
            m_samples, m_ts = self.marker_inlet.pull_chunk(timeout=0.0, max_samples=10)
            m_ts = [ts + self.marker_offset for ts in m_ts]
            for sample, ts in zip(m_samples, m_ts):
                val = sample[0]
                print(f"╞═ Received marker {val} @ {ts:.6f}")
                row = [f"{ts:.6f}", 'Marker'] + [''] * self.eeg_inlet.channel_count + [str(val)]
                self._writer.writerow(row)

            # Pull EEG
            eeg_samples, eeg_ts = self.eeg_inlet.pull_chunk(
                timeout=0.0,
                max_samples=self.eeg_chunk_size
            )
            eeg_ts = [ts + self.eeg_offset for ts in eeg_ts]
            for sample, ts in zip(eeg_samples, eeg_ts):
                row = [f"{ts:.6f}", 'EEG'] + [f"{v:.3f}" for v in sample] + ['']
                self._writer.writerow(row)
        self._cleanup()

    def stop(self):
        """Stop the recording loop gracefully."""
        self.running = False

    def _cleanup(self):
        self._csvfile.close()
        print(f"\nDone — data saved to '{self.output_csv}'")