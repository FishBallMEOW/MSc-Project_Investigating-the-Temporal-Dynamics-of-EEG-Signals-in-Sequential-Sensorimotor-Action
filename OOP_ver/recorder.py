"""
Recorder module: handles LSL EEG streaming and saving to CSV/EDF
"""
import csv
import time
import queue
from datetime import datetime
import numpy as np
import pyedflib
from pylsl import StreamInlet, resolve_byprop, local_clock

class Recorder:
    def __init__(self, stream_name='EEG', chunk_size=32, output_prefix=None, event_queue=None, start_event=None):
        self.stream_name = stream_name
        self.chunk_size = chunk_size
        self.event_queue = event_queue
        self.start_event = start_event
        self.running = True
        prefix = output_prefix or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = f"{prefix}_eeg.csv"
        self.edf_path = f"{prefix}.edf"

    def stop(self):
        self.running = False

    def run(self):
        # wait for start
        print('Recorder waiting for start...')
        self.start_event.wait()
        print('Recorder started.')
        streams = resolve_byprop('name', self.stream_name, timeout=5)
        if not streams:
            raise RuntimeError(f"No LSL stream '{self.stream_name}'")
        inlet = StreamInlet(streams[0], max_buflen=10)
        time.sleep(1)
        offset = inlet.time_correction()

        # prepare CSV
        header = ['timestamp','type'] + [f'ch{i+1}' for i in range(inlet.channel_count)] + ['marker']
        samples = []
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            while self.running or not self.event_queue.empty():
                data, ts = inlet.pull_chunk(max_samples=self.chunk_size, timeout=0.1)
                ts = [t+offset for t in ts]
                for sample, t in zip(data, ts):
                    writer.writerow([f'{t:.6f}','EEG'] + [f'{v:.3f}' for v in sample] + [''])
                    samples.append(sample)

                while not self.event_queue.empty():
                    trig, t = self.event_queue.get()
                    writer.writerow([f'{t:.6f}','Marker'] + ['']*inlet.channel_count + [str(trig)])

        # write EDF
        nchan, sfreq = inlet.channel_count, int(inlet.nominal_srate)
        hdrs=[]
        for i in range(nchan): hdrs.append({
            'label':f'ch{i+1}','dimension':'uV','sample_rate':sfreq,
            'physical_min':-100,'physical_max':100,'digital_min':-32768,'digital_max':32767,
            'transducer':'EEG','prefilter':''
        })
        with pyedflib.EdfWriter(self.edf_path,nchan) as edf:
            edf.setSignalHeaders(hdrs)
            edf.writeSamples(np.array(samples).T)
        print(f"Saved CSV: {self.csv_path}, EDF: {self.edf_path}")