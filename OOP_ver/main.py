import threading
from experiment import MotorImageryExperiment
from recorder import LSLDataRecorder

if __name__ == '__main__':
    # Instantiate recorder
    recorder = LSLDataRecorder(
        eeg_name='EEG',
        marker_name='GameMarkers',
        eeg_chunk_size=32
    )
    # Start recording in a background thread
    rec_thread = threading.Thread(target=recorder.start, daemon=True)
    rec_thread.start()

    # Run the motor imagery experiment
    exp = MotorImageryExperiment()
    exp.run()

    # After experiment ends, stop recorder and wait for thread
    recorder.stop()
    rec_thread.join()