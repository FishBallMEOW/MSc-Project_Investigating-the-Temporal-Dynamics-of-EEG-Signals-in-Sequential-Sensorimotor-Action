import threading
import json
from pylsl import resolve_stream, StreamInlet
from experiment import MotorImageryExperiment
from recorder import LSLDataRecorder

if __name__ == '__main__':
    # Resolve the participant info LSL stream
    print("Resolving UserInfo LSL stream...")
    userinfo_streams = resolve_stream('type', 'UserInfo', timeout=5.0)
    if not userinfo_streams:
        raise RuntimeError("No UserInfo LSL stream found")
    userinfo_inlet = StreamInlet(userinfo_streams[0])
    print("Waiting for participant info sample...")

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