#!/usr/bin/env python3
"""
Main entry: launches Recorder and Experiment
"""
import threading, queue
from recorder import Recorder
from gui import GUI
from experiment import Experiment

def main():
    event_queue = queue.Queue()
    start_event = threading.Event()

    gui = GUI()
    settings = gui.get_setup()
    if settings is None: return
    info = gui.get_participant_info()
    if info is None: return

    recorder = Recorder(event_queue=event_queue, start_event=start_event)
    exp = Experiment(settings, info, event_queue, start_event)

    rec_thread = threading.Thread(target=recorder.run,daemon=True)
    rec_thread.start()

    exp.run()
    recorder.stop()
    rec_thread.join()

if __name__=='__main__':
    main()
