#!/usr/bin/env python3
"""
GUI module: participant info and settings dialogs
"""
from psychopy import gui

class GUI:
    def __init__(self):
        pass

    def get_setup(self):
        settings = {'Sensory Mode':['visual','auditory','multisensory'],
                    'Class Mode':['2-class','4-class'],
                    'Feedback':['with feedback','without feedback']}
        dlg = gui.DlgFromDict(settings, title='Experiment Setup')
        if not dlg.OK: return None
        return dlg.data

    def get_participant_info(self):
        info = {'Initials':'','Age':'','Vision':['normal','corrected']}
        dlg = gui.DlgFromDict(info, title='Participant Info')
        if not dlg.OK: return None
        return dlg.data

    def show_instructions(self, win, text):
        from psychopy import visual, event
        instr = visual.TextStim(win, text=text, color='white', height=0.06)
        instr.draw(); win.flip()
        keys = event.waitKeys(keyList=['space','escape'])
        return 'space' in keys