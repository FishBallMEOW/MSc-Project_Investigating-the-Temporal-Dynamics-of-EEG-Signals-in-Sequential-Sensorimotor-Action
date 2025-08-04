#!/usr/bin/env python3
"""
Experiment module: motor imagery trial flow
"""
import numpy as np
# Back-fill the deprecated name for PsychoPy
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all
    
import random
from pylsl import local_clock
from psychopy import visual, core, event, sound
from gui import GUI

class Experiment:
    def __init__(self, settings, info, event_queue, start_event):
        self.mode, self.class_mode, self.feedback = settings
        self.initials, self.age, self.vision = info
        self.event_queue = event_queue
        self.start_event = start_event
        self.win = visual.Window(fullscr=True, color='black', units='norm')
        self._init_params()

    def _init_params(self):
        from datetime import datetime
        self.TRIGGERS = dict(baseline=1,cue=2,imagery=3,inter_trial=4,end=5)
        self.times = dict(baseline=2,cue=1,imagery=4,iti=2)
        freqs = dict(left_hand=1000,right_hand=600,feet=800,tongue=400)
        angs = dict(left_hand=270,right_hand=90,feet=180,tongue=360)
        self.classes = ['left_hand','right_hand'] if self.class_mode=='2-class' else list(freqs)
        self.beeps = {c:sound.Sound(freqs[c],secs=0.2) for c in self.classes}
        self.fix = visual.TextStim(self.win,'+',color='white',height=0.1)
        self.arrow = visual.ShapeStim(self.win,vertices=[(-0.25,0.5),(0,1),(0.25,0.5),(0,0.5)],fillColor='white',lineColor='white')
        self.angles = angs
        self.trials_per_block = 48; self.num_blocks = 6

    def _log(self, trig):
        ts = local_clock()
        self.event_queue.put((self.TRIGGERS[trig], ts))

    def _wait(self,dur):
        clock=core.Clock()
        while clock.getTime()<dur:
            if event.getKeys(['escape']): core.quit()
            core.wait(0.01)

    def run(self):
        # show instructions
        instr_text=(f"Participant: {self.initials} Age:{self.age} Vision:{self.vision}\n"
                    "Press SPACE to begin or ESC to abort.")
        if not GUI().show_instructions(self.win,instr_text): return
        self.start_event.set()

        for b in range(self.num_blocks):
            trials=self.classes*(self.trials_per_block//len(self.classes))
            random.shuffle(trials)
            for c in trials:
                # baseline
                self.fix.draw(); self.win.flip(); self._log('baseline'); self._wait(self.times['baseline'])
                # cue
                if self.mode in ('visual','multisensory'):
                    self.arrow.ori=self.angles[c]; self.arrow.draw()
                if self.mode in ('auditory','multisensory'):
                    self.beeps[c].play()
                self.win.flip(); self._log('cue'); self._wait(self.times['cue']); self.beeps[c].stop()
                # imagery
                self.fix.draw(); self.win.flip(); self._log('imagery'); self._wait(self.times['imagery'])
                # iti
                self.win.flip(); self._log('inter_trial'); self._wait(self.times['iti'])
            # break
            if b<self.num_blocks-1:
                rest=visual.TextStim(self.win,'1-minute break...',color='white',height=0.08)
                rest.draw(); self.win.flip(); self._wait(60)
        # end
        thanks=visual.TextStim(self.win,'Experiment complete. ESC to close.',color='white',height=0.08)
        thanks.draw(); self.win.flip(); self._log('end'); self._wait(5)
        self.win.close()
