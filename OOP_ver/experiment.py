import numpy as np
from psychopy import visual, core, event, sound, gui
import random
from pylsl import StreamInfo, StreamOutlet, local_clock
import json

# Back-fill the deprecated name for PsychoPy
if not hasattr(np, 'alltrue'):
    np.alltrue = np.all


class MotorImageryExperiment:
    """
    Class-based implementation of a Motor Imagery experiment with LSL marker streaming.

    Attributes:
        mode (str): 'visual', 'auditory', or 'multisensory'.
        class_mode (str): '2-class' or '4-class'.
        classes (list): List of motor imagery classes.
        timings (dict): Timing parameters for baseline, cue, imagery, ITI, and breaks.
        triggers (dict): Integer codes for each event marker.
        frequencies (dict): Frequencies for auditory cues.
        angles (dict): Angles for visual arrow cues.
        win (psychopy.visual.Window): PsychoPy window object.
        marker_outlet (StreamOutlet): LSL outlet for sending markers.
        fixation (psychopy.visual.TextStim): Fixation cross stimulus.
        arrow (psychopy.visual.ShapeStim): Arrow cue stimulus.
        beeps (dict): Sound stimuli for each class.
    """

    def __init__(self):
        self._init_trigger_stream()
        self._init_user_setting_stream()
        self._init_user_info_stream()
        self._init_parameters()
        self._get_user_settings()
        self._get_user_info()
        self._create_window()
        self._create_stimuli()

    def _init_trigger_stream(self):
        info = StreamInfo(
            name='GameMarkers',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='int32',
            source_id='psychopy_win_001'
        )
        self.marker_outlet = StreamOutlet(info)
        self.triggers = {
            'baseline':     1,
            'left':         21,
            'right':        22,
            'up':           23,
            'down':         24,
            'imagery':      3,
            'inter_trial':  4,
            'end':          5
        }

    def _init_parameters(self):
        # Timing parameters
        self.timings = {
            'baseline': 4.0,
            'cue':      1.0,
            'imagery':  10.0,
            'iti':      6.0,
            'break':    60.0
        }
        self.num_blocks = 6
        self.trials_per_block = 20

        # Stimulus properties
        self.frequencies = {
            'left_hand': 1000,
            'right_hand': 600,
            'feet': 800,
            'tongue': 400
        }
        self.angles = {
            'left_hand': 270,
            'right_hand': 90,
            'feet': 180,
            'tongue': 360
        }

    def _init_user_setting_stream(self):
        """
        Initialize an LSL stream to send participant settings.
        """
        info = StreamInfo(
            name='UserSettings',
            type='UserSettings',
            channel_count=1,
            nominal_srate=0,
            channel_format='string',
            source_id='psychopy_usersettings_001'
        )
        self.usersettings_outlet = StreamOutlet(info)

    def _get_user_settings(self):
        dlg_config = {
            'Sensory Mode': ['visual', 'auditory', 'multisensory'],
            'Class Mode':   ['2-class', '4-class']
        }
        dlg = gui.DlgFromDict(dictionary=dlg_config, title='MI Task Setup')
        if not dlg.OK:
            core.quit()

        # Extract selections
        self.mode       = dlg_config['Sensory Mode']
        self.class_mode = dlg_config['Class Mode']
        if self.class_mode == '2-class':
            self.classes = ['left_hand', 'right_hand']
        else:
            self.classes = ['left_hand', 'right_hand', 'feet', 'tongue']

        # --- NEW: send settings over LSL as a JSON string ---
        settings = {
            'mode':       self.mode,
            'class_mode': self.class_mode,
            'classes':    self.classes
        }
        try:
            settings_json = json.dumps(settings)
        except Exception:
            # fallback to str if something odd happens
            settings_json = str(settings)
        # push with timestamp
        self.usersettings_outlet.push_sample([settings_json], local_clock())
        print(f"User settings sent: {settings_json}")
    
    def _init_user_info_stream(self):
        """
        Initialize an LSL stream to send participant information.
        """
        info = StreamInfo(
            name='UserInfo',
            type='UserInfo',
            channel_count=1,
            nominal_srate=0,
            channel_format='string',
            source_id='psychopy_userinfo_001'
        )
        self.userinfo_outlet = StreamOutlet(info)

    def _get_user_info(self):
        """
        Collect user information via a dialog and send it over LSL.
        """
        user_config = {
            'Initials': '',
            'Age': '',
            'Gender': ['Male', 'Female', 'Other', 'Prefer not to say'],
            'Vision': ['Normal', 'Corrected (glasses/contact lenses)', 'Impaired']
        }
        dlg = gui.DlgFromDict(dictionary=user_config, title='Participant Info')
        if not dlg.OK:
            core.quit()
        # Prepare and send JSON-formatted user info
        try:
            user_info_json = json.dumps(user_config)
        except Exception:
            user_info_json = str(user_config)
        # Push the user info sample with a timestamp
        self.userinfo_outlet.push_sample([user_info_json], local_clock())

    def _create_window(self):
        self.win = visual.Window(fullscr=True, color='black', units='norm')

        instr_text = (
            "Welcome to the Motor Imagery Experiment.\n\n"
            f"Sensory mode: {self.mode}   Class mode: {self.class_mode}\n\n"
            "Press SPACE to begin, or ESC at any time to abort."
        )
        instr = visual.TextStim(self.win, text=instr_text, color='white', height=0.07)
        instr.draw()
        self.win.flip()
        keys = event.waitKeys(keyList=['space', 'escape'])
        if 'escape' in keys:
            self._cleanup()

    def _create_stimuli(self):
        self.fixation = visual.TextStim(self.win, text='+', height=0.5, color='white')
        self.arrow = visual.ShapeStim(
            self.win,
            vertices=[(-0.25, 0.5), (0, 1), (0.25, 0.5), (0, 0.5)],
            fillColor='white',
            lineColor='white'
        )
        self.beeps = {
            cls: sound.Sound(value=self.frequencies[cls], secs=self.timings['cue'])
            for cls in self.classes
        }

    def wait_with_escape(self, duration):
        timer = core.Clock()
        while timer.getTime() < duration:
            if event.getKeys(keyList=['escape']):
                self._cleanup()
            core.wait(0.01)

    def run(self):
        for block in range(self.num_blocks):
            trials = self.classes * (self.trials_per_block // len(self.classes))
            print(trials)
            random.shuffle(trials)
            self._run_block(trials)
            if block < self.num_blocks - 1:
                self._take_break()

        self._finish()

    def _run_block(self, trials):
        for cls in trials:
            self._baseline_phase()
            self._cue_phase(cls)
            self._imagery_phase()
            self._iti_phase()

    def _baseline_phase(self):
        self.fixation.draw()
        self.win.callOnFlip(self.marker_outlet.push_sample, [self.triggers['baseline']], local_clock())
        self.win.flip()
        self.wait_with_escape(self.timings['baseline'])

    def _cue_phase(self, cls):
        # draw fixation + cue
        self.fixation.draw()
        # set arrow orientation
        if self.mode in ('visual', 'multisensory'):
            self.arrow.ori = self.angles[cls]
            self.arrow.draw()
        # play sound
        if self.mode in ('auditory', 'multisensory'):
            self.beeps[cls].stop()
            self.win.callOnFlip(self.beeps[cls].play)

        # map class to direction and push specific trigger
        direction_map = {
            'left_hand': 'left',
            'right_hand': 'right',
            'feet': 'down',
            'tongue': 'up'
        }
        dir_key = direction_map[cls]
        self.win.callOnFlip(self.marker_outlet.push_sample, [self.triggers[dir_key]], local_clock())
        self.win.flip()
        self.wait_with_escape(self.timings['cue'])

    def _imagery_phase(self):
        self.fixation.draw()
        self.win.callOnFlip(self.marker_outlet.push_sample, [self.triggers['imagery']], local_clock())
        self.win.flip()
        self.wait_with_escape(self.timings['imagery'])

    def _iti_phase(self):
        self.win.callOnFlip(self.marker_outlet.push_sample, [self.triggers['inter_trial']], local_clock())
        self.win.flip()
        self.wait_with_escape(self.timings['iti'])

    def _take_break(self):
        rest = visual.TextStim(
            self.win,
            text="1-minute break...",
            color='white',
            height=0.08
        )
        rest.draw()
        self.win.flip()
        self.wait_with_escape(self.timings['break'])

    def _finish(self):
        thanks = visual.TextStim(
            self.win,
            text="Experiment complete. Thank you!\n(ESC will also close)",
            color='white',
            height=0.08
        )
        self.win.callOnFlip(self.marker_outlet.push_sample, [self.triggers['end']], local_clock())
        thanks.draw()
        self.win.flip()
        self.wait_with_escape(5)
        self._cleanup()

    def _cleanup(self):
        self.win.close()
        core.quit()



