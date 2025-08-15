import numpy as np
from psychopy import visual, core, event, sound, gui
import random
from pylsl import StreamInfo, StreamOutlet, local_clock
import json
import os
from datetime import datetime

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
            'iti':      4.0,
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
            'Vision': ['Normal', 'Corrected (glasses/contact lenses)', 'Impaired'],
            'Handedness': ['Left', 'Right', 'Ambidextrous'],
            'Hearing': ['Normal', 'Corrected (hearing aids or cochlear implant)', 'Impaired']
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
        from textwrap import dedent

        def _fmt_secs(secs):
            secs = int(round(float(secs)))
            if secs < 60:
                return f"{secs} s"
            m, s = divmod(secs, 60)
            return f"{m} min" if s == 0 else f"{m} min {s} s"

        self.win = visual.Window(fullscr=True, color='black', units='norm')

        # ---- Build instruction pages based on current settings ----
        header = f"Motor Imagery Task\nMode: {self.mode}    Classes: {self.class_mode}\n\nPlease read all instruction pages carefully before you begin.\n\n"
        flow = (
            "Trial flow: Fixate '+' (4 s) → cue (1 s) → IMAGINE (10 s) → rest (6 s).\n"
            "Keep your eyes on '+'.\n\n"
        )

        # Cues text (mode + class specific)
        cues = ""
        if self.class_mode.lower() == "2-class":
            if self.mode.lower() == "visual":
                cues = ("Visual cues:\n"
                        "→ : imagine squeezing & releasing your RIGHT fist (no movement)\n"
                        "← : imagine squeezing & releasing your LEFT fist\n")
            elif self.mode.lower() == "auditory":
                cues = ("Auditory cues:\n"
                        "Low-pitch beep : RIGHT fist imagery\n"
                        "High-pitch beep: LEFT fist imagery\n")
            else:  # multisensory
                cues = ("Multisensory (arrow + beep):\n"
                        "→ / low-pitch = RIGHT   |   ← / high-pitch = LEFT\n")
        else:  # 4-class
            if self.mode.lower() == "visual":
                cues = ("Visual cues:\n"
                        "↑ : imagine pressing your TONGUE to the roof of your mouth, then relax\n"
                        "↓ : imagine pressing BOTH FEET down, then relax\n"
                        "→ : imagine squeezing & releasing your RIGHT fist\n"
                        "← : imagine squeezing & releasing your LEFT fist\n")
            elif self.mode.lower() == "auditory":
                cues = ("Auditory cues (lowest → highest pitch):\n"
                        "Lowest: TONGUE   |   Low: RIGHT hand   |   Medium: FEET   |   High: LEFT hand\n")
            else:  # multisensory
                cues = ("Multisensory (arrow + beep): they match.\n"
                        "↑/lowest = TONGUE   ↓/medium = FEET   →/low = RIGHT   ←/high = LEFT\n")

        # Blocks & breaks
        trials_per_block = getattr(self, "trials_per_block", None)
        num_blocks = getattr(self, "num_blocks", None)
        break_secs = self.timings.get('break', 60)
        block_info = (
            f"Blocks & breaks:\n"
            f"{num_blocks} blocks × {trials_per_block} trials.\n"
            f"After each block: short break (~{_fmt_secs(break_secs)}).\n\n"
        )

        page1 = header + flow + cues + "\n" + block_info

        reminders = dedent("""\
            Reminders:
            • Keep completely still — no actual muscle contractions.
            • Minimise blinks/swallowing; if needed, do it during the REST period.
            • Sit comfortably, relax jaw, breathe naturally.
            • Focus on the FEELING of doing the action (kinesthetic imagery), not on seeing it.
        """)
        page2 = reminders

        pages = [page1, page2]

        # ---- Page display loop ----
        page_idx = 0
        footer_tmpl = "Page {}/{}    SPACE: begin    ←/→: navigate    ESC: abort"

        while True:
            text = pages[page_idx] + "\n\n" + footer_tmpl.format(page_idx + 1, len(pages))
            instr = visual.TextStim(
                self.win,
                text=text,
                color='white',
                height=0.06,
                wrapWidth=1.8,
            )
            instr.draw()
            self.win.flip()

            keys = event.waitKeys(keyList=['left', 'right', 'space', 'escape'])
            if 'escape' in keys:
                self._cleanup()
                return
            if 'left' in keys:
                page_idx = (page_idx - 1) % len(pages)
            elif 'right' in keys:
                page_idx = (page_idx + 1) % len(pages)
            elif 'space' in keys:
                break

    def _post_experiment_questionnaire(self):
        """
        Shows a 4-item Likert (1–4) questionnaire:
        Concentration, Sleepiness, Fatigue, Difficulty.
        Saves responses + basic metadata to a JSON file.
        Returns the response dict (or None if aborted).
        """
        def ask_item(name, scale_text):
            while True:
                msg = (f"Post-task questionnaire\n\n{name}\n\n"
                    f"Please rate using number keys 1–4:\n{scale_text}\n\n"
                    "Press 1 / 2 / 3 / 4   |   ESC: abort")
                visual.TextStim(self.win, text=msg, color='white',
                                height=0.08, wrapWidth=1.8).draw()
                self.win.flip()
                keys = event.waitKeys(keyList=['1', '2', '3', '4', 'escape'])
                if 'escape' in keys:
                    self._cleanup()
                    return None
                choice = next(k for k in keys if k in ['1','2','3','4'])
                return int(choice)

        items = [
            ("Concentration", "1 = very low      4 = very high"),
            ("Sleepiness",    "1 = not sleepy     4 = very sleepy"),
            ("Fatigue",       "1 = not tired      4 = very tired"),
            ("Difficulty",    "1 = very easy      4 = very difficult"),
        ]

        responses = {}
        for name, scale in items:
            val = ask_item(name, scale)
            if val is None:  # aborted
                return None
            responses[name.lower()] = val

        # Summary/confirm page
        summary_lines = [f"{k.capitalize()}: {v}" for k, v in responses.items()]
        summary_txt = "Review your answers:\n\n" + "\n".join(summary_lines) + \
                    "\n\nSPACE: save   BACKSPACE: redo   ESC: abort"
        while True:
            visual.TextStim(self.win, text=summary_txt, color='white',
                            height=0.07, wrapWidth=1.8).draw()
            self.win.flip()
            keys = event.waitKeys(keyList=['space', 'backspace', 'escape'])
            if 'escape' in keys:
                self._cleanup()
                return None
            if 'backspace' in keys:
                # redo all questions
                return self._post_experiment_questionnaire()
            if 'space' in keys:
                break

        # Build metadata
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": getattr(self, "mode", None),
            "class_mode": getattr(self, "class_mode", None),
            "num_blocks": getattr(self, "num_blocks", None),
            "trials_per_block": getattr(self, "trials_per_block", None),
            "break_seconds": (self.timings.get('break') if hasattr(self, "timings") else None),
            "participant": getattr(self, "participant_info", None),  # if you stored it earlier
        }

        out = {"questionnaire": responses, "meta": meta}

        # Decide filename/location
        fname = datetime.now().strftime("%Y%m%d_%H%M%S") + "_post_questionnaire.json"
        out_dir = getattr(self, "output_dir", None)
        path = os.path.join(out_dir, fname) if (out_dir and os.path.isdir(out_dir)) else fname

        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        # Final confirmation
        visual.TextStim(self.win,
                        text=f"Thank you!\n\nYour responses were saved to:\n{path}\n\nPress SPACE to finish.",
                        color='white', height=0.07, wrapWidth=1.8).draw()
        self.win.flip()
        event.waitKeys(keyList=['space'])
        return responses

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
            random.shuffle(trials)
            self._run_block(trials)
            if block < self.num_blocks - 1:
                self._take_break()
        self._post_experiment_questionnaire()
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



