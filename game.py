from psychopy import visual, core, event, sound, gui
import random
from pylsl import StreamInfo, StreamOutlet, local_clock

# === LSL MARKER STREAM ===
marker_info   = StreamInfo(name='GameMarkers',
                           type='Markers',
                           channel_count=1,
                           nominal_srate=0,
                           channel_format='int32',
                           source_id='psychopy_win_001')
marker_outlet = StreamOutlet(marker_info)

triggers = {
    'baseline':     1,
    'cue':          2,
    'imagery':      3,
    'inter_trial':  4,
    'end':          5
}

# Helper to wait while allowing an Escape abort
def wait_with_escape(duration):
    timer = core.Clock()
    while timer.getTime() < duration:
        if event.getKeys(keyList=['escape']):
            win.close()
            core.quit()
        core.wait(0.01)

# 1. Setup variables for timing and trial/block structure
baseline_time     = 2.0    # seconds
cue_time          = 1.0    # seconds
imagery_time      = 4.0    # seconds
iti_time          = 2.0    # seconds
trials_per_block  = 48
num_blocks        = 6

# Define stimulus parameters
frequencies = {
    'left_hand':  400,  # Hz
    'right_hand': 600,
    'feet':  800,
    'tongue': 1000
}
angles = {
    'left_hand':  90,  # Degrees
    'right_hand': 270,
    'feet':  180,
    'tongue': 360
}

# 2. Instruction page with sensory-mode selection
params = {'Sensory Mode': ['visual', 'auditory', 'multisensory']}
dlg = gui.DlgFromDict(dictionary=params, title='Motor Imagery Task Setup')
if not dlg.OK:
    core.quit()
mode = params['Sensory Mode']

win = visual.Window(fullscr=True, color='black', units='norm')
instr = visual.TextStim(win,
    text=(
        "Welcome to the Motor Imagery Experiment.\n\n"
        "Press SPACE to begin, or ESC at any time to abort.\n\n"
        f"Mode selected: {mode}"
    ),
    color='white',
    height=0.07
)
instr.draw()
win.flip()
keys = event.waitKeys(keyList=['space','escape'])
if 'escape' in keys:
    win.close()
    core.quit()

# --- EEG recording would start here ---
# Pre-create stimuli objects
fixation = visual.TextStim(win, text='+', height=0.1, color='white')
arrow    = visual.ShapeStim(
    win,
    vertices=[(-0.25,0), (0,0.5), (0.25,0), (0,-0.03)],
    fillColor='white',
    lineColor='white'
)
beeps = {cls: sound.Sound(value=freq, secs=0.2) for cls, freq in frequencies.items()}

# 3 & 4. Run blocks and trials
classes = list(angles.keys())
for block in range(num_blocks):
    # Create and shuffle balanced trial list for this block
    block_trials = classes * (trials_per_block // len(classes))
    random.shuffle(block_trials)
    
    for cls in block_trials:
        # Baseline
        fixation.draw()
        win.callOnFlip(marker_outlet.push_sample,[triggers['baseline']], local_clock())
        win.flip()
        wait_with_escape(baseline_time)

        # Cue
        # recording.send_trigger('cue')
        win.flip()  # Clear previous stimuli
        if mode in ('visual', 'multisensory'):
            arrow.ori = angles[cls]
            arrow.draw()
        if mode in ('auditory', 'multisensory'):
            beeps[cls].play()
        win.callOnFlip(marker_outlet.push_sample,[triggers['cue']], local_clock())
        win.flip()
        wait_with_escape(cue_time)

        # Motor imagery
        # recording.send_trigger('imagery')
        win.callOnFlip(marker_outlet.push_sample,[triggers['imagery']], local_clock())
        win.flip()  # Clear cue
        wait_with_escape(imagery_time)

        # Inter-trial interval
        win.callOnFlip(marker_outlet.push_sample,[triggers['inter_trial']], local_clock())
        win.flip()
        wait_with_escape(iti_time)

    # Short break after each block (except after the last)
    if block < num_blocks - 1:
        rest = visual.TextStim(win,
            text="1-minute break...\nPress ESC to abort.",
            color='white',
            height=0.08
        )
        rest.draw()
        win.flip()
        wait_with_escape(60)

# End of experiment
# recording.send_trigger('end')
thanks = visual.TextStim(win,
    text="Experiment complete. Thank you!\n(ESC will also close)",
    color='white',
    height=0.08
)
thanks.draw()
win.callOnFlip(marker_outlet.push_sample,[triggers['end']], local_clock())
win.flip()
wait_with_escape(5)

win.close()
core.quit()
