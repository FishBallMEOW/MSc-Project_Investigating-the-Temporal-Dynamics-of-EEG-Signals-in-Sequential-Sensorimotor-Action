#!/usr/bin/env python3
"""
EEG CSV Replayer with Marker Indicators (PyQt5 + pyqtgraph)

- Plots all channels (auto-detected from CSV header; expects 16 EEG channel columns)
- Replays time using the recorded timestamps (relative to the first EEG sample)
- Draws vertical color-coded lines when events (markers) occur, with labels
- Smooth scrolling "real-time" window
- Command-line options for speed, window length, and file path

Usage:
    python eeg_replay_plotter.py --file "/mnt/data/20250816_172416_eeg_with_markers_renamed.csv" --speed 1.0 --win 15 --interval 30

Keyboard:
    Space: Pause/Resume
    Left/Right: Step backwards/forwards by 1 second (while paused)
    M: Toggle markers on/off
    Q or Esc: Quit
"""
import argparse
import sys
import json
import math
from collections import deque, defaultdict

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


# -----------------------------
# Marker color & name mapping
# -----------------------------
TRIGGER_MAP = {
    1:  "baseline",
    21: "left",
    22: "right",
    23: "up",
    24: "down",
    3:  "imagery",
    4:  "inter_trial",
    5:  "end"
}
TRIGGER_COLORS = {
    "baseline":    (150,150,150),  # grey
    "left":        (0, 114, 189),
    "right":       (217, 83, 25),
    "up":          (0, 158, 115),
    "down":        (237, 177, 32),
    "imagery":     (204, 121, 167),
    "inter_trial": (86, 180, 233),
    "end":         (255, 255, 255),
}


def load_events_only(path):
    """Return a list of (timestamp, code, label) for rows where type=='Marker'."""
    events = []
    # Only read necessary columns to keep it fast
    use = ["timestamp", "type", "marker"]
    for chunk in pd.read_csv(path, usecols=use, chunksize=200_000):
        m = chunk[chunk["type"] == "Marker"]
        if not len(m):
            continue
        ts = m["timestamp"].to_numpy(dtype=float)
        codes = m["marker"].to_numpy()
        # codes may be strings; try to coerce to int when possible
        for t, c in zip(ts, codes):
            try:
                code = int(c)
            except Exception:
                # If non-numeric, keep original string
                code = c
            label = TRIGGER_MAP.get(code, str(code))
            events.append((t, code, label))
    # Keep sorted by time
    events.sort(key=lambda x: x[0])
    return events


def load_eeg(path, channel_cols=None):
    """Load EEG rows into memory as (timestamps, data_matrix, ch_labels)."""
    # Detect channel columns if not given
    if channel_cols is None:
        # Read only header and one non-Setup EEG row to detect channels
        head = pd.read_csv(path, nrows=5)
        # Exclude non-channel columns
        non_ch = {"timestamp","type","marker","user_settings","user_info"}
        channel_cols = [c for c in head.columns if c not in non_ch]
    # We'll stream and keep only EEG rows
    ts_list = []
    data_list = []
    use = ["timestamp", "type"] + channel_cols
    for chunk in pd.read_csv(path, usecols=use, chunksize=100_000):
        eeg = chunk[chunk["type"] == "EEG"]
        if not len(eeg):
            continue
        ts_list.append(eeg["timestamp"].to_numpy(dtype=float))
        data_list.append(eeg[channel_cols].to_numpy(dtype=float))
    if not ts_list:
        raise RuntimeError("No EEG rows found in CSV.")
    timestamps = np.concatenate(ts_list, axis=0)
    data = np.vstack(data_list)  # shape: (n_samples, n_channels)
    return timestamps, data, channel_cols


class MarkerOverlay:
    """Manages marker lines & labels drawn onto plots."""
    def __init__(self, plots):
        self.plots = plots  # dict idx->PlotItem
        self.lines_by_plot = defaultdict(list)
        self.labels = []  # labels added to the first plot only
        self.enabled = True

    def clear(self):
        for idx, plot in self.plots.items():
            for item in self.lines_by_plot[idx]:
                try:
                    plot.removeItem(item)
                except Exception:
                    pass
            self.lines_by_plot[idx].clear()
        for lab in self.labels:
            try:
                self.plots[0].removeItem(lab)
            except Exception:
                pass
        self.labels.clear()

    def setEnabled(self, enabled):
        self.enabled = enabled
        for idx, plot in self.plots.items():
            for item in self.lines_by_plot[idx]:
                item.setVisible(enabled)
        for lab in self.labels:
            lab.setVisible(enabled)

    def add_event(self, t_rel, label, color=(255,255,255)):
        """Draw a vertical line at t_rel on all plots; add a small text label on the first plot."""
        if not self.enabled:
            return
        pen = pg.mkPen(color=color, width=1.5, style=QtCore.Qt.SolidLine)
        for idx, plot in self.plots.items():
            line = pg.InfiniteLine(pos=t_rel, angle=90, pen=pen, movable=False)
            plot.addItem(line)
            self.lines_by_plot[idx].append(line)
        # Label only on the first plot to avoid clutter
        txt = pg.TextItem(text=str(label), color=color, anchor=(0,1))
        txt.setPos(t_rel, self.plots[0].viewRange()[1][1])  # top of y-range
        self.plots[0].addItem(txt)
        self.labels.append(txt)

    def trim_to_window(self, t_min):
        """Hide/remove marker lines older than t_min to keep UI snappy."""
        # We simply hide them; full removal happens on clear() or periodically.
        for idx in list(self.lines_by_plot.keys()):
            new_items = []
            for item in self.lines_by_plot[idx]:
                try:
                    x = item.value()
                except Exception:
                    x = None
                if x is not None and x < t_min:
                    item.setVisible(False)
                else:
                    new_items.append(item)
            self.lines_by_plot[idx] = new_items
        kept = []
        for lab in self.labels:
            try:
                x, _ = lab.pos()
            except Exception:
                x = None
            if x is not None and x < t_min:
                lab.setVisible(False)
            else:
                kept.append(lab)
        self.labels = kept


class EEGReplayer(QtWidgets.QMainWindow):
    def __init__(self, timestamps, data, ch_labels, events, speed=1.0, win_sec=15.0, interval_ms=30):
        super().__init__()
        self.setWindowTitle("EEG CSV Replayer (16 ch)")
        self.timestamps = timestamps
        self.data = data
        self.ch_labels = ch_labels
        self.events = events
        self.speed = float(speed)
        self.win_sec = float(win_sec)
        self.interval_ms = int(interval_ms)

        # Build relative time axis starting from first EEG timestamp
        self.t0 = float(self.timestamps[0])
        self.t_rel = self.timestamps - self.t0  # seconds, relative
        self.total_duration = float(self.t_rel[-1]) if len(self.t_rel) else 0.0

        # Iterator index for how far we've revealed
        self.i_cursor = 0
        self.e_cursor = 0
        self.paused = False

        # Buffers for current window per channel
        self.buffers_t = deque(maxlen=int(2*self.win_sec*2000))  # generous; pyqtgraph handles decimation
        self.buffers_y = [deque(maxlen=int(2*self.win_sec*2000)) for _ in range(self.data.shape[1])]

        # UI
        self._init_ui()

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(self.interval_ms)

        # Replay clock
        self.wall_clock = QtCore.QElapsedTimer()
        self.wall_clock.start()

    def _init_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("▶ Playing")
        self.lbl_time = QtWidgets.QLabel("t = 0.00 s")
        self.chk_markers = QtWidgets.QCheckBox("Show markers")
        self.chk_markers.setChecked(True)
        self.chk_markers.stateChanged.connect(self._toggle_markers)
        ctrl.addWidget(self.lbl_status)
        ctrl.addStretch(1)
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch(1)
        ctrl.addWidget(self.chk_markers)
        layout.addLayout(ctrl)

        # Grid of 4x4 plots
        self.grid = pg.GraphicsLayoutWidget()
        layout.addWidget(self.grid, stretch=1)

        pg.setConfigOptions(antialias=True, useOpenGL=False)  # robust default

        self.plots = {}
        self.curves = {}

        rows, cols = 4, 4
        n = len(self.ch_labels)
        for k in range(n):
            r, c = divmod(k, cols)
            p = self.grid.addPlot(row=r, col=c, title=f"{self.ch_labels[k]}")
            p.setLabel('bottom', 'Time', units='s')
            p.setLabel('left', 'Amplitude', units='µV')
            p.showGrid(x=True, y=True, alpha=0.2)
            # Performance helpers
            p.setDownsampling(mode='peak')
            p.setClipToView(True)
            # curve
            curve = p.plot([], [], pen=pg.intColor(k, n))
            self.plots[k] = p
            self.curves[k] = curve

        # Marker overlay manager
        self.markers = MarkerOverlay(self.plots)

        # Legend panel (simple text with code→label)
        legend_text = "Markers: " + ", ".join([f"{code}={{label}}" for code, label in TRIGGER_MAP.items()])
        self.legend = QtWidgets.QLabel(legend_text)
        layout.addWidget(self.legend)

        # Shortcuts
        QtWidgets.QShortcut(QtCore.Qt.Key_Space, self, activated=self._toggle_pause)
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self, activated=lambda: self._nudge(+1.0))
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self, activated=lambda: self._nudge(-1.0))
        QtWidgets.QShortcut(QtCore.Qt.Key_M, self, activated=lambda: self._toggle_markers(self.chk_markers.isChecked()))  # mirror
        QtWidgets.QShortcut(QtCore.Qt.Key_Q, self, activated=self.close)
        QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self, activated=self.close)

        # Nice default size
        self.resize(1400, 900)

    # ---- Playback controls ----

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            self.lbl_status.setText("⏸ Paused")
            self.wall_clock.invalidate()
        else:
            self.lbl_status.setText("▶ Playing")
            self.wall_clock.restart()

    def _nudge(self, seconds):
        """Seek by a small amount (works while paused)."""
        if not self.paused:
            return
        # Move cursor to closest time +/- seconds
        if self.i_cursor >= len(self.t_rel):
            return
        current_t = self.t_rel[self.i_cursor-1] if self.i_cursor > 0 else 0.0
        target = max(0.0, min(self.total_duration, current_t + seconds))
        # Binary search to find new cursor
        idx = int(np.searchsorted(self.t_rel, target, side='left'))
        self._seek(idx)

    def _seek(self, new_index):
        # Reset buffers and redraw
        self.i_cursor = max(0, min(int(new_index), len(self.t_rel)))
        self.buffers_t.clear()
        for dq in self.buffers_y:
            dq.clear()
        # Fill buffers from some margin before window start (optional)
        t_now = self.t_rel[self.i_cursor-1] if self.i_cursor>0 else 0.0
        t_min = max(0.0, t_now - self.win_sec)
        i_start = int(np.searchsorted(self.t_rel, t_min, side='left'))
        # Append samples up to current index
        for i in range(i_start, self.i_cursor):
            self.buffers_t.append(self.t_rel[i])
            for ch in range(self.data.shape[1]):
                self.buffers_y[ch].append(self.data[i, ch])
        self._draw_curves()
        # markers
        self.markers.clear()
        self._add_markers_up_to_time(t_now, clear=False)
        self._update_x_range(t_now)

    def _toggle_markers(self, state=None):
        if isinstance(state, bool):
            enabled = state
        else:
            enabled = self.chk_markers.isChecked()
        self.markers.setEnabled(enabled)

    # ---- Timer tick ----

    def _on_tick(self):
        if self.paused:
            return
        # Compute current playback time in the recording
        elapsed_s = self.wall_clock.elapsed() / 1000.0  # ms -> s
        t_now = min(self.total_duration, elapsed_s * self.speed)
        self.lbl_time.setText(f"t = {t_now:.2f} s")

        # Reveal all samples up to t_now
        # (self.i_cursor keeps how many we've already appended)
        n = len(self.t_rel)
        if self.i_cursor < n:
            # Find the farthest index where t_rel <= t_now
            i_end = int(np.searchsorted(self.t_rel, t_now, side='right'))
            if i_end > self.i_cursor:
                # Append new samples into buffers
                for i in range(self.i_cursor, i_end):
                    self.buffers_t.append(self.t_rel[i])
                    for ch in range(self.data.shape[1]):
                        self.buffers_y[ch].append(self.data[i, ch])
                self.i_cursor = i_end
                # Draw
                self._draw_curves()

        # Add markers that occur up to t_now
        self._add_markers_up_to_time(t_now)

        # Scroll x-range
        self._update_x_range(t_now)

        # If we've reached the end, pause automatically
        if math.isclose(t_now, self.total_duration, rel_tol=0, abs_tol=1e-3):
            self._toggle_pause()

    def _draw_curves(self):
        # Optional decimation: pyqtgraph handles some via setDownsampling/ClipToView
        # But to keep it smooth, we can subsample when too many points
        max_pts = 4000
        t = np.array(self.buffers_t, dtype=float)
        for ch in range(self.data.shape[1]):
            y = np.array(self.buffers_y[ch], dtype=float)
            if len(t) > max_pts:
                step = max(1, len(t)//max_pts)
                self.curves[ch].setData(t[::step], y[::step])
            else:
                self.curves[ch].setData(t, y)

    def _update_x_range(self, t_now):
        t_min = max(0.0, t_now - self.win_sec)
        t_max = max(self.win_sec, t_now)  # ensure positive width
        for p in self.plots.values():
            p.setXRange(t_min, t_max, padding=0.0)
        # Also trim hidden markers
        self.markers.trim_to_window(t_min)

    def _add_markers_up_to_time(self, t_now, clear=False):
        # Convert absolute event times to relative
        while self.e_cursor < len(self.events) and (self.events[self.e_cursor][0] - self.t0) <= t_now:
            abs_t, code, label = self.events[self.e_cursor]
            t_rel = abs_t - self.t0
            color = TRIGGER_COLORS.get(label, (255,255,255))
            self.markers.add_event(t_rel, label, color=color)
            self.e_cursor += 1

    # ---- Qt Overrides ----
    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        event.accept()


def main():
    ap = argparse.ArgumentParser(description="Replay EEG CSV with markers in a real-time-like plotter.")
    ap.add_argument("--file", default="/mnt/data/20250816_172416_eeg_with_markers_renamed.csv", help="Path to CSV file.")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (1.0 = real-time).")
    ap.add_argument("--win", type=float, default=15.0, help="Visible time window (seconds).")
    ap.add_argument("--interval", type=int, default=30, help="GUI update interval (ms).")
    args = ap.parse_args()

    # Load channels+EEG
    # Detect EEG channel columns (anything not in the non-channel set)
    head = pd.read_csv(args.file, nrows=5)
    non_ch = {"timestamp","type","marker","user_settings","user_info"}
    ch_cols = [c for c in head.columns if c not in non_ch]
    if len(ch_cols) != 16:
        print(f"[Info] Detected {len(ch_cols)} channels: {ch_cols}")

    # Full load
    print("Loading EEG…")
    ts, data, ch_labels = load_eeg(args.file, channel_cols=ch_cols)
    print(f"Loaded EEG: {data.shape[0]} samples × {data.shape[1]} channels. Duration ~= {ts[-1]-ts[0]:.1f} s")

    print("Loading markers…")
    events = load_events_only(args.file)
    print(f"Loaded markers: {len(events)} events")

    app = QtWidgets.QApplication(sys.argv)
    w = EEGReplayer(ts, data, ch_labels, events, speed=args.speed, win_sec=args.win, interval_ms=args.interval)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
