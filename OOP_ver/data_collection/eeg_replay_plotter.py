#!/usr/bin/env python3
"""
EEG CSV Replayer with Marker Indicators (PyQt5 + pyqtgraph)

Now with optional filtering:
- Bandpass: 0.5–100 Hz
- Notch: 50 Hz (Q=30)

Toggle filtering:
- Checkbox in the UI
- Keyboard shortcut: F
- Command line flag: --filter

Usage:
    python eeg_replay_plotter.py --file "/path/to/your.csv" --speed 1.0 --win 15 --interval 30 --filter

Keyboard:
    Space: Pause/Resume
    Left/Right: Step -/+ 1 s (while paused)
    M: Toggle markers
    F: Toggle filter (0.5–100 Hz + 50 Hz notch)
    Q or Esc: Quit
"""
import argparse
import sys
import json
import math
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# Optional SciPy for filtering
_SCIPY_OK = True
try:
    from scipy.signal import butter, filtfilt, iirnotch
except Exception:
    _SCIPY_OK = False


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
    use = ["timestamp", "type", "marker"]
    for chunk in pd.read_csv(path, usecols=use, chunksize=200_000):
        m = chunk[chunk["type"] == "Marker"]
        if not len(m):
            continue
        ts = m["timestamp"].to_numpy(dtype=float)
        codes = m["marker"].to_numpy()
        for t, c in zip(ts, codes):
            try:
                code = int(c)
            except Exception:
                code = c
            label = TRIGGER_MAP.get(code, str(code))
            events.append((t, code, label))
    events.sort(key=lambda x: x[0])
    return events


def load_eeg(path, channel_cols=None):
    """Load EEG rows into memory as (timestamps, data_matrix, ch_labels)."""
    if channel_cols is None:
        head = pd.read_csv(path, nrows=5)
        non_ch = {"timestamp","type","marker","user_settings","user_info"}
        channel_cols = [c for c in head.columns if c not in non_ch]
    ts_list, data_list = [], []
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
    data = np.vstack(data_list)  # (n_samples, n_channels)
    return timestamps, data, channel_cols


def estimate_fs(timestamps):
    """Estimate sampling rate from timestamps (Hz)."""
    if len(timestamps) < 3:
        return 250.0
    diffs = np.diff(timestamps)
    med_dt = np.median(diffs)
    if med_dt <= 0:
        return 250.0
    fs = 1.0 / med_dt
    return float(fs)


def design_filters(fs):
    """Design bandpass (0.5–100 Hz) and 50 Hz notch filters."""
    # Protect against extreme fs values
    nyq = fs / 2.0
    lo = max(0.5 / nyq, 1e-6)
    hi = min(100.0 / nyq, 0.999999)
    if hi <= lo:
        # If bandwidth invalid (e.g., very low fs), relax hi
        hi = min(0.45, max(lo + 1e-3, 0.49))
    b_bp, a_bp = butter(4, [lo, hi], btype='bandpass')
    # Notch
    w0 = 50.0 / nyq
    Q = 30.0
    if w0 >= 1.0:
        w0 = 0.99
    b_notch, a_notch = iirnotch(w0=w0, Q=Q)
    return (b_bp, a_bp), (b_notch, a_notch)


def apply_filters(data, fs):
    """Return filtered copy of data (n_samples, n_channels)."""
    if not _SCIPY_OK:
        raise RuntimeError("SciPy is not available for filtering.")
    (b_bp, a_bp), (b_notch, a_notch) = design_filters(fs)
    filt = np.empty_like(data)
    for ch in range(data.shape[1]):
        x = data[:, ch]
        # Notch then bandpass
        try:
            y = filtfilt(b_notch, a_notch, x, axis=0, method="pad")
            y = filtfilt(b_bp, a_bp, y, axis=0, method="pad")
        except Exception:
            # Fallback: if filtfilt fails due to short length, skip filtering
            y = x.copy()
        filt[:, ch] = y
    return filt


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
        if not self.enabled:
            return
        pen = pg.mkPen(color=color, width=1.5, style=QtCore.Qt.SolidLine)
        for idx, plot in self.plots.items():
            line = pg.InfiniteLine(pos=t_rel, angle=90, pen=pen, movable=False)
            plot.addItem(line)
            self.lines_by_plot[idx].append(line)
        # Label only on the first plot to avoid clutter
        # Place near the top of current y-range
        y_top = self.plots[0].viewRange()[1][1]
        txt = pg.TextItem(text=str(label), color=color, anchor=(0,1))
        txt.setPos(t_rel, y_top)
        self.plots[0].addItem(txt)
        self.labels.append(txt)

    def trim_to_window(self, t_min):
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
    def __init__(self, timestamps, data, ch_labels, events, speed=1.0, win_sec=15.0, interval_ms=30, start_filtered=False):
        super().__init__()
        self.setWindowTitle("EEG CSV Replayer (16 ch)")
        self.timestamps = timestamps
        self.data_raw = data
        self.data_filt = None  # lazily computed
        self.ch_labels = ch_labels
        self.events = events
        self.speed = float(speed)
        self.win_sec = float(win_sec)
        self.interval_ms = int(interval_ms)

        # Sampling rate
        self.fs = estimate_fs(self.timestamps)

        # Build relative time axis
        self.t0 = float(self.timestamps[0])
        self.t_rel = self.timestamps - self.t0
        self.total_duration = float(self.t_rel[-1]) if len(self.t_rel) else 0.0

        # Iterator indices
        self.i_cursor = 0
        self.e_cursor = 0
        self.paused = False

        # Buffers: we store time and sample indices (not values) for flexibility
        cap = int(2*self.win_sec*max(50, int(self.fs)))  # cap based on fs (>= ~100 samples/sec window)
        self.buffers_t = deque(maxlen=cap)
        self.buffers_idx = deque(maxlen=cap)

        # UI
        self._init_ui(start_filtered)

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(self.interval_ms)

        # Replay clock
        self.wall_clock = QtCore.QElapsedTimer()
        self.wall_clock.start()

    # ------------- UI -------------
    def _init_ui(self, start_filtered):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Controls row
        ctrl = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("▶ Playing")
        self.lbl_time = QtWidgets.QLabel("t = 0.00 s")
        self.chk_markers = QtWidgets.QCheckBox("Show markers")
        self.chk_markers.setToolTip("Toggle vertical event lines and labels")
        self.chk_markers.setChecked(True)
        self.chk_markers.stateChanged.connect(self._toggle_markers)

        self.chk_filter = QtWidgets.QCheckBox("Filter (0.5–100 Hz + 50 Hz notch)")
        self.chk_filter.setToolTip(f"Sampling rate ≈ {self.fs:.2f} Hz. Applies zero-phase filtfilt notch then bandpass.")
        self.chk_filter.setChecked(bool(start_filtered))
        self.chk_filter.stateChanged.connect(self._toggle_filter)

        ctrl.addWidget(self.lbl_status)
        ctrl.addStretch(1)
        ctrl.addWidget(self.lbl_time)
        ctrl.addStretch(1)
        ctrl.addWidget(self.chk_markers)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.chk_filter)
        layout.addLayout(ctrl)

        # Grid of 4x4 plots
        self.grid = pg.GraphicsLayoutWidget()
        layout.addWidget(self.grid, stretch=1)

        pg.setConfigOptions(antialias=True, useOpenGL=False)

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
            p.setDownsampling(mode='peak')
            p.setClipToView(True)
            curve = p.plot([], [], pen=pg.intColor(k, n))
            self.plots[k] = p
            self.curves[k] = curve

        # Marker overlay
        self.markers = MarkerOverlay(self.plots)

        # Legend
        legend_text = "Markers: " + ", ".join([f"{code}={label}" for code, label in TRIGGER_MAP.items()])
        self.legend = QtWidgets.QLabel(legend_text)
        layout.addWidget(self.legend)

        # Shortcuts
        QtWidgets.QShortcut(QtCore.Qt.Key_Space, self, activated=self._toggle_pause)
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self, activated=lambda: self._nudge(+1.0))
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self, activated=lambda: self._nudge(-1.0))
        QtWidgets.QShortcut(QtCore.Qt.Key_M, self, activated=lambda: self._toggle_markers(self.chk_markers.isChecked()))
        QtWidgets.QShortcut(QtCore.Qt.Key_F, self, activated=self._shortcut_filter)
        QtWidgets.QShortcut(QtCore.Qt.Key_Q, self, activated=self.close)
        QtWidgets.QShortcut(QtCore.Qt.Key_Escape, self, activated=self.close)

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
        if not self.paused:
            return
        if self.i_cursor >= len(self.t_rel):
            return
        current_t = self.t_rel[self.i_cursor-1] if self.i_cursor > 0 else 0.0
        target = max(0.0, min(self.total_duration, current_t + seconds))
        idx = int(np.searchsorted(self.t_rel, target, side='left'))
        self._seek(idx)

    def _seek(self, new_index):
        self.i_cursor = max(0, min(int(new_index), len(self.t_rel)))
        self.buffers_t.clear()
        self.buffers_idx.clear()
        t_now = self.t_rel[self.i_cursor-1] if self.i_cursor>0 else 0.0
        t_min = max(0.0, t_now - self.win_sec)
        i_start = int(np.searchsorted(self.t_rel, t_min, side='left'))
        for i in range(i_start, self.i_cursor):
            self.buffers_t.append(self.t_rel[i])
            self.buffers_idx.append(i)
        self._draw_curves()
        self.markers.clear()
        self._add_markers_up_to_time(t_now, clear=False)
        self._update_x_range(t_now)

    def _toggle_markers(self, state=None):
        if isinstance(state, bool):
            enabled = state
        else:
            enabled = self.chk_markers.isChecked()
        self.markers.setEnabled(enabled)

    def _shortcut_filter(self):
        self.chk_filter.setChecked(not self.chk_filter.isChecked())

    def _toggle_filter(self, _state=None):
        # Lazily compute filtered data on first enable
        if self.chk_filter.isChecked():
            if self.data_filt is None:
                if not _SCIPY_OK:
                    QtWidgets.QMessageBox.critical(self, "Filtering unavailable",
                        "SciPy is not installed, so filtering cannot be applied.\n\n"
                        "Install SciPy (e.g., pip install scipy) and try again.")
                    self.chk_filter.setChecked(False)
                    return
                # Precompute filtered data (blocking)
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
                try:
                    self.data_filt = apply_filters(self.data_raw, self.fs)
                except Exception as e:
                    QtWidgets.QApplication.restoreOverrideCursor()
                    QtWidgets.QMessageBox.critical(self, "Filtering error",
                        f"Failed to filter data:\n{e}")
                    self.chk_filter.setChecked(False)
                    self.data_filt = None
                    return
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
        # Force redraw using the other data source
        self._draw_curves()

    # ---- Timer tick ----
    def _on_tick(self):
        if self.paused:
            return
        elapsed_s = self.wall_clock.elapsed() / 1000.0
        t_now = min(self.total_duration, elapsed_s * self.speed)
        self.lbl_time.setText(f"t = {t_now:.2f} s")

        n = len(self.t_rel)
        if self.i_cursor < n:
            i_end = int(np.searchsorted(self.t_rel, t_now, side='right'))
            if i_end > self.i_cursor:
                for i in range(self.i_cursor, i_end):
                    self.buffers_t.append(self.t_rel[i])
                    self.buffers_idx.append(i)
                self.i_cursor = i_end
                self._draw_curves()

        self._add_markers_up_to_time(t_now)
        self._update_x_range(t_now)

        if math.isclose(t_now, self.total_duration, rel_tol=0, abs_tol=1e-3):
            self._toggle_pause()

    def _draw_curves(self):
        # Decide which dataset to render from
        use_filtered = self.chk_filter.isChecked() and (self.data_filt is not None)
        data_src = self.data_filt if use_filtered else self.data_raw

        max_pts = 4000
        t = np.array(self.buffers_t, dtype=float)
        idx = np.array(self.buffers_idx, dtype=int)
        if len(t) == 0:
            for ch in range(data_src.shape[1]):
                self.curves[ch].setData([], [])
            return

        if len(t) > max_pts:
            step = max(1, len(t)//max_pts)
            t_draw = t[::step]
            idx_draw = idx[::step]
        else:
            t_draw = t
            idx_draw = idx

        for ch in range(data_src.shape[1]):
            y = data_src[idx_draw, ch]
            self.curves[ch].setData(t_draw, y)

        # Update filter indicator in status text
        filt_txt = "ON" if use_filtered else "OFF"
        base = "⏸ Paused" if self.paused else "▶ Playing"
        self.lbl_status.setText(f"{base} • Filter: {filt_txt} • Fs≈{self.fs:.2f} Hz")

    def _update_x_range(self, t_now):
        t_min = max(0.0, t_now - self.win_sec)
        t_max = max(self.win_sec, t_now)
        for p in self.plots.values():
            p.setXRange(t_min, t_max, padding=0.0)
        self.markers.trim_to_window(t_min)

    def _add_markers_up_to_time(self, t_now, clear=False):
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
    ap.add_argument("--file", default="", help="Path to CSV file.")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (1.0 = real-time).")
    ap.add_argument("--win", type=float, default=15.0, help="Visible time window (seconds).")
    ap.add_argument("--interval", type=int, default=30, help="GUI update interval (ms).")
    ap.add_argument("--filter", action="store_true", help="Start with filtering enabled (0.5–100 Hz + 50 Hz notch).")
    args = ap.parse_args()

    # Resolve default CSV if not provided
    path = args.file if args.file else "{default_csv}"
    if not path:
        print("ERROR: Please provide --file path to your CSV.")
        sys.exit(2)

    # Detect channel columns
    head = pd.read_csv(path, nrows=5)
    non_ch = {"timestamp","type","marker","user_settings","user_info"}
    ch_cols = [c for c in head.columns if c not in non_ch]
    if len(ch_cols) != 16:
        print(f"[Info] Detected {len(ch_cols)} channels: {ch_cols}")

    # Load EEG & markers
    print("Loading EEG…")
    ts, data, ch_labels = load_eeg(path, channel_cols=ch_cols)
    print(f"Loaded EEG: {data.shape[0]} samples × {data.shape[1]} channels. Duration ~= {ts[-1]-ts[0]:.1f} s")

    print("Loading markers…")
    events = load_events_only(path)
    print(f"Loaded markers: {len(events)} events")

    app = QtWidgets.QApplication(sys.argv)
    w = EEGReplayer(ts, data, ch_labels, events, speed=args.speed, win_sec=args.win, interval_ms=args.interval, start_filtered=args.filter)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    default_csv = str(
        Path(__file__).resolve().parents[2]
        / "data"
        / "20250816_172416_2blocks"
        / "20250816_172416_eeg_with_markers_renamed.csv"
    )
    main()
