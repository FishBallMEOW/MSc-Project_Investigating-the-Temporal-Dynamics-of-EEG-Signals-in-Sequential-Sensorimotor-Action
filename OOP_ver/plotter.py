from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import numpy as np
import scipy.signal as signal

class EEGPlotter(QtWidgets.QMainWindow):
    def __init__(self, recorder, num_channels=16, update_interval_ms=100, max_points=500):
        super().__init__()
        self.recorder = recorder
        self.num_channels = num_channels
        self.max_points = max_points
        self.selected_channel = None
        self.show_fft = False
        self.filter_active = False
        self.filter_band = None
        self.notch = False

        # Data buffers
        self.ts_windows = {ch: [] for ch in range(self.num_channels)}
        self.val_windows = {ch: [] for ch in range(self.num_channels)}

        # Initialize UI and start timer
        self.init_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval_ms)

    def init_ui(self):
        self.setWindowTitle("Real-time EEG")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Control panel
        control_layout = QtWidgets.QHBoxLayout()
        self.channel_selector = QtWidgets.QComboBox()
        for ch in range(1, self.num_channels + 1):
            self.channel_selector.addItem(str(ch))
        self.show_btn = QtWidgets.QPushButton("Show Channel")
        self.all_btn = QtWidgets.QPushButton("Show All")
        self.fft_btn = QtWidgets.QPushButton("Show FFT")
        # Filter controls
        self.band_selector = QtWidgets.QComboBox()
        self.band_selector.addItem("None")
        self.band_selector.addItem("Theta (4-8 Hz)")
        self.band_selector.addItem("Alpha (8-12 Hz)")
        self.band_selector.addItem("Beta (12-30 Hz)")
        self.band_selector.addItem("Gamma (30-100 Hz)")
        self.notch_check = QtWidgets.QCheckBox("Notch 50 Hz")
        self.filter_btn = QtWidgets.QPushButton("Apply Filter")

        control_layout.addWidget(QtWidgets.QLabel("Channel:"))
        control_layout.addWidget(self.channel_selector)
        control_layout.addWidget(self.show_btn)
        control_layout.addWidget(self.all_btn)
        control_layout.addWidget(self.fft_btn)
        control_layout.addWidget(QtWidgets.QLabel("Band:"))
        control_layout.addWidget(self.band_selector)
        control_layout.addWidget(self.notch_check)
        control_layout.addWidget(self.filter_btn)
        layout.addLayout(control_layout)

        # Plot area
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        self.plots = {}
        self.curves = {}
        cols = 4
        rows = int(np.ceil(self.num_channels / cols))
        for idx in range(self.num_channels):
            row = idx // cols
            col = idx % cols
            p = self.plot_widget.addPlot(row=row, col=col, title=f"Channel {idx+1}")
            p.setLabel('bottom', 'Time', units='s')
            p.setLabel('left', 'Amplitude', units='uV')
            c = p.plot(pen='y')
            self.plots[idx] = p
            self.curves[idx] = c

        # FFT plot
        self.fft_plot = self.plot_widget.addPlot(row=rows, col=0, colspan=cols)
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_curves = {idx: self.fft_plot.plot(pen=pg.intColor(idx, self.num_channels)) for idx in range(self.num_channels)}
        self.fft_plot.setVisible(False)

        # Connect controls
        self.show_btn.clicked.connect(self.on_show_channel)
        self.all_btn.clicked.connect(self.on_show_all)
        self.fft_btn.clicked.connect(self.on_show_fft)
        self.filter_btn.clicked.connect(self.on_apply_filter)

    def on_show_channel(self):
        ch = int(self.channel_selector.currentText()) - 1
        self.selected_channel = ch
        self.show_fft = False
        self.fft_plot.setVisible(False)
        for idx, p in self.plots.items():
            p.setVisible(idx == ch)

    def on_show_all(self):
        self.selected_channel = None
        self.show_fft = False
        self.fft_plot.setVisible(False)
        for p in self.plots.values():
            p.setVisible(True)

    def on_show_fft(self):
        if self.selected_channel is None:
            self.fft_plot.setTitle("FFT All Channels")
        else:
            self.fft_plot.setTitle(f"FFT Channel {self.selected_channel+1}")
        self.show_fft = True
        for p in self.plots.values():
            p.setVisible(False)
        self.fft_plot.setVisible(True)

    def on_apply_filter(self):
        text = self.band_selector.currentText()
        if text.startswith("Theta"):
            self.filter_band = (4, 8)
        elif text.startswith("Alpha"):
            self.filter_band = (8, 12)
        elif text.startswith("Beta"):
            self.filter_band = (12, 30)
        elif text.startswith("Gamma"):
            self.filter_band = (30, 100)
        else:
            self.filter_band = None
        self.notch = self.notch_check.isChecked()
        self.filter_active = (self.filter_band is not None) or self.notch

    def apply_filter(self, y, dt):
        fs = 1.0 / dt if dt > 0 else 1.0
        # bandpass
        if self.filter_band:
            low, high = self.filter_band
            sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
            y = signal.sosfiltfilt(sos, y)
        # notch
        if self.notch:
            b, a = signal.iirnotch(50, 30, fs)
            y = signal.filtfilt(b, a, y)
        return y

    def update_plot(self):
        data = self.recorder.get_eeg_buffer()
        if not data:
            return
        for ts, sample in data:
            for idx, value in enumerate(sample):
                self.ts_windows[idx].append(ts)
                self.val_windows[idx].append(value)
                if len(self.ts_windows[idx]) > self.max_points:
                    self.ts_windows[idx] = self.ts_windows[idx][-self.max_points:]
                    self.val_windows[idx] = self.val_windows[idx][-self.max_points:]
        # plotting
        if self.show_fft:
            # FFT view
            for idx in range(self.num_channels if self.selected_channel is None else 1):
                ch = idx if self.selected_channel is None else self.selected_channel
                y = np.array(self.val_windows[ch])
                t = np.array(self.ts_windows[ch])
                if len(y) > 1:
                    dt = np.mean(np.diff(t))
                    if self.filter_active:
                        y = self.apply_filter(y, dt)
                    yf = np.fft.rfft(y)
                    xf = np.fft.rfftfreq(len(y), d=dt)
                    self.fft_curves[ch].setData(xf, np.abs(yf))
        else:
            if self.selected_channel is not None:
                y = np.array(self.val_windows[self.selected_channel])
                t = np.array(self.ts_windows[self.selected_channel])
                if len(y) > 1 and self.filter_active:
                    dt = np.mean(np.diff(t))
                    y = self.apply_filter(y, dt)
                self.curves[self.selected_channel].setData(self.ts_windows[self.selected_channel], y)
            else:
                for idx in range(self.num_channels):
                    y = np.array(self.val_windows[idx])
                    t = np.array(self.ts_windows[idx])
                    if len(y) > 1 and self.filter_active:
                        dt = np.mean(np.diff(t))
                        y = self.apply_filter(y, dt)
                    self.curves[idx].setData(self.ts_windows[idx], y)
