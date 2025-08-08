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

        # Containers for FFT subplots
        self.fft_plots = {}
        self.fft_curves_individual = {}

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
        self.band_selector.addItems(["None", "Theta (4-8 Hz)", "Alpha (8-12 Hz)", "Beta (12-30 Hz)", "Gamma (30-100 Hz)"])
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

        # Stacked plot area for time and FFT views
        self.plot_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.plot_stack)

        cols = 4
        rows = int(np.ceil(self.num_channels / cols))

        # Time-domain widget
        self.time_widget = pg.GraphicsLayoutWidget()
        self.plot_stack.addWidget(self.time_widget)
        self.plots = {}
        self.curves = {}
        for idx in range(self.num_channels):
            row = idx // cols
            col = idx % cols
            p = self.time_widget.addPlot(row=row, col=col, title=f"Channel {idx+1}")
            p.setLabel('bottom', 'Time', units='s')
            p.setLabel('left', 'Amplitude', units='uV')
            c = p.plot(pen='y')
            self.plots[idx] = p
            self.curves[idx] = c

        # FFT widget
        self.fft_widget = pg.GraphicsLayoutWidget()
        self.plot_stack.addWidget(self.fft_widget)
        # Individual FFT subplots
        for idx in range(self.num_channels):
            row = idx // cols
            col = idx % cols
            p_fft = self.fft_widget.addPlot(row=row, col=col, title=f"FFT Channel {idx+1}")
            p_fft.setLabel('bottom', 'Frequency', units='Hz')
            p_fft.setLabel('left', 'Magnitude')
            c_fft = p_fft.plot(pen=pg.intColor(idx, self.num_channels))
            self.fft_plots[idx] = p_fft
            self.fft_curves_individual[idx] = c_fft
        # Combined FFT plot for single-channel view
        self.fft_plot = self.fft_widget.addPlot(row=rows, col=0, colspan=cols)
        self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.fft_plot.setLabel('left', 'Magnitude')
        self.fft_curves = {idx: self.fft_plot.plot(pen=pg.intColor(idx, self.num_channels)) for idx in range(self.num_channels)}

        # Start in time-domain view
        self.plot_stack.setCurrentWidget(self.time_widget)

        # Connect controls
        self.show_btn.clicked.connect(self.on_show_channel)
        self.all_btn.clicked.connect(self.on_show_all)
        self.fft_btn.clicked.connect(self.on_show_fft)
        self.filter_btn.clicked.connect(self.on_apply_filter)

    def on_show_channel(self):
        ch = int(self.channel_selector.currentText()) - 1
        self.selected_channel = ch
        self.show_fft = False
        # Switch to time view and show only selected channel plot
        self.plot_stack.setCurrentWidget(self.time_widget)
        for idx, p in self.plots.items():
            p.setVisible(idx == ch)

    def on_show_all(self):
        self.selected_channel = None
        self.show_fft = False
        # Switch to time view and show all channels
        self.plot_stack.setCurrentWidget(self.time_widget)
        for p in self.plots.values():
            p.setVisible(True)

    def on_show_fft(self):
        self.show_fft = True
        # Switch to FFT view
        self.plot_stack.setCurrentWidget(self.fft_widget)
        if self.selected_channel is None:
            # Show individual FFT subplots
            for p_fft in self.fft_plots.values():
                p_fft.setVisible(True)
            self.fft_plot.setVisible(False)
        else:
            # Show combined FFT for selected channel
            for p_fft in self.fft_plots.values():
                p_fft.setVisible(False)
            self.fft_plot.setVisible(True)
            self.fft_plot.setTitle(f"FFT Channel {self.selected_channel+1}")

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
        # Update time-domain curves
        if not self.show_fft:
            if self.selected_channel is not None:
                y = np.array(self.val_windows[self.selected_channel])
                t = np.array(self.ts_windows[self.selected_channel])
                if len(y) > 1 and self.filter_active:
                    dt = np.mean(np.diff(t))
                    y = self.apply_filter(y, dt)
                self.curves[self.selected_channel].setData(t, y)
            else:
                for idx in range(self.num_channels):
                    y = np.array(self.val_windows[idx])
                    t = np.array(self.ts_windows[idx])
                    if len(y) > 1 and self.filter_active:
                        dt = np.mean(np.diff(t))
                        y = self.apply_filter(y, dt)
                    self.curves[idx].setData(t, y)
        else:
            # FFT mode
            if self.selected_channel is None:
                # Individual FFT plots
                for idx in range(self.num_channels):
                    y = np.array(self.val_windows[idx])
                    t = np.array(self.ts_windows[idx])
                    if len(y) > 1:
                        dt = np.mean(np.diff(t))
                        if self.filter_active:
                            y = self.apply_filter(y, dt)
                        yf = np.fft.rfft(y)
                        xf = np.fft.rfftfreq(len(y), d=dt)
                        self.fft_curves_individual[idx].setData(xf, np.abs(yf))
            else:
                # Combined FFT view for selected channel
                y = np.array(self.val_windows[self.selected_channel])
                t = np.array(self.ts_windows[self.selected_channel])
                if len(y) > 1:
                    dt = np.mean(np.diff(t))
                    if self.filter_active:
                        y = self.apply_filter(y, dt)
                    yf = np.fft.rfft(y)
                    xf = np.fft.rfftfreq(len(y), d=dt)
                    # clear previous curves
                    for curve in self.fft_curves.values():
                        curve.setData([], [])
                    self.fft_curves[self.selected_channel].setData(xf, np.abs(yf))
