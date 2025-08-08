from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class EEGPlotter(QtWidgets.QMainWindow):
    def __init__(self, recorder, update_interval_ms=100):
        super().__init__()
        self.recorder = recorder

        # set up pyqtgraph plot
        self.plot_widget = pg.PlotWidget(title="Real-time EEG (channel 1)")
        self.setCentralWidget(self.plot_widget)
        self.curve = self.plot_widget.plot(pen='y')

        # timer to update the plot
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(update_interval_ms)

        # storage for a scrolling window (optional)
        self.ts_window = []
        self.val_window = []
        self.max_points = 500

    def update_plot(self):
        data = self.recorder.get_eeg_buffer()
        if not data:
            return

        # unpack and, say, take channel 0 from each sample
        for ts, sample in data:
            self.ts_window.append(ts)
            self.val_window.append(sample[0])  # first channel

        # keep only the last max_points
        self.ts_window = self.ts_window[-self.max_points:]
        self.val_window = self.val_window[-self.max_points:]

        self.curve.setData(self.ts_window, self.val_window)
