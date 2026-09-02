#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: Indrani
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import radar
import sip
import threading



class step02_single_tone(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "step02_single_tone")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2e6
        self.pulse_width = pulse_width = 10e-6
        self.pri = pri = 100e-6
        self.emitter2_pulse_width = emitter2_pulse_width = 4e-6
        self.emitter2_pri = emitter2_pri = 70e-6
        self.pulse_samples = pulse_samples = int(round(pulse_width * samp_rate))
        self.pri_samples = pri_samples = int(round(pri * samp_rate))
        self.emitter2_pulse_samples = emitter2_pulse_samples = int(round(emitter2_pulse_width * samp_rate))
        self.emitter2_pri_samples = emitter2_pri_samples = int(round(emitter2_pri * samp_rate))
        self.wait_samples = wait_samples = pri_samples - pulse_samples
        self.packet_samples = packet_samples = pri_samples
        self.noise_power = noise_power = 0.0025
        self.noise_amplitude = noise_amplitude = 0.05
        self.emitter_rf_freq = emitter_rf_freq = 3.2e9
        self.emitter_id = emitter_id = 1
        self.emitter2_wait_samples = emitter2_wait_samples = emitter2_pri_samples - emitter2_pulse_samples
        self.emitter2_rf_freq = emitter2_rf_freq = 8e9
        self.emitter2_packet_samples = emitter2_packet_samples = emitter2_pri_samples
        self.emitter2_id = emitter2_id = 2
        self.emitter2_baseband_freq = emitter2_baseband_freq = -250e3
        self.emitter2_amplitude = emitter2_amplitude = 0.7
        self.emitter1_amplitude = emitter1_amplitude = 1.0
        self.baseband_freq = baseband_freq = 100e3

        ##################################################
        # Blocks
        ##################################################

        self.radar_signal_generator_sync_pulse_c_0_0 = radar.signal_generator_sync_pulse_c(emitter2_packet_samples, [emitter2_pulse_samples], [emitter2_wait_samples], emitter2_amplitude, '')
        self.radar_signal_generator_sync_pulse_c_0 = radar.signal_generator_sync_pulse_c(packet_samples, [pulse_samples], [wait_samples], emitter1_amplitude, '')
        self.qtgui_time_sink_x_0 = qtgui.time_sink_c(
            1024, #size
            samp_rate, #samp_rate
            "", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)

        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")

        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(False)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_0.enable_control_panel(False)
        self.qtgui_time_sink_x_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(2):
            if len(labels[i]) == 0:
                if (i % 2 == 0):
                    self.qtgui_time_sink_x_0.set_line_label(i, "Re{{Data {0}}}".format(i/2))
                else:
                    self.qtgui_time_sink_x_0.set_line_label(i, "Im{{Data {0}}}".format(i/2))
            else:
                self.qtgui_time_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_0_win)
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, #size
            window.WIN_BLACKMAN_hARRIS, #wintype
            emitter_rf_freq, #fc
            samp_rate, #bw
            "", #name
            1,
            None # parent
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis((-140), 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)



        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])

        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        self.blocks_multiply_xx_0_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_add_xx_1 = blocks.add_vcc(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_sig_source_x_0_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, emitter2_baseband_freq, 1.0, 0.0, 0.0)
        self.analog_sig_source_x_0 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, baseband_freq, 1.0, 0.0, 0.0)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amplitude, 42)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_1, 1))
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.analog_sig_source_x_0_0, 0), (self.blocks_multiply_xx_0_0, 1))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_add_xx_1, 0))
        self.connect((self.blocks_add_xx_1, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.blocks_add_xx_1, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.blocks_multiply_xx_0_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.radar_signal_generator_sync_pulse_c_0, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.radar_signal_generator_sync_pulse_c_0_0, 0), (self.blocks_multiply_xx_0_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "step02_single_tone")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_emitter2_pri_samples(int(round(self.emitter2_pri * self.samp_rate)))
        self.set_emitter2_pulse_samples(int(round(self.emitter2_pulse_width * self.samp_rate)))
        self.set_pri_samples(int(round(self.pri * self.samp_rate)))
        self.set_pulse_samples(int(round(self.pulse_width * self.samp_rate)))
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.analog_sig_source_x_0_0.set_sampling_freq(self.samp_rate)
        self.qtgui_freq_sink_x_0.set_frequency_range(self.emitter_rf_freq, self.samp_rate)
        self.qtgui_time_sink_x_0.set_samp_rate(self.samp_rate)

    def get_pulse_width(self):
        return self.pulse_width

    def set_pulse_width(self, pulse_width):
        self.pulse_width = pulse_width
        self.set_pulse_samples(int(round(self.pulse_width * self.samp_rate)))

    def get_pri(self):
        return self.pri

    def set_pri(self, pri):
        self.pri = pri
        self.set_pri_samples(int(round(self.pri * self.samp_rate)))

    def get_emitter2_pulse_width(self):
        return self.emitter2_pulse_width

    def set_emitter2_pulse_width(self, emitter2_pulse_width):
        self.emitter2_pulse_width = emitter2_pulse_width
        self.set_emitter2_pulse_samples(int(round(self.emitter2_pulse_width * self.samp_rate)))

    def get_emitter2_pri(self):
        return self.emitter2_pri

    def set_emitter2_pri(self, emitter2_pri):
        self.emitter2_pri = emitter2_pri
        self.set_emitter2_pri_samples(int(round(self.emitter2_pri * self.samp_rate)))

    def get_pulse_samples(self):
        return self.pulse_samples

    def set_pulse_samples(self, pulse_samples):
        self.pulse_samples = pulse_samples
        self.set_wait_samples(self.pri_samples - self.pulse_samples)

    def get_pri_samples(self):
        return self.pri_samples

    def set_pri_samples(self, pri_samples):
        self.pri_samples = pri_samples
        self.set_packet_samples(self.pri_samples)
        self.set_wait_samples(self.pri_samples - self.pulse_samples)

    def get_emitter2_pulse_samples(self):
        return self.emitter2_pulse_samples

    def set_emitter2_pulse_samples(self, emitter2_pulse_samples):
        self.emitter2_pulse_samples = emitter2_pulse_samples
        self.set_emitter2_wait_samples(self.emitter2_pri_samples - self.emitter2_pulse_samples)

    def get_emitter2_pri_samples(self):
        return self.emitter2_pri_samples

    def set_emitter2_pri_samples(self, emitter2_pri_samples):
        self.emitter2_pri_samples = emitter2_pri_samples
        self.set_emitter2_packet_samples(self.emitter2_pri_samples)
        self.set_emitter2_wait_samples(self.emitter2_pri_samples - self.emitter2_pulse_samples)

    def get_wait_samples(self):
        return self.wait_samples

    def set_wait_samples(self, wait_samples):
        self.wait_samples = wait_samples

    def get_packet_samples(self):
        return self.packet_samples

    def set_packet_samples(self, packet_samples):
        self.packet_samples = packet_samples

    def get_noise_power(self):
        return self.noise_power

    def set_noise_power(self, noise_power):
        self.noise_power = noise_power

    def get_noise_amplitude(self):
        return self.noise_amplitude

    def set_noise_amplitude(self, noise_amplitude):
        self.noise_amplitude = noise_amplitude
        self.analog_noise_source_x_0.set_amplitude(self.noise_amplitude)

    def get_emitter_rf_freq(self):
        return self.emitter_rf_freq

    def set_emitter_rf_freq(self, emitter_rf_freq):
        self.emitter_rf_freq = emitter_rf_freq
        self.qtgui_freq_sink_x_0.set_frequency_range(self.emitter_rf_freq, self.samp_rate)

    def get_emitter_id(self):
        return self.emitter_id

    def set_emitter_id(self, emitter_id):
        self.emitter_id = emitter_id

    def get_emitter2_wait_samples(self):
        return self.emitter2_wait_samples

    def set_emitter2_wait_samples(self, emitter2_wait_samples):
        self.emitter2_wait_samples = emitter2_wait_samples

    def get_emitter2_rf_freq(self):
        return self.emitter2_rf_freq

    def set_emitter2_rf_freq(self, emitter2_rf_freq):
        self.emitter2_rf_freq = emitter2_rf_freq

    def get_emitter2_packet_samples(self):
        return self.emitter2_packet_samples

    def set_emitter2_packet_samples(self, emitter2_packet_samples):
        self.emitter2_packet_samples = emitter2_packet_samples

    def get_emitter2_id(self):
        return self.emitter2_id

    def set_emitter2_id(self, emitter2_id):
        self.emitter2_id = emitter2_id

    def get_emitter2_baseband_freq(self):
        return self.emitter2_baseband_freq

    def set_emitter2_baseband_freq(self, emitter2_baseband_freq):
        self.emitter2_baseband_freq = emitter2_baseband_freq
        self.analog_sig_source_x_0_0.set_frequency(self.emitter2_baseband_freq)

    def get_emitter2_amplitude(self):
        return self.emitter2_amplitude

    def set_emitter2_amplitude(self, emitter2_amplitude):
        self.emitter2_amplitude = emitter2_amplitude

    def get_emitter1_amplitude(self):
        return self.emitter1_amplitude

    def set_emitter1_amplitude(self, emitter1_amplitude):
        self.emitter1_amplitude = emitter1_amplitude

    def get_baseband_freq(self):
        return self.baseband_freq

    def set_baseband_freq(self, baseband_freq):
        self.baseband_freq = baseband_freq
        self.analog_sig_source_x_0.set_frequency(self.baseband_freq)




def main(top_block_cls=step02_single_tone, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
