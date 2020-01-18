import numpy as np
import pygame
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io import wavfile
import soundfile as sf


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1017, 462)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 1111, 421))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")

        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout_2")

        self.widget = QtWidgets.QWidget(self.gridLayoutWidget)
        self.widget.setObjectName("widget")

        self.gridLayoutWidgetGraph = QtWidgets.QWidget(self.widget)
        self.gridLayoutWidgetGraph.setGeometry(QtCore.QRect(9, 9, 1021, 421))
        self.gridLayoutWidgetGraph.setObjectName("gridLayoutWidgetGraph")

        self.gridLayoutGraph = QtWidgets.QGridLayout(self.gridLayoutWidgetGraph)
        self.gridLayoutGraph.setContentsMargins(0, 0, 0, 0)
        self.gridLayoutGraph.setObjectName("gridLayoutGraph")

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.gridLayoutGraph.addWidget(self.canvas)

        self.gridLayout.addWidget(self.widget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1017, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionOpen = QtWidgets.QAction(MainWindow, triggered=self.openWav)
        self.actionOpen.setObjectName("actionOpen")

        self.actionSave = QtWidgets.QAction(MainWindow, triggered=self.saveBoth)
        self.actionSave.setObjectName("actionSave")

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Rozpoznaję nuty - Małgorzata Wiśniewska"))
        self.menuFile.setTitle(_translate("MainWindow", "Plik"))
        self.actionOpen.setText(_translate("MainWindow", "Otwórz"))
        self.actionSave.setText(_translate("MainWindow", "Zapisz wykres"))

    def openWav(self):
        title = self.actionOpen.text()
        dialog = QtWidgets.QFileDialog()
        filename, _filter = dialog.getOpenFileNames(dialog, title, None, 'wav-files: *.wav')
        if not filename:
            return 0
        self.filePath = str(filename[0])
        self.calculateData()
        self.playWav()
        self.drawGraph(self.xLeftLimit, self.xRightLimit, self.yBottomLimitAmp, self.yTopLimitAmp)
        self.checkNote()

    def calculateData(self):
        # samplerate -> częstotliwość, data-> wartości amplitudy dla każdej próbki
        global wave
        self.samplerate, self.data = wavfile.read(self.filePath)
        self.times = np.arange(len(self.data)) / float(self.samplerate)
        self.xLeftLimit = 0
        self.xRightLimit = self.times[-1]
        if self.data.ndim == 2:
            self.dataDimension = self.data[:, 0]
        elif self.data.ndim == 1:
            self.dataDimension = self.data
        self.yBottomLimitAmp = min(self.dataDimension)
        self.yTopLimitAmp = max(self.dataDimension)

        y, samplerate = sf.read(self.filePath)
        if y.ndim == 2:
            x = y[:, 0]
        elif y.ndim == 1:
            x = y
        chunks = np.array_split(x, int(samplerate/2000))
        self.peaks = []

        for chunk in chunks:
            t = np.linspace(0, 1, samplerate)
            wave = chunk
            # compute the magnitude of the Fourier Transform and its corresponding frequency values
            freq_magnitudes = np.abs(np.fft.fft(wave))
            freq_values = np.fft.fftfreq(wave.shape[0], 1/samplerate)
            # find the max. magnitude
            max_positive_freq_idx = np.argmax(freq_magnitudes[:samplerate // 2 + 1])
            self.peaks.append(freq_values[max_positive_freq_idx])

        print(self.peaks)

    def drawGraph(self, xLeft, xRight, yBottom, yTop):
        self.figure.clear()
        self.ampliGraph = self.figure.add_axes([0.1, 0.125, 0.85, 0.8],
                                               ylim=(yBottom, yTop), xlim=(xLeft, xRight))
        self.prepareAmplitudeGraph()
        self.ampliGraph.draw(renderer=None)
        self.canvas.draw()

    def prepareAmplitudeGraph(self):
        self.ampliGraph.plot(self.times, self.dataDimension, linewidth=0.1)
        self.ampliGraph.set_ylabel('Amplituda')
        self.ampliGraph.set_xlabel('Czas [s]')

    def saveBoth(self):
        title = self.actionSave.text()
        dialog = QtWidgets.QFileDialog()
        filename = dialog.getSaveFileName(dialog, title, None, 'Image Files (*.png *.jpg *.jpeg)')
        self.figure.savefig(str(filename[0]))

    def playWav(self):
        pygame.init()
        pygame.mixer.music.load(str(self.filePath))
        pygame.mixer.music.play()

    def checkNote(self):
        i = 0
        a = self.peaks[i]
        while (a < 390 or a > 790) and i < len(self.peaks)-1:
            i += 1
            a = self.peaks[i]
        msgBox = QtWidgets.QMessageBox()
        msgBox.setWindowTitle('Nutka')
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        if (390 <= a < 403.5) or (762 <= a < 790):
            info = ("Nuta zagrana to G")
        elif 403.5 <= a < 427.5:
            info = ("Nuta zagrana to G#")
        elif 427.5 <= a < 451.5:
            info = ("Nuta zagrana to A")
        elif 451.5 <= a < 479.5:
            info = ("Nuta zagrana to A#")
        elif 479.5 <= a < 508:
            info = ("Nuta zagrana to B")
        elif 508 <= a < 538.5:
            info = ("Nuta zagrana to C")
        elif 538.5 <= a < 570.5:
            info = ("Nuta zagrana to C#")
        elif 570.5 <= a < 604.5:
            info = ("Nuta zagrana to D")
        elif 604.5 <= a < 640.5:
            info = ("Nuta zagrana to D#")
        elif 640.5 <= a < 678.5:
            info = ("Nuta zagrana to E")
        elif 678.5 <= a < 719:
            info = ("Nuta zagrana to F")
        elif 719 <= a < 762:
            info = ("Nuta zagrana to F#")
        else:
            info = ("Nie udało się rozpoznać tej nuty.")
            msgBox.setIcon(QtWidgets.QMessageBox.Warning)
        
        msgBox.setText(info)
        msgBox.exec()