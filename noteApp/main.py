import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication

from MainWindow import Ui_MainWindow


class App:
    def __init__(self, appWindow):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(appWindow)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    init = App(window)
    window.show()
    sys.exit(app.exec_())
