from app import App
import sys
from PyQt5 import QtWidgets
import warnings
warnings.filterwarnings('ignore')
app = QtWidgets.QApplication(sys.argv)
window = App()
window.setWindowTitle('Classification of microearray')
app.exec_()
