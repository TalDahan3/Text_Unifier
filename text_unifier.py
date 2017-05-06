import sys
import os
import subprocess
#import time
import rnnWrapper as runner
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMessageBox

qtCreatorFile = "text_unifier.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.init()        
        
    def initMsg(self):
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setWindowTitle("Info!")
        self.msg.setText("")
        
    def initClickables(self):
        self.createNewMachineBtn.clicked.connect(self.setNewMachinePage)
        self.loadMachineBtn.clicked.connect(self.loadMachine)
        self.backBtn.clicked.connect(self.goBack)
        self.generateTextBtn.clicked.connect(self.generateText)
        self.loadDBBtn.clicked.connect(self.loadDBFile)
        self.trainMachineBtn.clicked.connect(self.trainNewMachine)
        self.abortBtn.clicked.connect(self.abortTraining)
        self.openOutputFileBtn.clicked.connect(self.openingTextOutput)
        
    def init(self):
        self.initMsg()
        self.initClickables()
        self.setMainPage()
        
    def setNewMachinePage(self):
        self.mainWidget.setCurrentIndex(1)
        self.abortBtn.setVisible(False)
        self.trainMachineBtn.setEnabled(False)
        self.progressBar.setVisible(False)
        self.backBtn.setVisible(True)
        
    def setGenTextPage(self):
        self.mainWidget.setCurrentIndex(2)
        self.backBtn.setVisible(True)
        self.openOutputFileBtn.setEnabled(False)
        
    def setMainPage(self):
        self.mainWidget.setCurrentIndex(0)
        self.backBtn.setVisible(False)
        
    def goBack(self):
        temp_index = self.mainWidget.currentIndex()
#        if temp_index == 1:
#            self.setMainPage
#        elif temp_index == 2:
        self.setMainPage()

    def abortTraining(self):
        self.msg.setText("Stopped at checkpoint number " + str(self.lastCheckPoint) + "!\n")
        self.msg.exec_()
        self.setMainPage()
    
    def loadMachine(self):
        #fileName = QtGui.QFileDialog.getSaveFileName(self, 'Open DB file', '', '*.txt')
        tmp_machine_name = QtGui.QFileDialog.getOpenFileName(self, 'Open DB file', '', '*.t7')
        if tmp_machine_name != '':
            tmp_machine_name = tmp_machine_name.split('/')
            machine_name = str(tmp_machine_name[-2]) + '/' + str(tmp_machine_name[-1])
            status,self.txt=runner.sample(["th","sample.lua","-checkpoint",\
            machine_name,"-length","2000","-gpu","-1"])
            print(status)
            self.setGenTextPage()
    
    def loadDBFile(self):
        tmp_machine_name = QtGui.QFileDialog.getOpenFileName(self, 'Open DB file', '', '*.txt')
        
        if tmp_machine_name != '':
            self.trainMachineBtn.setEnabled(True)
        
    def trainNewMachine(self):
        self.lastCheckPoint = 0
        self.abortBtn.setVisible(True)
        self.progressBar.setVisible(True)
        tmp_machine_name = self.machineNameField.text()
        print tmp_machine_name
        self.repaint()
        #self.setGenTextPage()
        
    def generateText(self):
        tmp_length = self.lengthToGenerateField.value()
        self.openOutputFileBtn.setEnabled(True)
        self.outputTextField.setText(self.txt)
        #runner.start()
        
    def openingTextOutput(self):
#        os.system("start "+'kljsda.txt')
        try:
            if os.name == 'nt':
                retcode = subprocess.call("start " + 'kljsda.txt' , shell=True)
            else:
                retcode = subprocess.call("xdg-open " + 'kljsda.txt' , shell=True)
            if retcode < 0:
                print >>sys.stderr, "Child was terminated by signal", -retcode
            else:
                print >>sys.stderr, "Child returned", retcode
        except OSError, e:
            print >>sys.stderr, "Execution failed:", e
        
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
