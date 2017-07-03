#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import subprocess
import time
import rnnWrapper as runner
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtGui import QMessageBox
from thread import start_new_thread

qtCreatorFile = "text_unifier.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtGui.QMainWindow, Ui_MainWindow, QtCore.QObject):
    
    new_msg = QtCore.pyqtSignal(str) 
    prog_bar = QtCore.pyqtSignal(int)
    text_field = QtCore.pyqtSignal(str)
    openOutputEnabled = QtCore.pyqtSignal(str)
    
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        QtCore.QObject.__init__(self)
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
        self.startPreProcessBtn.clicked.connect(self.trainNewMachine)
        self.abortBtn.clicked.connect(self.abortTraining)
        self.openOutputFileBtn.clicked.connect(self.openingTextOutput)
        self.fileToUnifyBtn.clicked.connect(self.loadFileToUnify)
        self.startTrainingBtn.clicked.connect(self.startTraining)
        self.new_msg.connect(self.popUpMsg)
        self.prog_bar.connect(self.updateProgressBar)
        self.text_field.connect(self.updateTextField)
        self.openOutputEnabled.connect(self.openOutputChangeToEnabled)
        
    def init(self):
        self.initMsg()
        self.initClickables()
        self.setMainPage()
        
    def setNewMachinePage(self):
        self.startPreProcessBtn.setEnabled(False)
        self.backBtn.setVisible(True)
        self.machineNameField.setEnabled(True)
        self.valFracSpinBox.setEnabled(True)
        self.val_frac = 0.1
        self.loaded_db = ''
        self.valFracSpinBox.setValue(0.1)
        self.machineNameField.setText('')
        self.machineNameField.setPlaceholderText('Enter machine name')
        self.param_list_1 = []
        self.mainWidget.setCurrentIndex(1)
        
    def setGenTextPage(self):
        self.text_field.emit('')
        self.generateTextBtn.setEnabled(False)
        self.txt = ''
        self.loaded_unify_file = ''
        self.fileToUnifyLbl.setText('')
        self.useGPUCheckBox.setChecked(True)
        self.num_of_iterations = 4000
        self.numOfIterationsSpinBox.setValue(4000)
        self.backBtn.setVisible(True)
        self.openOutputFileBtn.setEnabled(False)
        self.mainWidget.setCurrentIndex(2)
        
    def setMainPage(self):
        self.before_db = ''
        self.proc = ''
        self.machine_name = ''
        self.tmp_machine_name = 'default'
        self.loadDBBtn.setEnabled(True)
        self.gpu_enabled = True
        self.mainWidget.setCurrentIndex(0)
        self.backBtn.setVisible(False)
        
    def setTrainingPage(self):
        self.startTrainingBtn.setEnabled(True)
        self.trainingInSession = False
        self.progressBar.setValue(0)
        self.backBtn.setVisible(True)
        self.mainWidget.setCurrentIndex(3)
        self.abortBtn.setVisible(False)
        self.progressBar.setVisible(False)
        self.param_list_2 = []
        self.modelType = 'LSTM'
        self.num_layers = 3
        self.num_input_nodes = 64
        self.num_hidden_nodes = 128
        self.num_output_nodes = 128
        self.num_max_epochs = 50
        self.modelTypeRNN.setEnabled(True)
        self.modelTypeLSTM.setEnabled(True)
        self.inputNodesSpinBox.setEnabled(True)
        self.numberOfLayersSpinBox.setEnabled(True)
        self.outputNodesSpinBox.setEnabled(True)
        self.hiddenNodesSpinBox.setEnabled(True)
        self.epochSpinBox.setEnabled(True)
        self.useGPUCheckBox_2.setEnabled(True)
        self.modelTypeRNN.setChecked(False)
        self.modelTypeLSTM.setChecked(True)
        self.numberOfLayersSpinBox.setValue(3)
        self.inputNodesSpinBox.setValue(64)
        self.outputNodesSpinBox.setValue(128)
        self.hiddenNodesSpinBox.setValue(128)
        self.epochSpinBox.setValue(50)
        self.useGPUCheckBox_2.setChecked(True)   
        
        
    def openOutputChangeToEnabled(self, path):
        self.pathToOutputFile = path
        self.openOutputFileBtn.setEnabled(True)
        
    def goBack(self):
        self.setMainPage()

    def updateProgressBar(self, value):
        self.progressBar.setValue(value)

    def popUpMsg(self, msg):
        self.msg.setText(msg)
        self.msg.exec_()
        if msg != 'Finished unifying!\n':
            self.setMainPage()
        
    def abortTraining(self):
        self.trainingInSession = False
        runner.abortTraining(self.proc)
        self.new_msg.emit("Stopped at checkpoint number " + str(self.lastCheckPoint//1000) + "!\n")
        
    def trainingFailed(self):
        self.new_msg.emit("Couldn't train!\n")
        
    def samplingFailed(self):
        self.new_msg.emit("Couldn't unify!\n")
        
    def updateTextField(self, text):
        self.outputTextField.setText(text)
    
    def loadMachine(self):
        tmp_machine_name = QtGui.QFileDialog.getOpenFileName(self, 'Open DB file', os.getcwd()+'/cv', '*.t7')
        if tmp_machine_name != '':
            tmp_machine_name = tmp_machine_name.split('/')
            self.machine_name = str(tmp_machine_name[-2]) + '/' + str(tmp_machine_name[-1])
            self.setGenTextPage()
    
    def loadDBFile(self):
        tmp_machine_name = QtGui.QFileDialog.getOpenFileName(self, 'Open DB file', os.getcwd()+'/data', '*.txt')
        if tmp_machine_name != '':
            tmp_machine_name = tmp_machine_name.split('/') 
            self.loaded_db = tmp_machine_name[-1]
            if tmp_machine_name != '':
                self.before_db = str(tmp_machine_name[-2]) + '/'
                self.loaded_db = str(tmp_machine_name[-2]) + '/' + self.loaded_db
                self.startPreProcessBtn.setEnabled(True)
        
    def trainNewMachine(self):
        self.val_frac = self.valFracSpinBox.value()
        self.tmp_machine_name = str(self.machineNameField.text())
        if self.tmp_machine_name == '':
            self.tmp_machine_name == 'default'
        self.loadDBBtn.setEnabled(False)
        self.machineNameField.setEnabled(False)
        self.valFracSpinBox.setEnabled(False)
        self.repaint()
        self.param_list_1 = ['python', 'scripts/preprocess.py', '--input_txt', str(self.loaded_db),
                        '--output_h5', self.before_db + self.tmp_machine_name + '.h5', '--output_json',
                        self.before_db + self.tmp_machine_name + '.json', '--val_frac', str(self.val_frac)]
        runner.preProcess(self.param_list_1)
        self.setTrainingPage()
    
    def startTraining(self):
        self.modelTypeRNN.setEnabled(False)
        self.modelTypeLSTM.setEnabled(False)
        self.inputNodesSpinBox.setEnabled(False)
        self.numberOfLayersSpinBox.setEnabled(False)
        self.outputNodesSpinBox.setEnabled(False)
        self.hiddenNodesSpinBox.setEnabled(False)
        self.epochSpinBox.setEnabled(False)
        self.useGPUCheckBox_2.setEnabled(False)
        if self.modelTypeLSTM.isChecked() == True:
            model_type = 'lstm'
        else:
            model_type = 'rnn'
        number_of_layers = str(self.numberOfLayersSpinBox.value()-1)
        number_of_input_nodes = str(self.inputNodesSpinBox.value())
        number_of_output_nodes = str(self.outputNodesSpinBox.value())
        number_of_hidden_nodes = str(self.hiddenNodesSpinBox.value())
        max_epochs = str(self.epochSpinBox.value())
        gpu_enabled = self.useGPUCheckBox_2.isChecked()
        self.param_list_2 = ['th','train.lua','-input_h5', self.before_db + self.tmp_machine_name + '.h5',
                             '-input_json', self.before_db + self.tmp_machine_name + '.json', 
                             '-model_type', model_type, '-num_layers', number_of_layers, 
                             '-wordvec_size', number_of_input_nodes, '-rnn_size', number_of_hidden_nodes,
                             '-max_epochs', max_epochs, '-checkpoint_name', 'cv/' + self.tmp_machine_name, '-gpu']
        if gpu_enabled == True:
            self.param_list_2.append('0')
        else:
            self.param_list_2.append('-1')   
        self.lastCheckPoint = 0
        self.percentage = 0
        self.abortBtn.setVisible(True)
        self.progressBar.setVisible(True)
        train_status, self.proc = runner.train(self.param_list_2)
        self.trainingInSession = True
        self.startTrainingBtn.setEnabled(False)
        start_new_thread(self.training,())
    
    def training(self):
        while self.trainingInSession == True:
            time.sleep(2)
            self.lastCheckPoint, self.percentage = runner.getTrainPercentage()
            if self.percentage < 100:
                self.prog_bar.emit(self.percentage)
            elif self.percentage >= 100:
                self.trainingInSession = False
                self.new_msg.emit("Finished training!\n")
                
    def generateText(self):
        self.samplingInSession = False
        self.num_of_iterations = self.numOfIterationsSpinBox.value()
        self.gpu_enabled = self.useGPUCheckBox.isChecked()
        if self.loaded_unify_file != '':
            tmp_param_list = ['th','sample.lua','-checkpoint', self.machine_name,
                              '-start_file',self.loaded_unify_file, '-gpu']
            if self.gpu_enabled == True:
                tmp_param_list.append('0')
            else:                             
                tmp_param_list.append('-1')
            with open('test_params.txt', 'w') as f:
                tmp_str = ''           
                for param in tmp_param_list:
                    tmp_str += str(param) + " "
                tmp_str += '\n'
                f.write(tmp_str)
                f.write(str(self.num_of_iterations))
                f.close()
            param = ["python","sample_file_call.py"]
            subprocess.Popen(param)
            self.samplingInSession = True
            start_new_thread(self.generating,())
                
    def generating(self):
        tmp_txt = ''
        self.text_field.emit("I've started generating unified text!")
        while self.samplingInSession == True:
            time.sleep(2)
            tmp_iter = (runner.getIterNum()-1)
            print (str(tmp_iter))
            print (self.loaded_unify_file)
            if tmp_iter < self.num_of_iterations:
                tmp_txt = "I'm at iteration number: " + str(tmp_iter+1) + ' out of ' + str(self.num_of_iterations) + '\n'            
                self.text_field.emit(tmp_txt)
            else:
                with open(self.loaded_unify_file + '_Final_OUTPUT', 'r') as f:
                    tmp_txt = f.read()
                    self.text_field.emit(tmp_txt)
                self.openOutputEnabled.emit(self.loaded_unify_file + '_Final_OUTPUT')
                self.samplingInSession = False
                self.new_msg.emit("Finished unifying!\n")
        
    def openingTextOutput(self):
        if self.pathToOutputFile != '':
            try:
                if os.name == 'nt':
                    retcode = subprocess.call("start " + str(self.pathToOutputFile), shell=True)
                else:
                    retcode = subprocess.call("xdg-open " + str(self.pathToOutputFile), shell=True)
                if retcode < 0:
                    print >>sys.stderr, "Child was terminated by signal", -retcode
                else:
                    print >>sys.stderr, "Child returned", retcode
            except OSError, e:
                print >>sys.stderr, "Execution failed:", e
            
    def loadFileToUnify(self):
        tmp_machine_name = QtGui.QFileDialog.getOpenFileName(self, 'Open file to unify', os.getcwd() + '/input_text', '')
        if tmp_machine_name != '':       
            tmp_machine_name = tmp_machine_name.split('/') 
            self.loaded_unify_file = str(tmp_machine_name[-2]) + '/' + str(tmp_machine_name[-1])
            self.openOutputFileBtn.setEnabled(False)
            self.generateTextBtn.setEnabled(True)
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
