# -*- coding: utf-8 -*-
"""
"""

import os
import subprocess
import time
import ConfigParser
import shutil
import copy
import math

def getTrainPercentage():
    with open("stdout.txt","r") as stdout:
        lines = stdout.readlines()[-2]
        line_list = lines.split(' ')
        for index, line in enumerate(line_list):
            if line == 'i':
                iter_num = float(line_list[index+2])
                total_num = line_list[index+4]
                total_num = float(total_num[:-1])
                #"{0:.3f}".format(((()))
                return iter_num, (iter_num/total_num)*100
    print "couldn't read"
    return -1, -1
    
def getCharsOnly(bigString):
    results = ""    
    for line in bigString:
      for char in line:
          results += char
    return results
    
def copy_rename(old_file_name, new_file_name):
    src_dir= os.path.join(os.curdir ,"input text")
    dst_dir= os.path.join(os.curdir , "subfolder")
    src_file = os.path.join(src_dir, old_file_name)
    shutil.copy(src_file,dst_dir)
    
    dst_file = os.path.join(dst_dir, old_file_name)
    new_dst_file_name = os.path.join(dst_dir, new_file_name)
    os.rename(dst_file, new_dst_file_name)


def getParamValue(param,section,key):
    section = str(section).upper()
    cfg = ConfigParser.ConfigParser()
    cfg.read("paramConfig.ini")
    idx = cfg.get(section,key)
    return param[int(idx)]
    
def changeParamValue(param,section,key,val):
    
    section = str(section).upper()
    cfg = ConfigParser.ConfigParser()
    cfg.read("paramConfig.ini")
    idx = cfg.get(section,key)
    param[int(idx)] = val
    return param

def preProcess (param):

    proc = subprocess.check_output(param)
    if (proc.find("Total vocabulary") < 0):
        return -1
    print (proc)
    
    print ("Pre Training success")
    return 0

def sample(param):
   
    proc = subprocess.check_output(param)
    if (proc == ""):
        return -1,proc
    #print (proc)
    return 0 , proc
    
def sampleFile(param,numOfIterations):        
    _param = copy.copy(param)
    checkpointIndex = 3
    #add length variable to list
    _param.append('-length')
    _param.append('0')
    lengthIdx = _param.index('-length')
    lengthIdx = lengthIdx + 1       
    fileIdx = _param.index('-start_file')
    fileIdx = fileIdx+1
    
    #getting file length and updating the param list   
    with open(_param[fileIdx],"r") as f:
        _param[lengthIdx] = len(f.read())
        
    length = int(_param[lengthIdx])
    
    _param[lengthIdx] = str(length + length)
        
    startFile = _param[fileIdx]      
    newFile = startFile 
    
    paramForPreProc =["python","scripts/preprocess.py","--input_txt",\
    "","--output_h5",".h5","--output_json",".json","--val_frac","0.3","--test_frac","0.3"]
   # paramForPreProc =["python","scripts/preprocess.py","--input_txt",\
    #newFile,"--output_h5",newFile+".h5","--output_json",newFile+".json","--val_frac","0.3","--test_frac","0.3"]
    
    paramForLossCalc = ["th","LossCalc.lua","-init_from","cv/checkpoint_6040.t7","-input_h5",\
    ".h5","-input_json",".json","-gpu","-1",]
 #   paramForLossCalc = ["th","LossCalc.lua","-init_from",_param[checkpointIndex],"-input_h5",\
 #   newFile+".h5","-input_json",newFile+".json","-gpu","-1",]
    print "before"
    print (paramForPreProc)
    print (paramForLossCalc)
    
    
    input_idx = 3
    outputH5_idx = 5
    outputJSON_idx = 7
    
    #creating file for the loss calculation resaults - override if exists
    with open("Results.txt","w") as resFile:
        resFile.write("Start Sampling")
        
        
    
    for i in range(0,numOfIterations):                
        
        #_param[fileIdx] = newFile
        _param[fileIdx] = newFile
        print " before sample"
        print _param
        status, res = sample(_param)
        res = res[length:]
        if status != 0:
            return 1
        
        #creating a new file in each iteration
        newFile = startFile+str(i)
        with open(newFile,'w+') as tempFile:
            tempFile.write(res)
            
        #Pre process in order to calculate total loss
        paramForPreProc[input_idx] = newFile
        paramForPreProc[outputH5_idx] = newFile+paramForPreProc[outputH5_idx]
        paramForPreProc[outputJSON_idx] = newFile+paramForPreProc[outputJSON_idx]

        print (paramForPreProc)
        if(preProcess(paramForPreProc) <> 0):
            return 1
        paramForPreProc = ["python","scripts/preprocess.py","--input_txt",\
        "","--output_h5",".h5","--output_json",".json","--val_frac","0.3","--test_frac","0.3"]
        
        #calculate total loss
        
        paramForLossCalc[outputH5_idx] = newFile+paramForLossCalc[outputH5_idx]
        paramForLossCalc[outputJSON_idx] = newFile+paramForLossCalc[outputJSON_idx]
        paramForLossCalc[outputJSON_idx] = _param[checkpointIndex]
        print(paramForLossCalc)
        proc = subprocess.check_output(paramForLossCalc)
        if ( proc == ""):
            return 1
        
        #parsing results
        resList = proc.split("val_loss =")
        val_loss = resList[1]
        val_loss = '\n'+val_loss.strip()
        
        #print resaults to file
        with open("Results.txt","a") as resFile:
            resFile.write(val_loss)
            
        paramForLossCalc = ["th","LossCalc.lua","-init_from",_param[checkpointIndex],"-input_h5",\
        ".h5","-input_json",".json","-gpu","-1",]
              
        
    os.rename(newFile,startFile+"_Final_OUTPUT") 
    #os.remove(newFile)               
    return 0, startFile+"_Final_OUTPUT"

         
def train (param):
    """
    **************************************************************************
    This function returns a Handle to the training process
    The training process is running in the background after the functions ends.
    **************************************************************************
    """
    flag = False

    print("#ethalnu Training")
    with open("stdout.txt","r+") as out:
        proc = subprocess.Popen(param, stdout=out, stderr=subprocess.STDOUT)
        flag = True
        time.sleep(4)
        file_data=out.read()
        if (file_data == ""):
            print("Error reading output file")
            return -1, proc
       
    if flag == False:
        print "error loading training"
        
    
    
    print("Training Running")
    
    return 0, proc
    
def abortTraining (proc):   
    if type(proc) is subprocess.Popen:
        proc.terminate()
        print ("Training aborted")
        return 0
    else:
        return -1

def start():    
    preProcess(["python","scripts/preprocess.py","--input_txt",\
    "data/tiny-shakespeare.txt","--output_h5","my_dataTemp.h5","--output_json","my_dataTemp.json"])
    #train([])
    sample(["th","sample.lua","-checkpoint",\
    "cv/checkpoint_10000.t7","-length","2000","-gpu","-1"])
    


if __name__ == "__main__":

    
    sampleList = ["th","sample.lua","-checkpoint",\
    "cv/checkpoint_6000.t7","-start_file","input text/AZZ","-gpu","-1"]
    sampleFile(sampleList,1)

    #trainList = ["th","train.lua","-input_h5",\
    #"data/big.h5","-input_json","data/big.json","-gpu","-1","-max_epochs","20"]
    #train(trainList)
