# -*- coding: utf-8 -*-
"""
"""

import os
import subprocess
import time
import ConfigParser
import shutil
import copy   
    

    
    
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
    status, res = sample(param)
    _param = copy.copy(param)
    lengthIdx = _param.index('-length')
    lengthIdx = lengthIdx + 1
    length = int(_param[lengthIdx])
    fileIdx = _param.index('-start_file')
    fileIdx = fileIdx+1
    
    #fileType = _param[fileIdx] [:(len(_param[fileIdx])-4)]    
    #startFile = _param[fileIdx][:-4]       
    #newFile = startFile+'0'+fileType
    startFile = _param[fileIdx]      
    newFile = startFile+'0'
	
    
    with open(newFile,'w+') as tempFile:
        tempFile.write(res)     
    #copy_rename(startFile,newFile)
    #changing params from start_file to start_text    
    #_param = changeParamValue(param,"sample","start_file","")
    #_param = changeParamValue(_param,"sample","start_file_val","")
    #_param = changeParamValue(_param,"sample","start_text","-start_text")
    #_param = changeParamValue(_param,"sample","start_text_val",res)
    
    _param[lengthIdx] = str(length + length)
    for i in range(1,numOfIterations):                
        
        _param[fileIdx] = newFile
        status, res = sample(_param)
        res = res[length:]
        if status != 0:
            return 1
        #os.remove(newFile)     
        newFile = startFile+str(i)
        with open(newFile,'w+') as tempFile:
            tempFile.write(res)
    
        
    os.rename(newFile,startFile+"_Final_OUTPUT") 
    #os.remove(newFile)               
    return 0         
    
         
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
    "cv/checkpoint_10000.t7","-length","2000","-start_file","input text/Tal_King","-gpu","-1"]
    sampleFile(sampleList,10)




#Tal


#start()
#status, proc=train(["th","train.lua","-input_h5",\
#        "data/tiny-shakespeare.h5","-input_json","data/tiny-shakespeare.json","-gpu","-1"])
#print (status)
#time.sleep(10)
#print (abortTraining(proc))
#