# -*- coding: utf-8 -*-
"""
"""

import os
import subprocess
import copy

def preProcess (param):

    proc = subprocess.check_output(param)
    if (proc.find("Total vocabulary") < 0):
        return -1
        
    return 0

def sample(param):
   
    proc = subprocess.check_output(param)
    if (proc == ""):
        return -1,proc
    return 0 , proc
    
def sampleFile(param,numOfIterations): 
    checkpoint_idx = 3       
    _param = copy.copy(param)
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
    
    paramForLossCalc = ["th","LossCalc.lua","-init_from",param[checkpoint_idx],"-input_h5",\
    ".h5","-input_json",".json","-gpu","-1",]
    
    input_idx = 3
    outputH5_idx = 5
    outputJSON_idx = 7
    
    #creating file for the loss calculation resaults - override if exists
    with open("Results.txt","w") as resFile:
        resFile.close()
        
    for i in range(0,numOfIterations):                
        
        _param[fileIdx] = newFile
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
        if(preProcess(paramForPreProc) <> 0):
            return 1
        paramForPreProc = ["python","scripts/preprocess.py","--input_txt",\
        "","--output_h5",".h5","--output_json",".json","--val_frac","0.3","--test_frac","0.3"]
        
        #calculate total loss
        
        paramForLossCalc[outputH5_idx] = newFile+paramForLossCalc[outputH5_idx]
        paramForLossCalc[outputJSON_idx] = newFile+paramForLossCalc[outputJSON_idx]
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
            
        paramForLossCalc = ["th","LossCalc.lua","-init_from",param[checkpoint_idx],"-input_h5",\
        ".h5","-input_json",".json","-gpu","-1",]
              
        
    os.rename(newFile,startFile+"_Final_OUTPUT")               
    return 0, startFile+"_Final_OUTPUT"       

if __name__ == "__main__":
    with open('test_params.txt', 'r') as f:
        lines = f.readlines()        
        for index, line in enumerate(lines):
            if index == 0:            
                params = line.split(' ')
            elif index == 1:
                iter_num = int(line)
            else:
                f.close()
    for index, param in enumerate(params):
        if param == '\n':
            del params[index]
#        elif param == 'input':
#            params[index] = param + ' ' + params[index+1]
#            del params[index+1]
    sampleFile(params,iter_num)


