# -*- coding: utf-8 -*-
"""
"""

import os
import subprocess
import time
import ConfigParser
import shutil
import copy

def getTrainPercentage():
    try:
        with open("stdout.txt","r") as stdout:
            lines = stdout.readlines()[-2]
            line_list = lines.split(' ')
            for index, line in enumerate(line_list):
                if line == 'i':
                    iter_num = float(line_list[index+2])
                    total_num = line_list[index+4]
                    total_num = float(total_num[:-1])
                    stdout.close()
                    return iter_num, (iter_num/total_num)*100
            stdout.close()
    except:
        print "here i failed"
    return -1, -1
    
def getIterNum():
    with open('Results.txt', 'r') as f:
        return sum(1 for _ in f)
        f.close()
    
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
    
    return 0
         
def train (param):
    """
    **************************************************************************
    This function returns a Handle to the training process
    The training process is running in the background after the functions ends.
    **************************************************************************
    """
    flag = False

    with open("stdout.txt","w") as out:
        proc = subprocess.Popen(param, stdout=out, stderr=subprocess.STDOUT)
        flag = True
        out.close()
       
    if flag == False:
        print "error loading training"
            
    return 0, proc
    
def abortTraining (proc):   
    if type(proc) is subprocess.Popen:
        proc.terminate()
        return 0
    else:
        return -1