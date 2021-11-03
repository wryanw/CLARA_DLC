#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:21:35 2019

@author: bioelectrics
"""
import subprocess
from multiprocessing import Process
import glob
import os
from pathlib import PurePath
import cv2
from deeplabcut.CLARA_DLC import CLARA_DLC_utils as clara
import shutil

class CLARA_compress(Process):
    def __init__(self):
        super().__init__()
        
    def run(self):
        try:
            dirlist = list()
            destlist = list()
            user_cfg = clara.read_config()
            read_dir = user_cfg['raw_data_dir']
            write_dir = user_cfg['compressed_video_dir']
            prev_date_list = [name for name in os.listdir(read_dir)]
            for f in prev_date_list:
                unit_dirR = os.path.join(read_dir, f, user_cfg['unitRef'])
                unit_dirW = os.path.join(write_dir, f, user_cfg['unitRef'])
                if os.path.exists(unit_dirR):
                    prev_expt_list = [name for name in os.listdir(unit_dirR)]
                    for s in prev_expt_list:
                        dirlist.append(os.path.join(unit_dirR, s))
                        destlist.append(os.path.join(unit_dirW, s))
                            
            
            for ndx, s in enumerate(dirlist):
                avi_list = os.path.join(s, '*.avi')
                vid_list = glob.glob(avi_list)
                if not os.path.exists(destlist[ndx]):
                    os.makedirs(destlist[ndx])
                if len(vid_list):
                    proc = list()
                    for v in vid_list:
                        vid_name = PurePath(v)
                        dest_path = os.path.join(destlist[ndx], vid_name.stem+'.mp4')
                        passtest = self.testVids(v,str(dest_path))
                        if not passtest:
                            command = 'ffmpeg -y -i ' + v + ' -c:v libx265 -preset veryfast -strict experimental -crf 17 -loglevel quiet ' + str(dest_path)
                            proc.append(subprocess.Popen([command, '/usr/bin'], shell=True, stdout=subprocess.PIPE))
                    for p in proc:
                        p.wait()
                    passvals = list()
                    for v in vid_list:
                        vid_name = PurePath(v)
                        dest_path = os.path.join(destlist[ndx], vid_name.stem+'.mp4')
                        passval = self.testVids(v,str(dest_path))
                        passvals.append(passval)
                        if passval:
                            os.remove(v)
                            print('Deleted original video')
                        else:
                            print('Error while compressing')
                metafiles = glob.glob(os.path.join(s,'*'))
                for m in metafiles:
                    mname = PurePath(m).name
                    mdest = os.path.join(destlist[ndx],mname)
                    if not os.path.isfile(mdest):
                        shutil.copyfile(m,mdest)
        except Exception as ex:
            print(ex)
            
    def testVids(self, v, dest_path):
        try:
            vid = cv2.VideoCapture(v)
            numberFramesA = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            vid = cv2.VideoCapture(str(dest_path))
            numberFramesB = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if numberFramesA == numberFramesB:
                passval = True
            else:
                passval = False
        except:
            passval = False
            
        return passval
