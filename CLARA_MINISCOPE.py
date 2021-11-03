#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:46:43 2019

@author: bioelectrics
"""
import sys, linecache
from multiprocessing import Process
from queue import Empty
from deeplabcut.CLARA_DLC import CLARA_DLC_utils_v2 as clara
import serial

class CLARA_MS(Process):
    def __init__(self, msq, p2read, aq, frm):
        super().__init__()
        self.msq = msq
        self.p2read = p2read
        self.aq = aq
        self.frm = frm
        
    def run(self):
        user_cfg = clara.read_config()
        self.p2read.put('Ready')
        
        while True:
            try:
                msg = ''
                try:
                    msg = self.msq.get(block=False)
                except Empty:
                    pass
                if msg == 'recordPrep':
                    path_base = self.msq.get()
                    f = open('%s_miniscope.txt' % path_base, 'w')
                    ser = serial.Serial(user_cfg['COM2'], baudrate=9600, timeout = 15)
                    self.p2read.put('done')
                elif msg == 'Start':
                    scfrm = 0
                    Record = True
                    while Record:
                        try:
                            msg = self.msq.get(block=False)
                            if msg == 'Stop':
                                Record = False
                        except Empty:
                            pass
                        
                        if ser.inWaiting():
                            camfrm = ser.readlines(30)
                            for c in camfrm:
                                try:
                                    c = str(c.strip())[2:-1]
                                    c = int(c)
                                    scfrm+=1
                                    f.write('%d\t%d\n' % (scfrm,c))
                                except:
                                    pass
                                if c == 0:
                                    Record = False
                                    print('----- Camera Clock Fail -----')
                                    self.p2read.put('fail')
                    f.close()
                    ser.close()
            except:
                exc_type, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                lineno = tb.tb_lineno
                filename = f.f_code.co_filename
                linecache.checkcache(filename)
                line = linecache.getline(filename, lineno, f.f_globals)
                print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
                                