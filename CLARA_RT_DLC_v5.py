#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:46:43 2019

@author: bioelectrics
"""
import os, sys, linecache
from multiprocessing import Process
from queue import Empty
import numpy as np
from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.CLARA_DLC import CLARA_DLC_utils_v2 as clara
from pathlib import Path
import tensorflow as tf
import time
import serial

class CLARA_RT(Process):
    def __init__(self, dlcq, p2read, array, dlc, aq, autop, autos, px, py, frm, frate, bipolar):
        super().__init__()
        self.dlcq = dlcq
        self.p2read = p2read
        self.array = array
        self.dlc = dlc
        self.aq = aq
        self.autop = autop
        self.autos = autos
        self.px = px
        self.py = py
        self.frm = frm
        self.frate = frate
        self.bipolar = bipolar
        
    def run(self):
        camct = len(self.array)
        frmct = len(self.array[0][0])
        bs = frmct*camct
        print('child: ',os.getpid())
        user_cfg = clara.read_config()
        proj_cfg = auxiliaryfunctions.read_config(user_cfg['config_path'])
        bodyparts = proj_cfg['bodyparts']
        parts = list()
        categories = list()
        for cat in bodyparts.keys():
            categories.append(cat)
        for key in categories:
            for ptname in bodyparts[key]:
                parts.append(ptname)
        bodyparts = parts
        shuffle = user_cfg['shuffle']
        trainingsetindex = user_cfg['trainingsetindex']
        trainFraction = proj_cfg['TrainingFraction'][trainingsetindex]
        
        if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
            del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
        tf.reset_default_graph()
        
        modelfolder=os.path.join(proj_cfg["project_path"],str(auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,proj_cfg)))
        path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
        try:
            dlc_cfg = load_config(str(path_test_config))
        except FileNotFoundError:
            raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle, trainFraction))
        # Check which snapshots are available and sort them by # iterations
        try:
          Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
        except FileNotFoundError:
          raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(self.shuffle,self.shuffle))
    
        snapshotindex = proj_cfg['snapshotindex']
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)
        ##################################################
        # Load and setup CNN part detector
        ##################################################
        cpt = user_cfg['sideCrop']
        nx = cpt[1]
        ny = cpt[3]
        dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        frames = np.empty((bs, ny, nx, 3), dtype='ubyte') # this keeps all frames in a batch
        dlc_cfg['batch_size']=bs
        sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
        dtype = 'uint8'
        size = nx*ny
#        print(size)
        shape = [nx, ny]
#        print(shape)
        finb = np.repeat([i for i in range(frmct)],camct)
        camsInBatch = np.tile([i for i in range(camct)],frmct)
        record = False
        self.p2read.put('Ready')
        frame = np.zeros((ny,nx), dtype='ubyte')
        pdata = np.zeros((camct*frmct, 3 * len(bodyparts)))
        while True:
            try:
                msg = self.dlcq.get(block=False)
                try:
                    if msg == 'initSerial':
                        ser = serial.Serial(user_cfg['COM'], baudrate=115200)
                    elif msg == 'stopSerial':
                        ser.close()
                    elif msg == 'Q':
                        ser.write(b'Q')
                    elif msg == 'R':
                        ser.write(b'R')
                    elif msg == 'B':
                        self.bipolar = 1
                    elif msg == 'M':
                        self.bipolar = 0
                    elif msg == 'S':
                        if self.bipolar:
                            ser.write(b'T')
                        else:
                            ser.write(b'S')
                    elif msg == 'recordPrep':
                        path_base = self.dlcq.get()
                        f = open('%s_events.txt' % path_base, 'w')
                        record = True
                        self.p2read.put('done')
                    elif msg == 'Start':
                        obj2get = 0
                        act_sel = 4
                        sys_timer = time.time()
                        rch_timer = [0,0,0,0,0]
                        total_anal = 0
                        benchmark = 0
                        start = time.time()
                        PorigXY = list()
                        PprevXY = list()
                        for ndx in range(camct):
                            pxy = np.zeros((2 ,1))
                            PorigXY.append(pxy)
                            PprevXY.append(pxy)
                        PinHandX = np.zeros((5,1))
                        PinHandX[:] = np.nan
                        failtest = False
                        successtest = False
                        runDLC = True
                        detail = 2
                        fixCt = 0
#                        postY = np.nan
                        fixHolder = False
                        if user_cfg['unitRef'] == 'unit05':
                            time2wait = 15
                        else:
                            time2wait = 5
                        # Wait until a new batch of frames is ready
                        while self.aq.value:
                            try:
                                msg = self.dlcq.get(block=False)
                                if msg == 'pause':
                                    runDLC = False
                                    self.px[ndx].value = 0
                                    self.px[ndx].value = 0
                                    if 0 < act_sel < 5:    
                                        if self.autop:
                                            ser.write(b'R')
                                elif msg == 'resume':
                                    runDLC = True
                                    act_sel = 4
                                    fixCt = 0
                                    sys_timer = time.time()
                                elif msg == 'Q':
                                    ser.write(b'Q')
                                elif msg == 'R':
                                    ser.write(b'R')
                                elif msg == 'B':
                                    self.bipolar = 1
                                elif msg == 'M':
                                    self.bipolar = 0
                                elif msg == 'S':
                                    if self.bipolar:
                                        ser.write(b'T')
                                    else:
                                        ser.write(b'S')
                                    if record:
                                        f.write('stim - %d\n' % self.frm.value)
                            except Empty:
                                pass
                            
                            if not runDLC:
                                continue
                            
                            if obj2get == 0:
                                nextObj2get = 1
                            elif obj2get == 1:
                                nextObj2get = 2
                            elif obj2get == 2:
                                nextObj2get = 0
                            while self.dlc.value != nextObj2get:
                                if self.aq.value:
                                    break
                                else:
                                    pass
                            if self.dlc.value == 0:
                                obj2get = 2
                            elif self.dlc.value == 1:
                                obj2get = 0
                            elif self.dlc.value == 2:
                                obj2get = 1
                            for ndx, cn in enumerate(camsInBatch):
                                frame[:,:] = np.frombuffer(self.array[cn][obj2get][finb[ndx]].get_obj(), dtype, size).reshape(shape)
                                for fn in range(3):
                                    frames[ndx,:,:,fn] = frame
                            pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                            pdata[:,:] = pose
                            pref = 0
                            for frmref in range(frmct):
                                retrieving = False
                                pellet_on_post = [0,0,0]
                                pellet_fallFast = [0,0,0]
                                pellet_fallSlow = [0,0,0]
                                pellet_seen = [0,0,0]
                                tongue_seen = [0,0,0]
                                hand_seen = False
                                distP = [np.nan,np.nan,np.nan]
                                PcurrXY = list()
                                deltaPY = np.nan
                                for ndx in range(camct):                                        
                                    PcurrXY.append(np.zeros((2 ,1)))
                                    fdata = pdata[pref,:]
                                    pref+=1
                                    fdata = fdata.reshape(len(bodyparts),3)
                                    if ndx == 2:
                                        nanHP = np.isnan(PinHandX)
                                        nanHPtot = np.sum(nanHP)
                                        numHP = PinHandX[~nanHP]
                                        if nanHPtot < 2:
                                            hpdelta = -np.nanmean(np.diff(numHP))
                                            if hpdelta < -3:
                                                retrieving = True
                                        PinHandX = np.roll(PinHandX,1)
                                        PinHandX[0] = np.nan
                                    
                                    if ndx == 0:
                                        keylist = ['Hand','Pellet','Other']
                                    elif ndx == 1:
                                        keylist = ['Pellet','Other']
                                    else:
                                        keylist = ['Pellet','Other']
                                    for key in keylist:
                                        testlist = list()
                                        ndxlist = list()
                                        for ptname in proj_cfg['bodyparts'][key]:
                                            bpndx = bodyparts.index(ptname)
                                            ndxlist.append(bpndx)
                                            bp_test = fdata[bpndx,2]
                                            testlist.append(bp_test)
                                            
                                        if key == 'Other':
                                            bpndx = bodyparts.index('Tongue')
                                            bp_test = fdata[bpndx,2]
                                            if np.amax(bp_test) > 0.9:
                                                tongue_seen[ndx] = 1
                                            continue
                                        elif key == 'Hand':
                                            if np.amax(testlist) > 0.9: 
                                                hand_seen = True
                                        elif key == 'Pellet':
                                            if act_sel == 5:
                                                pthresh = 0.9
                                            else:
                                                pthresh = 0.99
                                            if np.amax(testlist) > pthresh:
                                                drawndx = ndxlist[np.argmax(testlist)]
                                                if bodyparts[drawndx] == 'InHand' and ndx < 2:
                                                    PprevXY[ndx][0] = np.nan
                                                    PprevXY[ndx][1] = np.nan
                                                    continue
                                                if bodyparts[drawndx] == 'InHand' and ndx == 2:
                                                    PinHandX[0] = fdata[drawndx,0]
                                                    hand_seen = True
                                                pellet_seen[ndx] = 1
                                                PcurrXY[ndx][0] = fdata[drawndx,0]
                                                PcurrXY[ndx][1] = fdata[drawndx,1]
                                                distP[ndx] = np.linalg.norm(PorigXY[ndx]-PcurrXY[ndx])
                                                if distP[ndx] < 5:
                                                    pellet_on_post[ndx] = 1
                                                if not np.isnan(PprevXY[ndx][0]):
                                                    deltaPY = PprevXY[ndx][1]-PcurrXY[ndx][1]
                                                    deltaPX = PprevXY[ndx][0]-PcurrXY[ndx][0]
                                                    if ndx == 2:
                                                        deltaPY = -deltaPY
                                                    fallA = deltaPY < -5 or deltaPX < -5
                                                    fallB = deltaPY < -15 or deltaPX < -15
                                                    fallC = abs(deltaPY) < 50 and abs(deltaPX) < 50
                                                    if fallA and fallC:
                                                        pellet_fallSlow[ndx] = 1
                                                    if fallB and fallC:
                                                        pellet_fallFast[ndx] = 1
#                                                self.px[ndx].value = PcurrXY[ndx][0]
#                                                self.py[ndx].value = PcurrXY[ndx][1]
                                                PprevXY[ndx] = PcurrXY[ndx]
                                            else:
#                                                self.px[ndx].value = 0
#                                                self.py[ndx].value = 0
                                                PprevXY[ndx][0] = np.nan
                                                PprevXY[ndx][1] = np.nan
                                                
                                if sum(tongue_seen) > 0:
                                    PinHandX[1] = np.nan
                                if sum(pellet_seen) > 1 and not act_sel == 4:
                                    pellet_seen = True
                                elif sum(pellet_seen) == 2 and act_sel == 4:
                                    pellet_seen = True
                                else:
                                    pellet_seen = False
                                if sum(pellet_on_post) > 1:
                                    pellet_on_post = True
                                else:
                                    pellet_on_post = False
                                
                                if benchmark < time2wait:
                                    continue
                                if not self.autop and not self.autos:
                                    continue
                                failbool = False
                                if act_sel == 0:
                                    failtest = False
                                    successtest = False
                                    if (time.time()-sys_timer) > 1:
                                        if self.autop:
                                            ser.write(b'Q')
                                        if record:
                                            f.write('raise - %d\n' % self.frm.value)
                                        act_sel = 1
                                        sys_timer = time.time()
                                        for ndx in range(3):
                                            self.px[ndx].value = 0
                                            self.py[ndx].value = 0
                                            
                                        if detail > 1:
                                            print('raising post')
                                        
                                elif act_sel == 1:
                                    if frmref == 0:
                                        if (time.time()-sys_timer) > 5:
                                            act_sel = 3
                                            sys_timer = time.time()
                                            if fixHolder:
                                                sys_timer+=5
                                                fixHolder = False
                                                if self.autop:
                                                    ser.write(b'F')
                                            if detail > 1:
                                                print('drop ready')
                                elif act_sel == 3:
                                    if (time.time()-sys_timer) > 1 and not hand_seen:
                                        act_sel = 4
                                        sys_timer = time.time()
                                        if self.autop:
                                            ser.write(b'R')
                                        if detail > 1:
                                            print('drop post')
                                    if (time.time()-sys_timer) > 10:
                                        if detail > 1:
                                            print('try loading again...')
                                        if self.autop:
                                            ser.write(b'R')
                                        act_sel = 4
                                        sys_timer = time.time()
                                        
                                elif act_sel == 4:
                                    if (time.time()-sys_timer) > 0.25 and pellet_seen:
                                        act_sel = 5
                                        rch_timer = [0,0,0,0,0]
                                        if user_cfg['unitRef'] == 'unit05':
                                            ser.write(b'Z')
                                        if record:
                                            f.write('placed - %d\n' % self.frm.value)
                                        if detail > 1:
                                            print('pellet placed')
                                        fixCt = 0
                                        for ndx in range(3):
                                            PorigXY[ndx][0] = PcurrXY[ndx][0]
                                            PorigXY[ndx][1] = PcurrXY[ndx][1]
                                            self.px[ndx].value = PcurrXY[ndx][0]
                                            self.py[ndx].value = PcurrXY[ndx][1]
                                                
                                    elif (time.time()-sys_timer) > 2:
                                        fixCt += 1
                                        act_sel = 0
                                        sys_timer = time.time()
                                        if fixCt > 4:
                                            runDLC = False
                                            self.px[ndx].value = -1
                                            self.px[ndx].value = -1
                                        elif fixCt > 1:
                                            fixHolder = True
                                            if detail > 1:
                                                print('fixing dispenser')
                                        if detail > 1:
                                            print('pellet placement fail')
                                    
                                if act_sel == 5:
                                    
                                    if failtest:
                                        if pellet_on_post:
                                            rch_timer[2] = 0
                                            rch_timer[1]+=1
                                            if rch_timer[1] > 10:
                                                failtest = False
                                                if detail > 1:
                                                    print('false alarm')
                                        else:
                                            rch_timer[2]+=1
                                            rch_timer[1] = 0
                                            
                                        if rch_timer[2] > 10:
                                            failbool = True
                                            if detail > 0:
                                                print('reach fail')
                                            
                                    if (sum(pellet_fallSlow) > 0 or sum(pellet_fallFast) > 0) and not pellet_on_post:
                                        rch_timer[4]+=1
                                        fallTestA = sum(pellet_fallSlow) > 0 and rch_timer[4] == 4
                                        fallTestB = sum(pellet_fallFast) > 0 and rch_timer[4] == 3
                                        fallTestC = sum(pellet_fallSlow) > 1
                                    
                                        if fallTestA or fallTestB or fallTestC:
                                            if not failtest:
                                                successtest = False
                                                failtest = True
                                                rch_timer[1] = 0
                                                rch_timer[2] = 0
                                                if detail > 1:
                                                    print('fail test')
                                    else:
                                        rch_timer[4] = 0
                                    
                                    if successtest:
                                        if not hand_seen and not pellet_on_post:
                                            rch_timer[3]+=1
                                            if rch_timer[3] > 10:
                                                act_sel = 0
                                                if detail > 0:
                                                    print('reach success!')
                                                if self.autos:
                                                    if self.bipolar:
                                                        ser.write(b'T')
                                                    else:
                                                        ser.write(b'S')
                                                if record:
                                                    f.write('stim - %d\n' % self.frm.value)
                                                sys_timer = time.time()
                                        elif pellet_on_post:
                                            successtest = False
                                        else:
                                            rch_timer[3] = 0
                                            
                                    if retrieving and not pellet_on_post:
                                        if not successtest:
                                            successtest = True
                                            rch_timer[3] = 0
                                            if detail > 1:
                                                print('success test')
                        
                                    if not pellet_seen:
                                        rch_timer[0]+=1
                                        if rch_timer[0] > self.frate*5:
                                            if detail > 1:
                                                print('pellet lost')
                                            if record:
                                                f.write('lost - %d\n' % self.frm.value)
                                            act_sel = 0
                                            sys_timer = time.time()
                                    else:
                                        rch_timer[0] = 0
                                    
                                    if failbool:
                                        act_sel = 0
                                        sys_timer = time.time()
                                        if record:
                                            f.write('fail - %d\n' % self.frm.value)
                            
                            if total_anal/self.frate == round(total_anal/self.frate):
                                benchmark = time.time()-start
#                                print(int((total_anal)/benchmark))
                            total_anal+=frmct      
                             
                        if 0 < act_sel < 5:    
                            if self.autop:
                                ser.write(b'R')
                        if record:
                            f.close()
                            record = False
                            

                except:
                    exc_type, exc_obj, tb = sys.exc_info()
                    f = tb.tb_frame
                    lineno = tb.tb_lineno
                    filename = f.f_code.co_filename
                    linecache.checkcache(filename)
                    line = linecache.getline(filename, lineno, f.f_globals)
                    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
                    self.p2read.put([0, 0])
            
            except Empty:
                pass
        
        
