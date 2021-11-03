#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:26:20 2019

@author: bioelectrics
"""
import PySpin
from math import floor
import os
from multiprocessing import Process
from queue import Empty
        
class CLARA_DLC_Cam(Process):
    def __init__(self, camq, camq_p2read, array, dlc, camID, idList, cpt, bs, aq, frm):
        super().__init__()
        self.camID = camID
        self.camq = camq
        self.camq_p2read = camq_p2read
        self.array = array
        self.dlc = dlc
        self.idList = idList
        self.cpt = cpt
        self.bs = bs
        self.aq = aq
        self.frm = frm
        
    def run(self):
        print('child: ',os.getpid())
        record = False
        ismaster = False
        while True:
            try:
                msg = self.camq.get(block=False)
#                print(msg)
                try:
                    if msg == 'InitM':
                        ismaster = True
                        system = PySpin.System.GetInstance()
                        cam_list = system.GetCameras()
                        cam = cam_list.GetBySerial(self.camID)
                        cam.Init()
                        cam.CounterSelector.SetValue(PySpin.CounterSelector_Counter0)
                        cam.CounterEventSource.SetValue(PySpin.CounterEventSource_ExposureStart)
                        cam.CounterEventActivation.SetValue(PySpin.CounterEventActivation_RisingEdge)
                        cam.CounterTriggerSource.SetValue(PySpin.CounterTriggerSource_ExposureStart)
                        cam.CounterTriggerActivation.SetValue(PySpin.CounterTriggerActivation_RisingEdge)
                        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
                        cam.V3_3Enable.SetValue(True)
                        cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
                        cam.LineSource.SetValue(PySpin.LineSource_Counter0Active)
                        cam.LineInverter.SetValue(False)
                        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                        cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                        self.camq_p2read.put('done')
                    if msg == 'InitS':
                        system = PySpin.System.GetInstance()
                        cam_list = system.GetCameras()
                        cam = cam_list.GetBySerial(self.camID)
                        cam.Init()
                        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
                        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
                        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_AnyEdge)
                        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                        self.camq_p2read.put('done')
                    elif msg == 'Release':
                        cam.DeInit()
                        del cam
                        for i in self.idList:
                            cam_list.RemoveBySerial(str(i))
                        system.ReleaseInstance() # Release instance
                        self.camq_p2read.put('done')
                    elif msg == 'recordPrep':
                        path_base = self.camq.get()
                        write_frame_rate = 30
                        s_node_map = cam.GetTLStreamNodeMap()
                        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
                        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
                            print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
                            return
                        handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
                        handling_mode.SetIntValue(handling_mode_entry.GetValue())
                        
                        avi = PySpin.SpinVideo()
                        option = PySpin.AVIOption()
                        option.frameRate = write_frame_rate
#                        option = PySpin.MJPGOption()
#                        option.frameRate = write_frame_rate
#                        option.quality = 75
                        
                        print(path_base)
                        avi.Open(path_base, option)
                        f = open('%s_timestamps.txt' % path_base, 'w')
                        start_time = 0
                        capture_duration = 0
                        record = True
                        self.camq_p2read.put('done')
                    elif msg == 'Start':
                        cam.BeginAcquisition()
                        if ismaster:
                            cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
                            cam.LineSource.SetValue(PySpin.LineSource_Counter0Active)
                            self.frm.value = 0
                            self.camq.get()
                            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                        bc = 0
                        while self.aq.value:
                            try:
                                image_result = cam.GetNextImage()
                                image_result = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                                if record:
                                    if start_time == 0:
                                        start_time = image_result.GetTimeStamp()
                                    else:
                                        capture_duration = image_result.GetTimeStamp()-start_time
                                        start_time = image_result.GetTimeStamp()
                                        # capture_duration = capture_duration/1000/1000
                                        avi.Append(image_result)
                                        f.write("%s\n" % round(capture_duration))
                                frame = image_result.GetNDArray()
                                cpt = self.cpt
                                frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                                self.array[self.dlc.value][bc][0:] = frame.flatten()
                                if ismaster:
                                    self.frm.value+=1
                                bc+=1
                                if bc >= self.bs:
                                    bc = 0
                                    if ismaster:
                                        if self.dlc.value == 0:
                                            self.dlc.value = 1
                                        elif self.dlc.value == 1:
                                            self.dlc.value = 2
                                        elif self.dlc.value == 2:
                                            self.dlc.value = 0
                                        
                            except Exception as ex:
                                print(ex)
                                print(self.camID)
                                break
                        self.camq.get()
                        if record:
                            avi.Close()
                            f.close()
                            record = False
                        cam.EndAcquisition()
                        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                        if ismaster:
                            self.dlc.value = 0
                            cam.LineSelector.SetValue(PySpin.LineSelector_Line1)
                            cam.LineSource.SetValue(PySpin.LineSource_FrameTriggerWait)
                            cam.LineInverter.SetValue(True)
                        self.camq_p2read.put('done')
                    elif msg == 'setBinning':
                        binsize = self.camq.get(block=True)
                        cam.BinningHorizontal.SetValue(binsize)
                        cam.BinningVertical.SetValue(binsize)
                    elif msg == 'exp_frmrate':
                        exposure_time_request = self.camq.get(block=True)
                        record_frame_rate = self.camq.get(block=True)
                        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                            print('Unable to disable automatic exposure. Aborting...')
                            continue
                        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                        if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                            print('Unable to set exposure time. Aborting...')
                            continue
                        # Ensure desired exposure time does not exceed the maximum
                        exposure_time_to_set = floor(1/record_frame_rate*1000*1000)
                        if exposure_time_request <= exposure_time_to_set:
                            exposure_time_to_set = exposure_time_request
                        max_exposure = cam.ExposureTime.GetMax()
                        exposure_time_to_set = min(max_exposure, exposure_time_to_set)
                        cam.AcquisitionFrameRateEnable.SetValue(True)
                        cam.ExposureTime.SetValue(exposure_time_to_set)
                        cam.AcquisitionFrameRate.SetValue(record_frame_rate)
                        exposure_time_to_set = cam.ExposureTime.GetValue()
                        record_frame_rate = cam.AcquisitionFrameRate.GetValue()
                        max_exposure = cam.ExposureTime.GetMax()
                        self.camq_p2read.put(exposure_time_to_set)
                        self.camq_p2read.put(record_frame_rate)
                        self.camq_p2read.put(max_exposure)
                    elif msg == 'restoreXYWH':
                        nodemap = cam.GetNodeMap()
                        cam.BinningHorizontal.SetValue(int(1))
                        cam.BinningVertical.SetValue(int(1))
                        node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
                        if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                            print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                            return False
                        # Retrieve entry node from enumeration node
                        node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
                        if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                                node_acquisition_mode_continuous):
                            print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                            return False
                        acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()
                        # Set integer value from entry node as new value of enumeration node
                        node_acquisition_mode.SetIntValue(acquisition_mode_continuous)
                        # Retrieve the enumeration node from the nodemap
                        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
                        if PySpin.IsAvailable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):
                            # Retrieve the desired entry node from the enumeration node
                            node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono8'))
                            if PySpin.IsAvailable(node_pixel_format_mono8) and PySpin.IsReadable(node_pixel_format_mono8):
                                # Retrieve the integer value from the entry node
                                pixel_format_mono8 = node_pixel_format_mono8.GetValue()
                                # Set integer as new value for enumeration node
                                node_pixel_format.SetIntValue(pixel_format_mono8)
                            else:
                                print('Pixel format mono 8 not available...')
                        else:
                            print('Pixel format not available...')
                        # Apply minimum to offset X
                        node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
                        if PySpin.IsAvailable(node_offset_x) and PySpin.IsWritable(node_offset_x):
                            node_offset_x.SetValue(node_offset_x.GetMin())
                        else:
                            print('Offset X not available...')
                        # Apply minimum to offset Y
                        node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
                        if PySpin.IsAvailable(node_offset_y) and PySpin.IsWritable(node_offset_y):
                            node_offset_y.SetValue(node_offset_y.GetMin())
                        else:
                            print('Offset Y not available...')
                        # Set maximum width
                        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
                        if PySpin.IsAvailable(node_width) and PySpin.IsWritable(node_width):
                            width_to_set = node_width.GetMax()
                            node_width.SetValue(width_to_set)
                        else:
                            print('Width not available...')
                        # Set maximum height
                        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
                        if PySpin.IsAvailable(node_height) and PySpin.IsWritable(node_height):
                            height_to_set = node_height.GetMax()
                            node_height.SetValue(height_to_set)
                        else:
                            print('Height not available...')
                    

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    print(self.camID)
            
            except Empty:
                pass
        
        

        
    
