#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:26:20 2019

@author: bioelectrics
"""
import PySpin
from math import floor
import os
from deeplabcut.utils import auxiliaryfunctions
from multiprocessing import Process
from queue import Empty
import numpy as np
        

        
class CLARA_DLC_Cam(Process):
    def __init__(self, user_cfg, camq, camq_p2read, array, crop):
        super().__init__()
        self.user_cfg = user_cfg
        self.camq = camq
        self.camq_p2read = camq_p2read
        self.array = array
        self.crop = crop
        self.camStrList = ['sideCam','frontCam','topCam']
        self.slist = list()
        self.cropPts = list()
        for s in self.camStrList:
            cpt = self.user_cfg[s.split('Cam')[0]+'Crop']
            self.cropPts.append(cpt)
            if self.user_cfg['masterCam'] != self.user_cfg[s]:
                self.slist.append(self.user_cfg[s])
        self.config_path=self.user_cfg['config_path']
        self.cfg = auxiliaryfunctions.read_config(self.config_path)
        self.bodyparts = self.cfg['bodyparts']
        self.iterationindex = self.cfg['iteration']
        
    def run(self):
        print('child',os.getpid())
        while True:
            try:
                msg = self.camq.get(block=False)
#                print(msg)
                try:
                    if msg == 'Init':
                        sync = self.camq.get(block=True)
                            
                        # Retrieve singleton reference to system object
                        system = PySpin.System.GetInstance()
                        # Retrieve list of cameras from the system
                        cam_list = system.GetCameras()
                        num_cameras = cam_list.GetSize()
                        print('Number of cameras detected:', num_cameras)
                        # Finish if there are no cameras
                        if num_cameras == 0:
                            # Clear camera list before releasing system
                            cam_list.Clear()
                            system.ReleaseInstance() # Release system instance
                            print('Not enough cameras!')
                            continue
                        # Set up primary camera trigger
                        cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                        cam.Init()
                        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
                        cam.V3_3Enable.SetValue(sync)
                        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                        # Set up slave camera triggers
                        for s in self.slist:
                            cam = cam_list.GetBySerial(str(s))
                            cam.Init()
                            if sync:
                                cam.TriggerSource.SetValue(PySpin.TriggerSource_Line3)
                                cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
                                cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
                            else:
                                cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                                cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                                cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                    
                    elif msg == 'Release':
                        for cam in cam_list:
                            cam.DeInit()
                        for cam in cam_list:
                            del cam
                        for ndx, s in enumerate(self.camStrList):
                            camID = str(self.user_cfg[s])
                            cam = cam_list.RemoveBySerial(camID)
                        system.ReleaseInstance() # Release instance
    
                    elif msg == 'Reset':
                        system = PySpin.System.GetInstance()
                        cam_list = system.GetCameras()
                        num_cameras = cam_list.GetSize()
                        if num_cameras > 0:
                            cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                            cam.Init()
                            cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
                            cam.V3_3Enable.SetValue(False)
                            cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                            for s in self.slist:
                                cam = cam_list.GetBySerial(str(s))
                                cam.Init()
                                cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                                cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
                                cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
                            for cam in cam_list:
                                cam.BeginAcquisition()
                            for _ in range(100):
                                for cam in cam_list:
                                    image_result = cam.GetNextImage()
                                    image_result.Release()
                            for cam in cam_list:
                                cam.EndAcquisition()
                            for cam in cam_list:
                                cam.DeInit()
                            for cam in cam_list:
                                del cam
                            for ndx, s in enumerate(self.camStrList):
                                camID = str(self.user_cfg[s])
                                cam = cam_list.RemoveBySerial(camID)
                        system.ReleaseInstance() # Release instance
                        self.camq_p2read.put('ResetComplete')
                    elif msg == 'StartAq':
                        cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                        cam.BeginAcquisition()
                        for s in self.slist:
                            cam = cam_list.GetBySerial(str(s))
                            cam.BeginAcquisition()
                        getFrames = True
                        getShared = True
                        while getFrames:
                            for ndx, s in enumerate(self.camStrList):
                                camID = str(self.user_cfg[s])
                                cam = cam_list.GetBySerial(camID)
                                image_result = cam.GetNextImage()
                                if image_result.IsIncomplete():
                                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                                else:
                                    image_result = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                                    frame = image_result.GetNDArray()
                                    if self.crop:
                                        cpt = self.cropPts[ndx]
                                        frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                                    if getShared:
                                        self.array[ndx][0:] = frame.reshape(np.shape(self.array[ndx]))
                #                    image_result.Release()
                            if getShared:
                                self.camq_p2read.put('HasFrame')
                            getShared = False
                            try:
                                msg = self.camq.get(block=False)
                                if msg == 'Stop':
                                    getFrames = False
                                elif msg == 'GetFrame':
                                    getShared = True
                            except Empty:
                                pass
                        for s in self.slist:
                            cam = cam_list.GetBySerial(str(s))
                            cam.EndAcquisition()
                        cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                        cam.EndAcquisition()
                    elif msg == 'GetOneFrame':
                        cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                        cam.BeginAcquisition()
                        for s in self.slist:
                            cam = cam_list.GetBySerial(str(s))
                            cam.BeginAcquisition()
                        frame_list = list()
                        for ndx, s in enumerate(self.camStrList):
                            camID = str(self.user_cfg[s])
                            cam = cam_list.GetBySerial(camID)
                            image_result = cam.GetNextImage()
                            if image_result.IsIncomplete():
                                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                            else:
                                image_result = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
                                frame = image_result.GetNDArray()
                                if self.crop:
                                    cpt = self.cropPts[ndx]
                                    frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                                frame_list.append(frame)
                #                image_result.Release()
                        for s in self.slist:
                            cam = cam_list.GetBySerial(str(s))
                            cam.EndAcquisition()
                        cam = cam_list.GetBySerial(str(self.user_cfg['masterCam']))
                        cam.EndAcquisition()
                        self.camq_p2read.put(frame_list)
                    elif msg == 'setBinning':
                        binsize = self.camq.get(block=True)
                        for cam in cam_list:
                            cam.BinningHorizontal.SetValue(binsize)
                            cam.BinningVertical.SetValue(binsize)
                            
                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
            
            except Empty:
                pass
                
        
        
    
    
    def recordPrep(self, cam, write_frame_rate):
        try:
            s_node_map = cam.GetTLStreamNodeMap()
            handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
            if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
                print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
                return
            handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
            handling_mode.SetIntValue(handling_mode_entry.GetValue())
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            
        avi_recorder = PySpin.SpinVideo()
        option = PySpin.MJPGOption()
        option.frameRate = write_frame_rate
        option.quality = 95
        
        return avi_recorder, option
        
    def imConvert(self, cam):
        image_result = cam.GetNextImage()
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
        else:
            image_result = image_result.Convert(PySpin.PixelFormat_Mono8, PySpin.HQ_LINEAR)
            frame = image_result.GetNDArray()
            image_result.Release()
        return frame
        
    def livePrep(self):
        for ndx, s in enumerate(self.camStrList):
            camID = str(self.user_cfg[s])
            cam = cam_list.GetBySerial(camID)
            try:
                s_node_map = cam.GetTLStreamNodeMap()
                handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
                if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
                    print('Unable to set Buffer Handling mode (node retrieval). Aborting...\n')
                    return
                handling_mode_entry = handling_mode.GetEntryByName('NewestFirst')
                handling_mode.SetIntValue(handling_mode_entry.GetValue())
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
        
    def configure_exposure_and_framerate(self, exposure_time_request, record_frame_rate):
        """
         This function configures a custom exposure time. Automatic exposure is turned
         off in order to allow for the customization, and then the custom setting is
         applied.
    
         :param cam: Camera to configure exposure for.
         :type cam: CameraPtr
         :return: True if successful, False otherwise.
         :rtype: bool
        """
                
        for ndx in self.camStrList:
            camID = str(self.user_cfg[ndx])
            cam = cam_list.GetBySerial(camID)
            try:
                if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                    print('Unable to disable automatic exposure. Aborting...')
                    return False
        
                cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                
                if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                    print('Unable to set exposure time. Aborting...')
                    return
        
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
                
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                
        max_exposure = cam_list[0].ExposureTime.GetMax()
        return exposure_time_to_set, record_frame_rate, max_exposure
    
    def restoreXYWH(self):
        for cam in cam_list:
            nodemap = cam.GetNodeMap()
            try:
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
                return
        
    