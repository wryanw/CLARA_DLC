"""
CLARA toolbox
https://github.com/wryanw/CLARA
W Williamson, wallace.williamson@ucdenver.edu

"""


from __future__ import print_function
from multiprocessing import Array, Queue
from queue import Empty
import wx
import wx.lib.dialogs
import cv2
import os
import numpy as np
from pathlib import Path
import pandas as pd
import sched, time
import ctypes
#import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from deeplabcut.CLARA_DLC_WRW import CLARA_DLC_utils as clara
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.CLARA_DLC_WRW import CLARA_DLC_PySpin_v3
import datetime
import serial

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(wx.Panel):

    def __init__(self, parent, gui_size, axesCt, **kwargs):
#        h=np.amax(gui_size)/4
#        w=np.amax(gui_size)/4
#        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER,size=(h,w))
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)

        self.figure = Figure()
        self.axes = list()
        for a in range(axesCt):
            self.axes.append(self.figure.add_subplot(1, axesCt, a+1, frameon=False))
            self.axes[a].xaxis.set_visible(False)
            self.axes[a].yaxis.set_visible(False)
            self.axes[a].set_position([a*1/axesCt+0.005,0.005,1/axesCt-0.01,1-0.01])
            
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return(self.figure,self.axes,self.canvas)

class ControlPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
        wSpace = 16
#        ctrlsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        
        ctrlsizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.figC = Figure()
        self.axC = self.figC.add_subplot(1, 1, 1)
#        self.axes = self.figure.add_subplot(1, 1, 1, frameon=False)
#        self.axes.xaxis.set_visible(False)
#        self.axes.yaxis.set_visible(False)
#        self.axC.set_position([0.005, 0.005, 0.99, 0.99])
        self.canC = FigureCanvas(self, -1, self.figC)
        ctrlsizer.Add(self.canC, 8, wx.ALL)
        
        self.com_ctrl = wx.CheckBox(self, id=wx.ID_ANY, label="Due COM")
        ctrlsizer.Add(self.com_ctrl, 1, wx.TOP | wx.LEFT | wx.RIGHT, wSpace)
        
        
        self.load_pellet = wx.Button(self, id=wx.ID_ANY, label="Load Pellet")
        ctrlsizer.Add(self.load_pellet, 1, wx.TOP, wSpace)
        
        self.release_pellet = wx.Button(self, id=wx.ID_ANY, label="Release Pellet")
        ctrlsizer.Add(self.release_pellet, 1, wx.TOP, wSpace)
        
        self.send_stim = wx.Button(self, id=wx.ID_ANY, label="Stim")
        ctrlsizer.Add(self.send_stim, 1, wx.TOP, wSpace)
        
        led_text = wx.StaticText(self, style=wx.ALIGN_RIGHT, label='LED Power:')
        ctrlsizer.Add(led_text, 1, wx.TOP, wSpace)
        
        self.light_slider = wx.Slider(self, -1, 0, 0, 100,size=(100, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        ctrlsizer.Add(self.light_slider, 2, wx.RIGHT | wx.TOP | wx.LEFT, wSpace/2)
        self.SetSizer(ctrlsizer)
        ctrlsizer.Fit(self)
        self.Layout()
        
    def getfigure(self):
        """
        Returns the figure, axes and canvas
        """
        return(self.figC,self.axC,self.canC)
        
    def getHandles(self):
        return self.com_ctrl, self.load_pellet, self.release_pellet, self.send_stim, self.light_slider
#    
#    def on_focus(self,event):
#        pass
#    
class WidgetPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)

class MainFrame(wx.Frame):
    """Contains the main GUI and button boxes"""
    def __init__(self, parent):
        
# Settting the GUI size and panels design
        displays = (wx.Display(i) for i in range(wx.Display.GetCount())) # Gets the number of displays
        screenSizes = [display.GetGeometry().GetSize() for display in displays] # Gets the size of each display
        index = 0 # For display 1.
        screenW = screenSizes[index][0]
        screenH = screenSizes[index][1]
        self.gui_size = (500,1750)
        if screenW > screenH:
            self.gui_size = (1750,500)
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = 'CLARA DLC Video Explorer',
                            size = wx.Size(self.gui_size), pos = wx.DefaultPosition, style = wx.RESIZE_BORDER|wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetStatusText("")

        self.SetSizeHints(wx.Size(self.gui_size)) #  This sets the minimum size of the GUI. It can scale now!
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyPressed)
        
###################################################################################################################################################
# Spliting the frame into top and bottom panels. Bottom panels contains the widgets. The top panel is for showing images and plotting!
        self.guiDim = 0
        if screenH > screenW:
            self.guiDim = 1
        topSplitter = wx.SplitterWindow(self)
        vSplitter = wx.SplitterWindow(topSplitter)
        self.image_panel = ImagePanel(vSplitter,self.gui_size, 3)
        self.ctrl_panel = ControlPanel(vSplitter)
        self.widget_panel = WidgetPanel(topSplitter)
        if self.guiDim == 0:
            vSplitter.SplitHorizontally(self.image_panel,self.ctrl_panel, sashPosition=self.gui_size[1]*0.75)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitVertically(vSplitter, self.widget_panel,sashPosition=self.gui_size[0]*0.8)#0.9
        else:
            vSplitter.SplitVertically(self.image_panel,self.ctrl_panel, sashPosition=self.gui_size[0]*0.5)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitHorizontally(vSplitter, self.widget_panel,sashPosition=self.gui_size[1]*0.8)#0.9
        topSplitter.SetSashGravity(0.5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.
        
        

        wSpace = 0
        wSpacer = wx.GridBagSizer(5, 5)
        
        camctrlbox = wx.StaticBox(self.widget_panel, label="Camera Control")
        bsizer = wx.StaticBoxSizer(camctrlbox, wx.HORIZONTAL)
        camsizer = wx.GridBagSizer(5, 5)
        
        bw = 76
        vpos = 0
        self.init = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Initialize", size=(bw,-1))
        camsizer.Add(self.init, pos=(vpos,0), span=(1,3), flag=wx.ALL, border=wSpace)
        self.init.Bind(wx.EVT_TOGGLEBUTTON, self.initCams)
        
        self.sync = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Sync Acquisition")
        camsizer.Add(self.sync, pos=(vpos,3), span=(1,6), flag=wx.ALL, border=wSpace)
        
        self.reset = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Reset", size=(bw, -1))
        camsizer.Add(self.reset, pos=(vpos,9), span=(1,3), flag=wx.ALL, border=wSpace)
        self.reset.Bind(wx.EVT_BUTTON, self.camReset)
        vpos+=1
        self.setROI = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Set ROI")
        camsizer.Add(self.setROI, pos=(vpos,0), span=(0,3), flag=wx.TOP | wx.BOTTOM, border=3)
        self.setROI.Enable(False)
        
        # Making radio selection for cam choice
        choices = ['Side','Front','Top']
#        choices = [l for l in choices]
        self.cambox = wx.RadioBox(self.widget_panel, majorDimension=1, style=wx.RA_SPECIFY_ROWS,
                                  choices=choices)
        camsizer.Add(self.cambox, pos=(vpos,3), span=(1,9), flag=wx.TOP, border=-10)
        self.cambox.Bind(wx.EVT_RADIOBOX, self.camChoice)
        self.cambox.SetSelection(0)
        vpos+=1
        self.crop = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Crop", size=(bw, -1))
        camsizer.Add(self.crop, pos=(vpos,0), span=(0,3), flag=wx.TOP, border=0)
        
        self.show_anno = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Show Annotations")
        camsizer.Add(self.show_anno, pos=(vpos,3), span=(0,6), flag=wx.TOP, border=5)
        self.show_anno.Bind(wx.EVT_CHECKBOX, self.showAnnotations)
        self.widget_panel.SetSizer(wSpacer)
        self.show_anno.Enable(False)
        
        self.bin = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Bin", size=(bw, -1))
        camsizer.Add(self.bin, pos=(vpos,9), span=(0,3), flag=wx.ALL, border=wSpace)
        self.bin.Bind(wx.EVT_BUTTON, self.xyBinning)
        self.bin.Enable(False)
        vpos+=1
        self.expSet = wx.SpinCtrl(self.widget_panel, value='0', size=(150, -1))#,style=wx.SP_VERTICAL)
        camsizer.Add(self.expSet, pos=(vpos+1,0), span=(1,6), flag=wx.ALL, border=wSpace)
        self.expSet.Enable(False)
        self.expSet.Bind(wx.EVT_SPINCTRL, self.updateExpRec)
        start_text = wx.StaticText(self.widget_panel, label='Exposure')
        camsizer.Add(start_text, pos=(vpos,0), span=(1,6), flag=wx.BOTTOM, border=-4)
        
        self.recSet = wx.SpinCtrl(self.widget_panel, value='1', size=(150, -1))#, min=1, max=120)
        camsizer.Add(self.recSet, pos=(vpos+1,6), span=(1,6), flag=wx.BOTTOM, border=10)
        self.recSet.Enable(False)
        self.expSet.Bind(wx.EVT_SPINCTRL, self.updateExpRec)
        end_text = wx.StaticText(self.widget_panel, label='Frame Rate')
        camsizer.Add(end_text, pos=(vpos,6), span=(1,6), flag=wx.BOTTOM, border=-4)
        vpos+=2
        self.play = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Live", size=(bw, -1))
        camsizer.Add(self.play, pos=(vpos,0), span=(1,3), flag=wx.ALL, border=wSpace)
        self.play.Bind(wx.EVT_TOGGLEBUTTON, self.liveFeed)
        self.play.Enable(False)
        
        self.rec = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Record", size=(bw, -1))
        camsizer.Add(self.rec, pos=(vpos,3), span=(1,3), flag=wx.ALL, border=wSpace)
        self.rec.Bind(wx.EVT_TOGGLEBUTTON, self.recordCam)
        self.rec.Enable(False)
        
        self.minRec = wx.TextCtrl(self.widget_panel, value='0', size=(50, -1))#,style=wx.SP_VERTICAL)
        self.minRec.Enable(False)
        min_text = wx.StaticText(self.widget_panel, label='M:')
        camsizer.Add(self.minRec, pos=(vpos,7), span=(1,2), flag=wx.ALL, border=wSpace)
        camsizer.Add(min_text, pos=(vpos,6), span=(1,1), flag=wx.TOP, border=5)
        
        self.secRec = wx.TextCtrl(self.widget_panel, value='1', size=(50, -1))#, min=1, max=120)
        self.secRec.Enable(False)
        sec_text = wx.StaticText(self.widget_panel, label='S:')
        camsizer.Add(self.secRec, pos=(vpos,10), span=(1,2), flag=wx.ALL, border=wSpace)
        camsizer.Add(sec_text, pos=(vpos,9), span=(1,1), flag=wx.TOP, border=5)
        vpos+=4
        bsizer.Add(camsizer, 1, wx.EXPAND | wx.ALL, 5)
        wSpacer.Add(bsizer, pos=(0, 0), span=(vpos,3),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border=5)
#       
        wSpace = 15
        
        self.slider = wx.Slider(self.widget_panel, -1, 0, 0, 100,size=(300, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        wSpacer.Add(self.slider, pos=(vpos,0), span=(0,3), flag=wx.LEFT, border=wSpace)
        self.slider.Enable(False)
        
        vpos+=1
        self.findP = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Find Pellet")
        wSpacer.Add(self.findP, pos=(vpos,1), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.findP.Bind(wx.EVT_BUTTON, self.findPellet)
        self.findP.Enable(False)
        
        self.rtdlc = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="RT DLC")
        wSpacer.Add(self.rtdlc, pos=(vpos,0), span=(1,1), flag=wx.LEFT, border=wSpace)
        self.rtdlc.Bind(wx.EVT_CHECKBOX, self.dlcChecked)
        self.widget_panel.SetSizer(wSpacer)
        self.rtdlc.Enable(False)
        
        vpos+=1
        self.run_expt = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Run Expt ID:")
        wSpacer.Add(self.run_expt, pos=(vpos,0), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.run_expt.Bind(wx.EVT_TOGGLEBUTTON, self.runExpt)
        self.run_expt.Enable(False)
        
        self.expt_id = wx.TextCtrl(self.widget_panel, id=wx.ID_ANY, size=(150, -1), value="0000000000000000")
        wSpacer.Add(self.expt_id, pos=(vpos,1), span=(0,2), flag=wx.LEFT, border=wSpace)
        self.expt_id.Bind(wx.EVT_TEXT, self.exptID)
        self.expt_id.Enable(False)
        
        vpos+=2
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        wSpacer.Add(self.quit, pos=(vpos,2), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)
        self.Bind(wx.EVT_CLOSE, self.quitButton)

#        wSpacer.Add(self, 1, wx.EXPAND)
        self.widget_panel.SetSizer(wSpacer)
        wSpacer.Fit(self.widget_panel)
        self.widget_panel.Layout()
        
        self.liveTimer = wx.Timer(self, wx.ID_ANY)
        self.camTimer = wx.Timer(self, wx.ID_ANY)
#        self.dlcTimer = wx.Timer(self, wx.ID_ANY)
        self.shuffle = 1
        self.trainingsetindex = 0
        self.currAxis = 0
        self.x1 = 0
        self.y1 = 0
        self.im = list()
        
        self.pellet_sel = 0
        self.hand_sel = 0
        self.act_sel = 0
        self.out_sel = 0
        self.pellet_state = ['Unknown','OnPost','OffPost']
        self.hand_state = ['Unknown','Reach','Retrieve']
        self.sys_action = ['None','DispPellet','PelletTest','ReachTest','StimTest']
        self.sys_outcome = ['NA','PelletInPlace','PelletFail','RetFail','RetPass']
        self.sys_tabs = np.zeros((50,2))
        self.sys_timer = [0,0]
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        self.figC,self.axC,self.canC = self.ctrl_panel.getfigure()
        self.com_ctrl, self.load_pellet, self.release_pellet, self.send_stim, self.light_slider = self.ctrl_panel.getHandles()
        self.com_ctrl.Bind(wx.EVT_CHECKBOX, self.comInit)
        self.load_pellet.Bind(wx.EVT_BUTTON, self.comFun)
        self.release_pellet.Bind(wx.EVT_BUTTON, self.comFun)
        self.send_stim.Bind(wx.EVT_BUTTON, self.comFun)
        self.light_slider.Bind(wx.EVT_BUTTON, self.comFun)
        self.axC.plot([0,100],[0,1])
        self.figC.canvas.draw()
        
        self.user_cfg = clara.read_config()
        if self.user_cfg['frontCam'] == None:
            wx.MessageBox('Camera serial numbers are missing', 'Info',
                          wx.OK | wx.ICON_INFORMATION)
            self.quitButton(event=None)
    
        self.camStrList = ['sideCam','frontCam','topCam']
        self.cropPts = list()
        self.array = list()
        self.big_array = list()
        for s in self.camStrList:
            cpt = self.user_cfg[s.split('Cam')[0]+'Crop']
            self.cropPts.append(cpt)
            self.array.append(Array(ctypes.c_ubyte, cpt[1]*cpt[3]))
            self.big_array.append(Array(ctypes.c_ubyte, 360*270))
        
        self.config_path=self.user_cfg['config_path']
        self.cfg = auxiliaryfunctions.read_config(self.config_path)
        
    def comInit(self, event):
        if self.com_ctrl.GetValue():
            self.ser = serial.Serial(self.user_cfg['COM'], baudrate=115200)
        else:
            self.ser.close()
        
    def comFun(self, event):
        if self.load_pellet == event.GetEventObject():
            self.ser.write(b'Q')
        elif self.release_pellet == event.GetEventObject():
            self.ser.write(b'R')
        elif self.send_stim == event.GetEventObject():
            self.ser.write(b'S')
        
    def OnSliderScroll(self, event):
        """
        Slider sets light intensity
        """
        self.ser.write('L'+str(self.light_slider.GetValue()))
        
    def OnKeyPressed(self, event):
        
#        print(event.GetKeyCode())
#        print(wx.WXK_RETURN)
        crpRef = self.cambox.GetSelection()
        if crpRef == 3:
            self.setROI.SetValue(False)
        if self.setROI.GetValue() == True:
            if event.GetKeyCode() == wx.WXK_UP:
                self.cropPts[crpRef][2]-=1
                self.croprec[crpRef].set_y(self.cropPts[crpRef][2])
            elif event.GetKeyCode() == wx.WXK_DOWN:
                self.cropPts[crpRef][2]+=1
                self.croprec[crpRef].set_y(self.cropPts[crpRef][2])
            elif event.GetKeyCode() == wx.WXK_LEFT:
                self.cropPts[crpRef][0]-=1
                self.croprec[crpRef].set_x(self.cropPts[crpRef][0])
            elif event.GetKeyCode() == wx.WXK_RIGHT:
                self.cropPts[crpRef][0]+=1
                self.croprec[crpRef].set_x(self.cropPts[crpRef][0])
            elif event.GetKeyCode() == wx.WXK_RETURN:
                self.setROI.SetValue(False)
            self.figure.canvas.draw()
        else:
            event.Skip()
    
    def updateExpRec(self,event):
        exp = self.expSet.GetValue()
        rec = self.recSet.GetValue()
        exp, rec, maxexp = self.cam.configure_exposure_and_framerate(exp, rec)
        self.expSet.SetMax(maxexp)
        self.expSet.SetValue(exp)
        self.recSet.SetValue(rec)
        
    def xyBinning(self,event):
        binlist = ['1','2','4']
        dlg = wx.SingleChoiceDialog(self, "Select bin count",'The Caption',binlist,wx.CHOICEDLG_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            binChoice = int(dlg.GetStringSelection())
        else:
            dlg.Destroy()
        self.cam.setBinning(binChoice)
    
    def showAnnotations(self,event):
        if self.show_anno.GetValue():
            for p in self.pLoc:
                p.set_alpha(self.alpha)
            if self.crop.GetValue() == False:
                for c in self.croprec:
                    c.set_alpha(self.alpha)
        else:
            for p in self.pLoc:
                p.set_alpha(0)
            for c in self.croprec:
                c.set_alpha(0)
        self.figure.canvas.draw()
        
    def camReset(self,event):
        if self.init.GetValue() == False:
            self.camq = Queue()
            self.camq_p2read = Queue()
            if self.crop.GetValue():
                self.cam = CLARA_DLC_PySpin_v3.CLARA_DLC_Cam(self.user_cfg, self.camq, self.camq_p2read, self.array, self.crop.GetValue())
            else:
                self.cam = CLARA_DLC_PySpin_v3.CLARA_DLC_Cam(self.user_cfg, self.camq, self.camq_p2read, self.big_array, self.crop.GetValue())
            self.cam.start()
        self.camq.put('Reset')
        try:
            self.camq_p2read.get(timeout = 5)
        except Empty:
            pass
        self.camq.close()
        self.camq_p2read.close()
        self.cam.terminate()
        print('\n*** CAMERAS RESET ***\n')
    
    def runExpt(self,event):
        print('todo')
    def exptID(self,event):
        print('todo')
        
    def liveFeed(self, event):
        if self.play.GetValue() == True:
#            self.cam.livePrep()
#            time.sleep(1)
            if not self.liveTimer.IsRunning():
#                s = sched.scheduler(time.time, time.sleep)
#                s.enter(5, 0, self.cam.startContAq, kwargs={'shared_array': self.array})
#                s.run()
                self.camq.put('StartAq')
                self.liveTimer.Start(100)
                print('passed')
#            if not self.camTimer.IsRunning():
#                self.camTimer.Start()
#            self.play.SetLabel('Stop')
        else:
            if self.liveTimer.IsRunning():
                self.liveTimer.Stop()
            self.camq.put('Stop')
            time.sleep(2)
            
#            if self.camTimer.IsRunning():
#                self.camTimer.Stop()
            self.play.SetLabel('Live')
        
    def vidPlayer(self, event):
        try:
            msg = self.camq_p2read.get(block=False)
#            print(msg)
            if msg == 'HasFrame':
                for ndx, im in enumerate(self.im):
                    
                    if self.crop.GetValue():
                        frame = np.frombuffer(self.array[ndx].get_obj(), self.dtype, self.size).reshape(self.shape)
                    else:
                        frame = np.frombuffer(self.big_array[ndx].get_obj(), self.dtype, self.size).reshape(self.shape)
                    im.set_data(frame)
                self.figure.canvas.draw()
                self.camq.put('GetFrame')
        except Empty:
            pass
#    def frameGrabber(self, event):
#        for ndx, s in enumerate(self.camStrList):
#            camID = str(self.user_cfg[s])
#            cam = self.cam_list.GetBySerial(camID)
#            image_result = cam.GetNextImage()
#            if image_result.IsIncomplete():
#                print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
#            else:
#                self.frame[ndx] = CLARA_DLC_PySpin.imConvert(image_result)
##                image_result.Release()
#                        #  Convert image to mono 8 and append to list
##                if self.start_time[ndx] == 0:
##                    self.start_time[ndx] = image_result.GetTimeStamp()
##                else:
##                    self.capture_duration[ndx] = image_result.GetTimeStamp()-self.start_time[ndx]
##                    # capture_duration = capture_duration/1000/1000
##                    self.frame[ndx] = CLARA_DLC_PySpin.imConvertRec(image_result)
##                    self.avi_list[ndx].Append(image_result)
##                    self.f_list[ndx].write("%s\n" % round(self.capture_duration[ndx]))
#    
        
    def camChoice(self, event):
        self.setROI.SetFocus()
    
    def initCams(self, event):
        if self.init.GetValue() == True:
            self.camq = Queue()
            self.camq_p2read = Queue()
            if self.crop.GetValue():
                self.cam = CLARA_DLC_PySpin_v3.CLARA_DLC_Cam(self.user_cfg, self.camq, self.camq_p2read, self.array, self.crop.GetValue())
            else:
                self.cam = CLARA_DLC_PySpin_v3.CLARA_DLC_Cam(self.user_cfg, self.camq, self.camq_p2read, self.big_array, self.crop.GetValue())
            self.cam.start()
            print('parent',os.getpid())
    #        self.cam.join()
            
            self.currFrame = 0
            self.bodyparts = self.cfg['bodyparts']
            # checks for unique bodyparts
            if len(self.bodyparts)!=len(set(self.bodyparts)):
                print("Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again.")
            self.colormap = plt.get_cmap(self.cfg['colormap'])
            self.colormap = self.colormap.reversed()
            self.markerSize = 6
            self.alpha = self.cfg['alphavalue']
            self.camq.put('Init')
            self.camq.put(self.sync.GetValue())
#            self.cam.initializeCameras(self.sync.GetValue())
#            self.cam.restoreXYWH()
            self.camq.put('setBinning')
            self.camq.put(self.user_cfg['bin'])
            self.expSet.SetMax(1000)
            self.recSet.SetMax(200)
            self.expSet.SetValue(self.user_cfg['exposure'])
            self.recSet.SetValue(self.user_cfg['framerate'])
#            self.updateExpRec(event)
            self.Bind(wx.EVT_TIMER, self.vidPlayer, self.liveTimer)
#            self.Bind(wx.EVT_TIMER, self.frameGrabber, self.camTimer)
            self.prepAxes()
            self.init.SetLabel('Release')
            self.rtdlc.Enable(True)
            self.findP.Enable(True)
            self.setROI.Enable(True)
            self.expSet.Enable(True)
            self.recSet.Enable(True)
            self.play.Enable(True)
            self.rec.Enable(True)
            self.minRec.Enable(True)
            self.secRec.Enable(True)
            self.run_expt.Enable(True)
            self.expt_id.Enable(True)
            self.bin.Enable(True)
            self.crop.Enable(False)
            self.show_anno.Enable(True)
            self.sync.Enable(False)
            self.reset.Enable(False)
        else:
            self.camq.put('Release')
            self.init.SetLabel('Enable')
            self.rtdlc.Enable(False)
            self.findP.Enable(False)
            self.setROI.Enable(False)
            self.expSet.Enable(False)
            self.recSet.Enable(False)
            self.play.Enable(False)
            self.rec.Enable(False)
            self.minRec.Enable(False)
            self.secRec.Enable(False)
            self.run_expt.Enable(False)
            self.expt_id.Enable(False)
            self.bin.Enable(False)
            self.crop.Enable(True)
            self.show_anno.Enable(False)
            self.sync.Enable(True)
            self.reset.Enable(True)
            self.show_anno.SetValue(False)
            if len(self.im) > 0:
                for hax in self.axes:
                    hax.clear()
                self.im = list()
            self.figure.canvas.draw()
            self.camq.close()
            self.camq_p2read.close()
            self.cam.terminate()
        
    def recordCam(self, event):
        if self.rec.GetValue():
            date_string = datetime.datetime.now().strftime("%Y%m%d")
            base_dir = os.path.join(self.user_cfg['raw_data_dir'], date_string, self.user_cfg['unitRef'])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            prev_expt_list = [name for name in os.listdir(base_dir) if name.startswith('session')]
            file_count = len(prev_expt_list)+1
            sess_string = '%s%03d' % ('session', file_count)
            sess_dir = os.path.join(base_dir, sess_string)
            if not os.path.exists(sess_dir):
                os.makedirs(sess_dir)
            date_string = datetime.datetime.now().strftime("%Y%m%d")
            self.save_path = list()
            self.avi_list = list()
            self.f_list = list()
            self.start_time = list()
            self.capture_duration = list()
            for ndx, s in enumerate(self.camStrList):
                camID = str(self.user_cfg[s])
                cam = self.cam_list.GetBySerial(camID)
                name_base = '%s_%s_%s_%s' % (date_string, self.user_cfg['unitRef'], sess_string, s)
                path_base = os.path.join(sess_dir,name_base)
                print(path_base)
                self.save_path.append(path_base)
                avi_recorder, option = CLARA_DLC_PySpin_v3.recordPrep(cam,30)
                avi_recorder.Open(path_base, option)
                self.f_list.append(open('%s_timestamps.txt' % path_base, 'w'))
                self.avi_list.append(avi_recorder)
                self.start_time.append(0)
                self.capture_duration.append(0)
        else:
            for ndx, s in enumerate(self.camStrList):
                camID = str(self.user_cfg[s])
                cam = self.cam_list.GetBySerial(camID)
                self.avi_list[ndx].Close()
                self.f_list[ndx].close()
    
    def prepAxes(self):
        if len(self.im) == 0:
            print('new axes')
            self.cropPts = list()
            self.circleH = list()
            self.circleP = list()
            self.textH = list()
            self.pLoc = list()
            self.croprec = list()
        
        self.camq.put('GetOneFrame')
        self.frame = self.camq_p2read.get(block=True)
#        self.frame = self.cam.getOneFrame()
        self.size = self.frame[0].size
        self.dtype = self.frame[0].dtype
        self.shape = self.frame[0].shape
        frmWH = np.shape(self.frame[0])
        self.width = frmWH[0]
        self.height = frmWH[1]
        for ndx, s in enumerate(self.camStrList):
            if len(self.im) == ndx:
                cpt = self.user_cfg[s.split('Cam')[0]+'Crop']
                self.cropPts.append(cpt)
                frame = self.frame[ndx]
                self.im.append(self.axes[ndx].imshow(frame,cmap='gray'))
                self.im[ndx].set_clim(0,255)
                self.points = [-10,-10,1.0]
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=0)]
                self.circleH.append(self.axes[ndx].add_patch(circle[0]))
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=0)]
                self.circleP.append(self.axes[ndx].add_patch(circle[0]))
                circle = [patches.Circle((-10, -10), radius=5, linestyle=':', fc=[1,0.75,0], alpha=0.0)]
                self.pLoc.append(self.axes[ndx].add_patch(circle[0]))
                croprec = [patches.Rectangle((cpt[0]+1,cpt[2]+1), cpt[1]-3, cpt[3]-3, fill=False, ec = [0.25,0.75,0.25], linewidth=2, linestyle='-',alpha=0.0)]
                self.croprec.append(self.axes[ndx].add_patch(croprec[0]))
                
        
        self.figure.canvas.draw()
        
    def findPellet(self, event):
        for ndx, hax in enumerate(self.axes):
            bpindexP = -1
            pellet_test = np.where(self.df_likelihood[ndx][bpindexP,:] > 0.9)
            zeroPx = np.median(self.df_x[ndx][bpindexP,pellet_test])
            zeroPy = np.median(self.df_y[ndx][bpindexP,pellet_test])
            self.PorigXY[ndx][0] = zeroPx
            self.PorigXY[ndx][1] = zeroPy
            self.pLoc[ndx].set_center([zeroPx,zeroPy])
        for p in range(0,3):
            self.pLoc[p].set_alpha(self.alpha)
        self.figure.canvas.draw()
                
    def dlcChecked(self, event):
        if self.rtdlc.GetValue():
            
            if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
                del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
            tf.reset_default_graph()
            
            modelfolder=os.path.join(self.cfg["config_path"],str(auxiliaryfunctions.GetModelFolder(self.trainFraction,self.shuffle,self.cfg)))
            path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
            try:
                self.dlc_cfg = load_config(str(path_test_config))
            except FileNotFoundError:
                raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(self.shuffle,self.trainFraction))
            # Check which snapshots are available and sort them by # iterations
            try:
              Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
            except FileNotFoundError:
              raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(self.shuffle,self.shuffle))
        
            snapshotindex = -1
            increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
            Snapshots = Snapshots[increasing_indices]
            print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)
            ##################################################
            # Load and setup CNN part detector
            ##################################################
            self.batch_num = 0 # keeps track of which batch you are at
            
            cpt = self.cropPts[0]
            nx = cpt[1]
            ny = cpt[3]
            
            self.benchmark = 0
            self.total_anal = 0
            self.batchsize=3*5
            self.dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
            self.PredicteData = np.zeros((self.numberFrames,len(self.bodyparts),3))
            self.frames = np.empty((self.batchsize, ny, nx, 3), dtype='ubyte') # this keeps all frames in a batch
            self.dlc_cfg['batch_size']=self.batchsize
            self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.dlc_cfg)
            self.playDelay = 500
            self.Bind(wx.EVT_TIMER, self.predict_frames, self.liveTimer)
                
    def predict_frames(self,event):
        camsInBatch = np.tile([0,1,2],int(self.batchsize/3))
        for batch_ind, camref in enumerate(camsInBatch):
            ret, frame = self.vid[camref].read()
            cpt = self.cropPts[camref]
            if ret:
                frame = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.frames[batch_ind] = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
        if self.total_anal > 0:
            start = time.time()
        pose = predict.getposeNP(self.frames, self.dlc_cfg, self.sess, self.inputs, self.outputs)
        if self.total_anal > 0:
            self.benchmark = self.benchmark+(time.time()-start)
            print(int((self.total_anal-4)/self.benchmark))
        self.total_anal+=int(self.batchsize/3)
        
        pose = pose.reshape(self.batchsize,len(self.bodyparts),3)
#        print(pose)
#        pdindex = pd.MultiIndex.from_product([[self.scorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])
#        DataMachine = pd.DataFrame(pose, columns=pdindex, index=range(batchsize))
#        print(DataMachine)
        self.PredicteData[self.batch_num*self.batchsize:(self.batch_num+1)*self.batchsize] = pose
        self.batch_num+=1
        
    def update(self,event):
        """
        Updates the image with the current slider index
        """
        self.pellet_sel = 0
        self.hand_sel = 0
        hand_seen = False
        for ndx, im in enumerate(self.im):
            # Draw
            if self.next != event.GetEventObject():
                self.vid[ndx].set(1,self.currFrame)
                ret, frame = self.vid[ndx].read()
                frame = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cpt = self.cropPts[ndx]
                frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                im.set_data(frame)
                
            PcurrXY = np.zeros((2 ,1))
            HcurrXY = np.zeros((2 ,1))
            if self.show_anno.GetValue():
                class_test = np.amax(self.df_likelihood[ndx][0:-3,self.currFrame])>0.8
                bpindexP = -1
                bpindexH = -2
                pellet_test = np.amax(self.df_likelihood[ndx][bpindexP,self.currFrame])>0.8
                hand_test = np.amax(self.df_likelihood[ndx][bpindexH,self.currFrame])>0.8
                bpindexC = np.argmax(self.df_likelihood[ndx][0:-3,self.currFrame])
                if hand_test or class_test:
                    if class_test:
                        self.points = [int(self.df_x[ndx][bpindexC,self.currFrame]),
                                   int(self.df_y[ndx][bpindexC,self.currFrame]),1.0]
                        self.drawCirc(self.circleH,ndx,bpindexC)
                    else:
                        self.points = [int(self.df_x[ndx][bpindexH,self.currFrame]),
                                   int(self.df_y[ndx][bpindexH,self.currFrame]),1.0]
                        self.drawCirc(self.circleH,ndx,bpindexH)
                    HcurrXY[0] = self.points[0]
                    HcurrXY[1] = self.points[1]
#                    distHP = np.linalg.norm(self.PorigXY[ndx]-HcurrXY)
                    self.tabsHX[ndx][0] = HcurrXY[0]
                    if ndx == 0:
                        hand_seen = True
                else:
                    self.circleH[ndx].set_alpha(0.0)
                    self.tabsHX[ndx][0] = np.nan
                    
                if pellet_test:
                    self.points = [int(self.df_x[ndx][bpindexP,self.currFrame]),
                                   int(self.df_y[ndx][bpindexP,self.currFrame]),1.0]
                    self.drawCirc(self.circleP,ndx,bpindexP)
                    PcurrXY[0] = self.points[0]
                    PcurrXY[1] = self.points[1]
                    distP = np.linalg.norm(self.PorigXY[ndx]-PcurrXY)
                    message= 'P/P-orig: %s\n' % f'{distP:.2f}'
                    if ndx < 2:
                        if distP < 10:
                            self.pellet_sel = 1
                        else:
                            self.pellet_sel = 2
                else:
                    self.circleP[ndx].set_alpha(0.0)
                    message= 'P/P-orig: na\n'
                    
                if pellet_test and hand_test:
                    distHP = np.linalg.norm(PcurrXY-HcurrXY)
                    if distHP > 10:
                        self.sys_outcome = 4
                    message+= 'H/P-dist: %s\n' % f'{distHP:.2f}'
                else:
                    message+= 'H/P-dist: na\n'
                    
                nanHP = np.isnan(self.tabsHX[ndx])
                numHPtot = np.sum(~nanHP)
                nanHPtot = np.sum(nanHP)
                numHP = self.tabsHX[ndx][~nanHP]
                if nanHPtot < 2:
                    hpdelta = -np.nanmean(np.diff(numHP))
                    message+= 'H/PO-delta: %s of %d\n' % (f'{hpdelta:.2f}',numHPtot)
                    if ndx < 2:
                        if hpdelta < 0:
                            self.hand_sel = 1
                        else:
                            self.hand_sel = 2
                else:
                    message+= 'H/PO-delta: na\n'
                self.tabsHX[ndx] = np.roll(self.tabsHX[ndx],1)
                self.textH[ndx].set_text(message)
            
                
#        pstr = self.pellet_state[self.pellet_sel]
#        hstr = self.hand_state[self.hand_sel]
        self.sys_tabs[1,:] = [self.pellet_sel,self.hand_sel]
        secs4raise = 1
        if self.act_sel == 0:
            if self.sys_timer[0] > 120:
                self.act_sel = 1
                self.sys_timer[0] = 0
                print('raise post')
            else:
                self.sys_timer[0]+=self.playSkip
        elif self.act_sel == 1:
            if (self.sys_timer[0] > 120*secs4raise) and not hand_seen:
                self.act_sel = 2
                self.sys_timer[0] = 0
                print('drop post')
            else:
                self.sys_timer[0]+=self.playSkip
        elif self.act_sel == 2:
            self.sys_timer[0]+=self.playSkip
            if self.pellet_sel == 1:
                if self.sys_timer[1] > 5:
                    self.act_sel = 3
                    self.sys_timer = [0,0,0]
                    print('pellet placed')
                else:
                    self.sys_timer[1]+=1
#            elif (self.sys_timer[0] > 120*10) or (self.pellet_sel == 2):
#                print('raise post')
#                self.sys_timer[0] = 0
#                self.act_sel = 1
        elif self.act_sel == 3:
            if (self.sys_outcome == 4) and (self.pellet_sel == 2):
                self.sys_timer[0]+=self.playSkip
                if self.sys_timer[0] >= 2:
                    print('fail test')
                    self.ret_found[1] = self.currFrame
                    self.act_sel = 4
                    self.sys_timer = [0,0,0]
            elif not hand_seen and (self.pellet_sel == 0):
                self.sys_timer[1]+=self.playSkip
                if self.sys_timer[1] > 5:
                    self.act_sel = 0
                    print('reach success!')
                    self.sys_timer[1] = 0
                    self.ret_found[1] = self.currFrame-5
                    self.ret_found[0] = 2
            else:
                self.sys_timer = [0,0]
        elif self.act_sel == 4:
            if self.pellet_sel == 1:
                self.sys_timer[1]+=self.playSkip
                if self.sys_timer[1] > 10:
                    self.act_sel = 3
                    self.sys_timer[0] = 0
            elif (self.sys_outcome == 4) and (self.pellet_sel == 2):
                self.sys_timer[2]+=self.playSkip
                if self.sys_timer[2] >= 2:
                    print('reach fail')
                    self.act_sel = 0
                    self.sys_timer = [0,0]
                    self.ret_found[0] = 1
                    self.ret_found[1] = self.currFrame
            elif self.sys_timer[0] > 120:
                self.act_sel = 0
                self.sys_timer[0] = 0
                self.ret_found[0] = 1
                print('reach fail!')
            else:
                self.sys_timer[1] = 0
                self.sys_timer[2] = 0
                self.sys_timer[0]+=self.playSkip
            
                
#        sys = self.sys_action[self.act_sel]
            
#        self.sys_action = ['None','DispPellet','PelletTest','ReachTest','StimTest','FailTest']
#        self.sys_outcome = ['NA','PelletPass','PelletFail','RetPass','RetFail']
        
        self.sys_tabs = np.roll(self.sys_tabs,[1,1])
        if self.next != event.GetEventObject():
            self.figure.canvas.draw()
        
    def drawCirc(self, handle, ndx, bpndx):
        color = self.colormap(self.norm(self.colorIndex[bpndx]))
        handle[ndx].set_facecolor(color)
        handle[ndx].set_center(self.points)
        handle[ndx].set_alpha(self.alpha)
            
    def chooseFrame(self):
        print('need to make')
        
    def grabFrame(self,event):
        print('need to make')
    
    def quitButton(self, event):
        """
        Quits the GUI
        """
        print('Close event called')
        if self.init.GetValue() == True:
            self.init.SetValue(False)
            self.initCams(event)
        if self.com_ctrl.GetValue():
            self.com_ctrl.SetValue(False)
            self.comInit(event)
        self.statusbar.SetStatusText("")
        self.Destroy()
    
def show():
    import imageio
    imageio.plugins.ffmpeg.download()
    app = wx.App()
    MainFrame(None).Show()
    app.MainLoop()
    
    s = sched.scheduler(time.time, time.sleep)
    s.enter(5, 0, app.MainLoop)
#    s.enter(5, 1, print_time, kwargs={'a': 'keyword'})
    s.run()

if __name__ == '__main__':
    
    show()