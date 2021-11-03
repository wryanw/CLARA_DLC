"""
CLARA toolbox
https://github.com/wryanw/CLARA
W Williamson, wallace.williamson@ucdenver.edu

"""


from __future__ import print_function
from multiprocessing import Array, Queue, Value
from queue import Empty
import wx
import wx.lib.dialogs
import os
import numpy as np
import time, datetime
import ctypes
from deeplabcut.utils import auxiliaryfunctions
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from deeplabcut.CLARA_DLC import CLARA_DLC_PySpin_v7 as spin
from deeplabcut.CLARA_DLC import CLARA_DLC_utils_v2 as clara
from deeplabcut.CLARA_DLC import CLARA_RT_DLC_v5 as rtdlc
from deeplabcut.CLARA_DLC import CLARA_MINISCOPE as mscam
from deeplabcut.CLARA_DLC import compressVideos_v3 as compressVideos
import serial
import shutil

# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(wx.Panel):

    def __init__(self, parent, gui_size, axesCt, **kwargs):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
            
        self.figure = Figure()
        self.axes = list()
        for a in range(axesCt):
            if gui_size[0] > gui_size[1]:
                self.axes.append(self.figure.add_subplot(1, axesCt, a+1, frameon=False))
                self.axes[a].set_position([a*1/axesCt+0.005,0.005,1/axesCt-0.01,1-0.01])
            else:
                self.axes.append(self.figure.add_subplot(axesCt, 1, a+1, frameon=False))
                self.axes[a].set_position([0.005,a*1/axesCt+0.005,1-0.01,1/axesCt-0.01])
            
            self.axes[a].xaxis.set_visible(False)
            self.axes[a].yaxis.set_visible(False)
            
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
    def __init__(self, parent, gui_size):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
        
        wSpace = 16
        if gui_size[0] > gui_size[1]:
            ctrlsizer = wx.BoxSizer(wx.HORIZONTAL)
        else:
            ctrlsizer = wx.BoxSizer(wx.VERTICAL)
        self.figC = Figure()
        self.axC = self.figC.add_subplot(1, 1, 1)
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
        
        self.bipolar = wx.CheckBox(self, id=wx.ID_ANY, label="Bipolar")
        ctrlsizer.Add(self.bipolar, 1, wx.TOP | wx.LEFT | wx.RIGHT, wSpace)
        
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
        return self.bipolar,self.com_ctrl,self.load_pellet,self.release_pellet,self.send_stim,self.light_slider
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
        self.gui_size = (800,1750)
        if screenW > screenH:
            self.gui_size = (1750,650)
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
        self.ctrl_panel = ControlPanel(vSplitter, self.gui_size)
        self.widget_panel = WidgetPanel(topSplitter)
        if self.guiDim == 0:
            vSplitter.SplitHorizontally(self.image_panel,self.ctrl_panel, sashPosition=self.gui_size[1]*0.75)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitVertically(vSplitter, self.widget_panel,sashPosition=self.gui_size[0]*0.8)#0.9
        else:
            vSplitter.SplitVertically(self.image_panel,self.ctrl_panel, sashPosition=self.gui_size[0]*0.75)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitHorizontally(vSplitter, self.widget_panel,sashPosition=self.gui_size[1]*0.7)#0.9
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
        
        self.reset = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Reset", size=(bw, -1))
        camsizer.Add(self.reset, pos=(vpos,3), span=(1,3), flag=wx.ALL, border=wSpace)
        self.reset.Bind(wx.EVT_BUTTON, self.camReset)
        
        self.save_user = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Save Settings", size=(bw*2, -1))
        camsizer.Add(self.save_user, pos=(vpos,6), span=(1,6), flag=wx.ALL, border=wSpace)
        self.save_user.Bind(wx.EVT_BUTTON, self.saveUser)
        self.save_user.Enable(False)
        
        vpos+=1
        self.setROI = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Set ROI")
        camsizer.Add(self.setROI, pos=(vpos,0), span=(0,3), flag=wx.TOP | wx.BOTTOM, border=3)
        self.setROI.Enable(False)
        
        text = wx.StaticText(self.widget_panel, label='User:')
        camsizer.Add(text, pos=(vpos,3), span=(1,3), flag=wx.ALL, border=wSpace)
        userlist = ['Anon','Spencer','Rongchen','Dailey','Xiaoyu','WRW','Jordan']
        self.users = wx.Choice(self.widget_panel, size=(100, -1), id=wx.ID_ANY, choices=userlist)
        camsizer.Add(self.users, pos=(vpos,6), span=(1,6), flag=wx.ALL, border=wSpace)
        

#        # Making radio selection for cam choice
#        choices = ['Side','Front','Top']
#        self.cambox = wx.RadioBox(self.widget_panel, majorDimension=1, style=wx.RA_SPECIFY_ROWS,
#                                  choices=choices)
#        camsizer.Add(self.cambox, pos=(vpos,3), span=(1,9), flag=wx.TOP, border=-10)
#        self.cambox.Bind(wx.EVT_RADIOBOX, self.camChoice)
#        self.cambox.SetSelection(0)
        vpos+=1
        self.crop = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Crop", size=(bw, -1))
        camsizer.Add(self.crop, pos=(vpos,0), span=(0,3), flag=wx.TOP, border=0)
        
        self.mini_scope = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Miniscope")
        self.mini_scope.Bind(wx.EVT_CHECKBOX, self.miniScope)
        camsizer.Add(self.mini_scope, pos=(vpos,3), span=(0,6), flag=wx.TOP, border=5)
        self.widget_panel.SetSizer(wSpacer)
        self.mini_scope.Enable(False)
        
        self.bin = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Bin", size=(bw, -1))
        camsizer.Add(self.bin, pos=(vpos,9), span=(0,3), flag=wx.ALL, border=wSpace)
        self.bin.Bind(wx.EVT_BUTTON, self.xyBinning)
        self.bin.Enable(False)
        vpos+=1
        self.expSet = wx.SpinCtrl(self.widget_panel, value='0', size=(150, -1))
        camsizer.Add(self.expSet, pos=(vpos+1,0), span=(1,6), flag=wx.ALL, border=wSpace)
        self.expSet.Enable(False)
        start_text = wx.StaticText(self.widget_panel, label='Exposure')
        camsizer.Add(start_text, pos=(vpos,0), span=(1,6), flag=wx.BOTTOM, border=-4)
        
        self.recSet = wx.SpinCtrl(self.widget_panel, value='1', size=(150, -1))
        camsizer.Add(self.recSet, pos=(vpos+1,6), span=(1,6), flag=wx.BOTTOM, border=10)
        self.recSet.Enable(False)
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
        
        self.minRec = wx.TextCtrl(self.widget_panel, value='20', size=(50, -1))
        self.minRec.Enable(False)
        min_text = wx.StaticText(self.widget_panel, label='M:')
        camsizer.Add(self.minRec, pos=(vpos,7), span=(1,2), flag=wx.ALL, border=wSpace)
        camsizer.Add(min_text, pos=(vpos,6), span=(1,1), flag=wx.TOP, border=5)
        
        self.secRec = wx.TextCtrl(self.widget_panel, value='0', size=(50, -1))
        self.secRec.Enable(False)
        sec_text = wx.StaticText(self.widget_panel, label='S:')
        camsizer.Add(self.secRec, pos=(vpos,10), span=(1,2), flag=wx.ALL, border=wSpace)
        camsizer.Add(sec_text, pos=(vpos,9), span=(1,1), flag=wx.TOP, border=5)
        vpos+=4
        bsizer.Add(camsizer, 1, wx.EXPAND | wx.ALL, 5)
        wSpacer.Add(bsizer, pos=(0, 0), span=(vpos,3),flag=wx.EXPAND|wx.TOP|wx.LEFT|wx.RIGHT, border=5)
#       
        wSpace = 10
        
        self.slider = wx.Slider(self.widget_panel, -1, 0, 0, 100,size=(300, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        wSpacer.Add(self.slider, pos=(vpos,0), span=(0,3), flag=wx.LEFT, border=wSpace)
        self.slider.Enable(False)
        
        vpos+=1
        self.rtdlc = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="RT DLC")
        wSpacer.Add(self.rtdlc, pos=(vpos,0), span=(1,1), flag=wx.LEFT, border=wSpace)
        self.rtdlc.Bind(wx.EVT_CHECKBOX, self.dlcChecked)
        self.widget_panel.SetSizer(wSpacer)
        self.rtdlc.Enable(False)
        
        self.pause_dlc = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Pause DLC")
        wSpacer.Add(self.pause_dlc, pos=(vpos,1), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.pause_dlc.Bind(wx.EVT_TOGGLEBUTTON, self.pauseDLC)
        self.pause_dlc.Enable(False)
        
        self.compress_vid = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Compress Vid")
        wSpacer.Add(self.compress_vid, pos=(vpos,2), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.compress_vid.Bind(wx.EVT_BUTTON, self.compressVid)
        
        vpos+=1
        self.run_expt = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Run Expt ID:")
        wSpacer.Add(self.run_expt, pos=(vpos,0), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.run_expt.Bind(wx.EVT_TOGGLEBUTTON, self.runExpt)
        self.run_expt.Enable(False)
        
        self.expt_id = wx.TextCtrl(self.widget_panel, id=wx.ID_ANY, size=(150, -1), value="Mouse Ref")
        wSpacer.Add(self.expt_id, pos=(vpos,1), span=(0,2), flag=wx.LEFT, border=wSpace)
        self.expt_id.Bind(wx.EVT_TEXT, self.exptID)
        self.expt_id.Enable(False)
        
        vpos+=1
        start_text = wx.StaticText(self.widget_panel, label='Automate:')
        wSpacer.Add(start_text, pos=(vpos,0), span=(0,1), flag=wx.LEFT, border=wSpace)
        
        self.auto_pellet = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Pellet")
        wSpacer.Add(self.auto_pellet, pos=(vpos,1), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.auto_pellet.SetValue(1)
        self.auto_stim = wx.CheckBox(self.widget_panel, id=wx.ID_ANY, label="Stimulus")
        wSpacer.Add(self.auto_stim, pos=(vpos,2), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.auto_stim.SetValue(1)
        
        vpos+=1
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        wSpacer.Add(self.quit, pos=(vpos,0), span=(0,1), flag=wx.LEFT, border=wSpace)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)
        self.Bind(wx.EVT_CLOSE, self.quitButton)

        self.widget_panel.SetSizer(wSpacer)
        wSpacer.Fit(self.widget_panel)
        self.widget_panel.Layout()
        
        self.liveTimer = wx.Timer(self, wx.ID_ANY)
        self.recTimer = wx.Timer(self, wx.ID_ANY)
        self.shuffle = 1
        self.trainingsetindex = 0
        self.currAxis = 0
        self.x1 = 0
        self.y1 = 0
        self.im = list()
        
        
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        self.figC,self.axC,self.canC = self.ctrl_panel.getfigure()
        self.bipolar,self.com_ctrl,self.load_pellet,self.release_pellet,self.send_stim,self.light_slider = self.ctrl_panel.getHandles()
        self.com_ctrl.Bind(wx.EVT_CHECKBOX, self.comInit)
        self.bipolar.Bind(wx.EVT_CHECKBOX, self.biVmono)
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

        if self.user_cfg['unitRef'] == 'unit05':
            self.users.SetSelection(4)
        elif self.user_cfg['unitRef'] == 'unit00':
            self.users.SetSelection(3)
        else:
            self.users.SetSelection(0)
            
        self.dlc_frmct=5
        self.camStrList = ['sideCam','frontCam','topCam']
        self.dlc = Value(ctypes.c_byte, 0)
        self.camaq = Value(ctypes.c_byte, 0)
        self.frmaq = Value(ctypes.c_int, 0)
        self.pX = list()
        self.pY = list()
        self.cropPts = list()
        self.array = list()
        self.slist = list()
        for s in self.camStrList:
            cpt = self.user_cfg[s.split('Cam')[0]+'Crop']
            self.cropPts.append(cpt)
            self.pX.append(Value(ctypes.c_int, 0))
            self.pY.append(Value(ctypes.c_int, 0))
            dlcA = list()
            dlcB = list()
            dlcC = list()
            for _ in range(self.dlc_frmct):
                dlcA.append(Array(ctypes.c_ubyte, cpt[1]*cpt[3]))
            for _ in range(self.dlc_frmct):
                dlcB.append(Array(ctypes.c_ubyte, cpt[1]*cpt[3]))
            for _ in range(self.dlc_frmct):
                dlcC.append(Array(ctypes.c_ubyte, cpt[1]*cpt[3]))
            self.array.append([dlcA, dlcB, dlcC])
            if self.user_cfg['masterCam'] != self.user_cfg[s]:
                self.slist.append(str(self.user_cfg[s]))
        self.config_path=self.user_cfg['config_path']
        self.cfg = auxiliaryfunctions.read_config(self.config_path)
        self.compressThread = compressVideos.CLARA_compress()
        self.compressThread.start()
    def biVmono(self, event):
        if self.rtdlc.GetValue():
            if self.bipolar.GetValue():
                self.dlcq.put('B')
            else:
                self.dlcq.put('M')
                
    def comInit(self, event):
        if self.com_ctrl.GetValue():
            if self.rtdlc.GetValue():
                self.dlcq.put('initSerial')
            else:
                self.ser = serial.Serial(self.user_cfg['COM'], baudrate=115200)
        else:
            if self.rtdlc.GetValue():
                self.dlcq.put('stopSerial')
            else:
                self.ser.close()
        
    def comFun(self, event):
        if self.load_pellet == event.GetEventObject():
            if self.rtdlc.GetValue():
                self.dlcq.put('Q')
            else:
                self.ser.write(b'Q')
        elif self.release_pellet == event.GetEventObject():
            if self.rtdlc.GetValue():
                self.dlcq.put('R')
            else:
                self.ser.write(b'R')
        elif self.send_stim == event.GetEventObject():
            if self.rtdlc.GetValue():
                self.dlcq.put('S')
            elif self.bipolar.GetValue():
                self.ser.write(b'T')
            else:
                self.ser.write(b'S')
            
    def OnSliderScroll(self, event):
        """
        Slider sets light intensity
        """
        self.ser.write('L'+str(self.light_slider.GetValue()))
        
    def OnKeyPressed(self, event):
        
#        print(event.GetKeyCode())
#        print(wx.WXK_RETURN)
        key = event.GetKeyCode()
        if self.setROI.GetValue() == True:
            crpRef = self.cambox.GetSelection()
            if key == wx.WXK_UP:
                self.cropPts[crpRef][2]-=1
                self.croprec[crpRef].set_y(self.cropPts[crpRef][2])
            elif key == wx.WXK_DOWN:
                self.cropPts[crpRef][2]+=1
                self.croprec[crpRef].set_y(self.cropPts[crpRef][2])
            elif key == wx.WXK_LEFT:
                self.cropPts[crpRef][0]-=1
                self.croprec[crpRef].set_x(self.cropPts[crpRef][0])
            elif key == wx.WXK_RIGHT:
                self.cropPts[crpRef][0]+=1
                self.croprec[crpRef].set_x(self.cropPts[crpRef][0])
            self.figure.canvas.draw()
        elif (key == wx.WXK_RETURN) or (key == wx.WXK_NUMPAD_ENTER):
            fc = self.FindFocus()
            if fc == self.expSet or fc == self.recSet:
                self.updateExpRec(event)
        else:
            event.Skip()
            
    def compressVid(self, event):
        if self.compressThread.is_alive():
            dlg = wx.MessageDialog(parent=None,message="Pausing until previous compression completes!",
                                   caption="Warning!", style=wx.OK|wx.ICON_EXCLAMATION)
            dlg.ShowModal()
            dlg.Destroy()
            while self.compressThread.is_alive():
                time.sleep(10)
            
            
        self.compressThread.terminate()   
        self.compressThread = compressVideos.CLARA_compress()
        self.compressThread.start()

    def updateExpRec(self,event):
        exp = self.expSet.GetValue()
        rec = self.recSet.GetValue()
        for ndx, s in enumerate(self.camStrList):
            camID = str(self.user_cfg[s])
            self.camq[camID].put('exp_frmrate')
            self.camq[camID].put(exp)
            self.camq[camID].put(rec)
            try:
                exp = self.camq_p2read[camID].get(timeout = 5)
                rec = self.camq_p2read[camID].get(timeout = 2)
                maxexp = self.camq_p2read[camID].get(timeout = 2)
            except Empty:
                print('Failed to set exposure/framerate')
                return
        self.expSet.SetMax(maxexp)
        self.expSet.SetValue(round(exp/10)*10)
        self.recSet.SetValue(round(rec/10)*10)
        print('Exposure and framerate set')
        
    def xyBinning(self,event):
        binlist = ['1','2','4']
        dlg = wx.SingleChoiceDialog(self, "Select bin count",'The Caption',binlist,wx.CHOICEDLG_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            binChoice = int(dlg.GetStringSelection())
        else:
            dlg.Destroy()
        for ndx, s in enumerate(self.camStrList):
            camID = str(self.user_cfg[s])
            self.camq[camID].put('setBinning')
            self.camq[camID].put(binChoice)
        
    def camReset(self,event):
        self.initThreads()
        
        self.expSet.SetMax(1000)
        self.recSet.SetMax(200)
        self.expSet.SetValue(self.user_cfg['exposure'])
        self.recSet.SetValue(self.user_cfg['framerate'])
        self.updateExpRec(event)
        self.startAq()
        time.sleep(3)
        self.stopAq()
        self.deinitThreads()
        print('\n*** CAMERAS RESET ***\n')
    
    def runExpt(self,event):
        print('todo')
    def exptID(self,event):
        pass
        
    def liveFeed(self, event):
        if self.play.GetValue() == True:
            if not self.liveTimer.IsRunning():
                self.startAq()
                self.liveTimer.Start(150)
                self.play.SetLabel('Stop')
            
            self.rtdlc.Enable(False)
            self.setROI.Enable(False)
            self.expSet.Enable(False)
            self.recSet.Enable(False)
            self.rec.Enable(False)
            self.minRec.Enable(False)
            self.secRec.Enable(False)
            self.bin.Enable(False)
            self.save_user.Enable(False)
            self.run_expt.Enable(False)
            self.expt_id.Enable(False)
            self.pause_dlc.SetValue(False)
        else:
            if self.liveTimer.IsRunning():
                self.liveTimer.Stop()
            self.stopAq()
            time.sleep(2)
            self.play.SetLabel('Live')
            
            self.rtdlc.Enable(True)
            self.setROI.Enable(True)
            self.expSet.Enable(True)
            self.recSet.Enable(True)
            self.rec.Enable(True)
            self.minRec.Enable(True)
            self.secRec.Enable(True)
            self.bin.Enable(True)
            self.save_user.Enable(True)
            self.run_expt.Enable(True)
            self.expt_id.Enable(True)
            
        
    def vidPlayer(self, event):
        for ndx, im in enumerate(self.im):
            if self.dlc.value == 0:
                obj2get = 2
            elif self.dlc.value == 1:
                obj2get = 0
            else:
                obj2get = 1
            self.frame[ndx][:,:] = np.frombuffer(self.array[ndx][obj2get][0].get_obj(), self.dtype, self.size).reshape(self.shape)
            im.set_data(self.frame[ndx])
            
            pXY = [self.pX[ndx].value,self.pY[ndx].value]
            if pXY[0] == 0:
                self.pLoc[ndx].set_alpha(0.0)
            elif pXY[0] == -1:
                self.pause_dlc.SetValue(True)
                self.pauseDLC(event)
            else:
                self.pLoc[ndx].set_center(pXY)
                self.pLoc[ndx].set_alpha(self.alpha)
        self.figure.canvas.draw()
        
    def autoCapture(self, event):
        self.sliderTabs+=self.sliderRate
        msg = '-'
        if self.mini_scope.GetValue():
            msg = self.ms2p.get(block=False)
        if (self.sliderTabs > self.slider.GetMax()) and not (msg == 'fail'):
            self.rec.SetValue(False)
            self.recordCam(event)
            self.slider.SetValue(0)
        else:
            self.slider.SetValue(round(self.sliderTabs))
            self.vidPlayer(event)
        
    def recordCam(self, event):
        if self.rec.GetValue():
            
            liveRate = 250
            self.Bind(wx.EVT_TIMER, self.autoCapture, self.recTimer)
            if int(self.minRec.GetValue()) == 0 and int(self.secRec.GetValue()) == 0:
                return
            totTime = int(self.secRec.GetValue())+int(self.minRec.GetValue())*60
            spaceneeded = 260*370*3*3*self.recSet.GetValue()*totTime+1024^3
                
            self.slider.SetMax(100)
            self.slider.SetMin(0)
            self.slider.SetValue(0)
            self.sliderTabs = 0
            self.sliderRate = 100/(totTime/(liveRate/1000))
            
            date_string = datetime.datetime.now().strftime("%Y%m%d")
            base_dir = os.path.join(self.user_cfg['raw_data_dir'], date_string, self.user_cfg['unitRef'])
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            freespace = shutil.disk_usage(base_dir)[2]
            if spaceneeded > freespace:
                dlg = wx.MessageDialog(parent=None,message="There is not enough disk space for the requested duration.",
                                       caption="Warning!", style=wx.OK|wx.ICON_EXCLAMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.rec.SetValue(False)
                return
            prev_expt_list = [name for name in os.listdir(base_dir) if name.startswith('session')]
            file_count = len(prev_expt_list)+1
            sess_string = '%s%03d' % ('session', file_count)
            self.sess_dir = os.path.join(base_dir, sess_string)
            if not os.path.exists(self.sess_dir):
                os.makedirs(self.sess_dir)
            clara.read_metadata
            meta,ruamelFile = clara.metadata_template()
            meta['frontCrop']=self.user_cfg['frontCrop']
            meta['sideCrop']=self.user_cfg['sideCrop']
            meta['topCrop']=self.user_cfg['topCrop']
            meta['exposure']=self.expSet.GetValue()
            meta['framerate']=self.recSet.GetValue()
            meta['bin']=4
            meta['unitRef']=self.user_cfg['unitRef']
            meta['frontCam']=self.user_cfg['frontCam']
            meta['sideCam']=self.user_cfg['sideCam']
            meta['topCam']=self.user_cfg['topCam']
            meta['masterCam']=self.user_cfg['masterCam']
            meta['framecount']=totTime*self.recSet.GetValue()
            meta['config_path']=self.user_cfg['config_path']
            meta['trainingsetindex']=self.user_cfg['trainingsetindex']
            meta['shuffle']=self.user_cfg['shuffle']
            meta['ID']=self.expt_id.GetValue()
            meta['placeholderA']='info'
            meta['placeholderB']='info'
            meta['Designer']=self.users.GetStringSelection()
            meta['Stim']='none'
            meta['StartTime']=datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            meta['Collection']='info'
            meta_name = '%s_%s_%s_metadata.yaml' % (date_string, self.user_cfg['unitRef'], sess_string)
            metapath = os.path.join(self.sess_dir,meta_name)
            clara.write_metadata(meta, metapath)
            for ndx, s in enumerate(self.camStrList):
                camID = str(self.user_cfg[s])
                self.camq[camID].put('recordPrep')
                name_base = '%s_%s_%s_%s' % (date_string, self.user_cfg['unitRef'], sess_string, s)
                path_base = os.path.join(self.sess_dir,name_base)
                self.camq[camID].put(path_base)
                self.camq_p2read[camID].get()
                
            if self.rtdlc.GetValue():
                self.dlcq.put('recordPrep')
                event_base = '%s_%s_%s' % (date_string, self.user_cfg['unitRef'], sess_string)
                self.dlcq.put(os.path.join(self.sess_dir,event_base))
                self.dlc2p.get()
                
            if self.mini_scope.GetValue():
                self.msq.put('recordPrep')
                event_base = '%s_%s_%s' % (date_string, self.user_cfg['unitRef'], sess_string)
                self.msq.put(os.path.join(self.sess_dir,event_base))
                self.ms2p.get()
                
            self.rtdlc.Enable(False)
            self.setROI.Enable(False)
            self.expSet.Enable(False)
            self.recSet.Enable(False)
            self.play.Enable(False)
            self.minRec.Enable(False)
            self.secRec.Enable(False)
            self.bin.Enable(False)
            self.save_user.Enable(False)
            self.run_expt.Enable(False)
            self.expt_id.Enable(False)
            
            if not self.recTimer.IsRunning():
                self.startAq()
                self.recTimer.Start(250)
            self.rec.SetLabel('Stop')
        else:
            if self.recTimer.IsRunning():
                self.recTimer.Stop()
            self.stopAq()
            if self.mini_scope.GetValue():
                self.msq.put('Stop')
            time.sleep(2)
            self.rec.SetLabel('Record')
            self.compressVid(event=None)
            self.play.Enable(True)
            self.rtdlc.Enable(True)
            self.setROI.Enable(True)
            self.expSet.Enable(True)
            self.recSet.Enable(True)
            self.minRec.Enable(True)
            self.secRec.Enable(True)
            self.bin.Enable(True)
            self.save_user.Enable(True)
            self.run_expt.Enable(True)
            self.expt_id.Enable(True)
            self.expt_id.SetValue('Mouse Ref')
            

    def camChoice(self, event):
        self.setROI.SetFocus()
        
    def initThreads(self):
        self.camq = dict()
        self.camq_p2read = dict()
        self.cam = list()
        idList = [self.user_cfg[c] for c in self.camStrList]
        for ndx, s in enumerate(self.camStrList):
            camID = str(self.user_cfg[s])
            self.camq[camID] = Queue()
            self.camq_p2read[camID] = Queue()
            self.cam.append(spin.CLARA_DLC_Cam(self.camq[camID], self.camq_p2read[camID],
                                               self.array[ndx], self.dlc, camID, idList,
                                               self.cropPts[ndx], self.dlc_frmct, self.camaq,
                                               self.frmaq))
            self.cam[ndx].start()
            
        self.masterID = str(self.user_cfg['masterCam'])
        self.camq[self.masterID].put('InitM')
        self.camq_p2read[self.masterID].get()
        for s in self.slist:
            self.camq[s].put('InitS')
            self.camq_p2read[s].get()
            
    def deinitThreads(self):
        for ndx, s in enumerate(self.camStrList):
            camID = str(self.user_cfg[s])
            self.camq[camID].put('Release')
            self.camq_p2read[camID].get()
            self.camq[camID].close()
            self.camq_p2read[camID].close()
            self.cam[ndx].terminate()
            
    def startAq(self):
        self.camaq.value = 1
        if self.rec.GetValue():
            if self.mini_scope.GetValue():
                self.msq.put('Start')
        if self.rtdlc.GetValue():
            self.dlcq.put('Start')
        self.camq[self.masterID].put('Start')
        for s in self.slist:
            self.camq[s].put('Start')
        self.camq[self.masterID].put('TrigOff')
        
    def stopAq(self):
        self.camaq.value = 0
        for s in self.slist:
            self.camq[s].put('Stop')
            self.camq_p2read[s].get()
        self.camq[self.masterID].put('Stop')
        self.camq_p2read[self.masterID].get()
        
    def initCams(self, event):
        if self.init.GetValue() == True:
            self.Enable(False)
            self.initThreads()
            self.colormap = plt.get_cmap(self.cfg['colormap'])
            self.colormap = self.colormap.reversed()
            self.markerSize = 6
            self.alpha = self.cfg['alphavalue']
            
            for ndx, s in enumerate(self.camStrList):
                camID = str(self.user_cfg[s])
                self.camq[camID].put('restoreXYWH')
                self.camq[camID].put('setBinning')
                self.camq[camID].put(self.user_cfg['bin'])
            self.expSet.SetMax(1000)
            self.recSet.SetMax(200)
            self.expSet.SetValue(self.user_cfg['exposure'])
            self.recSet.SetValue(self.user_cfg['framerate'])
            self.updateExpRec(event)
            self.Bind(wx.EVT_TIMER, self.vidPlayer, self.liveTimer)
            self.prepAxes()
            self.init.SetLabel('Release')
            self.rtdlc.Enable(True)
            self.setROI.Enable(True)
            self.expSet.Enable(True)
            self.recSet.Enable(True)
            self.play.Enable(True)
            self.rec.Enable(True)
            self.minRec.Enable(True)
            self.secRec.Enable(True)
            self.bin.Enable(True)
            self.crop.Enable(False)
            if self.user_cfg['unitRef'] == 'unit05':
                self.mini_scope.Enable(True)
                self.mini_scope.SetValue(True)
                self.miniScope(event)
            self.reset.Enable(False)
            self.save_user.Enable(True)
            self.Enable(True)
        else:
            self.init.SetLabel('Enable')
            self.rtdlc.Enable(False)
            self.setROI.Enable(False)
            self.expSet.Enable(False)
            self.recSet.Enable(False)
            self.play.Enable(False)
            self.rec.Enable(False)
            self.minRec.Enable(False)
            self.secRec.Enable(False)
            self.bin.Enable(False)
            self.crop.Enable(True)
            self.mini_scope.Enable(False)
            self.reset.Enable(True)
            self.mini_scope.SetValue(False)
            self.save_user.Enable(False)
            if len(self.im) > 0:
                for hax in self.axes:
                    hax.clear()
                self.im = list()
            self.figure.canvas.draw()
            self.deinitThreads()
        
    def prepAxes(self):
        if len(self.im) == 0:
            print('new axes')
            self.cropPts = list()
            self.circleH = list()
            self.circleP = list()
            self.textH = list()
            self.pLoc = list()
            self.croprec = list()
        self.frame = list()
        self.dtype = 'uint8'
        cpt = self.user_cfg[self.camStrList[0].split('Cam')[0]+'Crop']
        self.size = cpt[1]*cpt[3]
        self.shape = [cpt[1], cpt[3]]
        self.startAq()
        time.sleep(1)
        frame = np.zeros(self.shape, dtype='ubyte')
        for arr in range(3):
            frame[:,:] = np.frombuffer(self.array[arr][0][0].get_obj(), self.dtype, self.size).reshape(self.shape)
            self.frame.append(frame)
        self.stopAq()
        self.width = cpt[1]
        self.height = cpt[3]
        for ndx, s in enumerate(self.camStrList):
            if len(self.im) == ndx:
                cpt = self.user_cfg[s.split('Cam')[0]+'Crop']
                self.cropPts.append(cpt)
                self.im.append(self.axes[ndx].imshow(self.frame[ndx],cmap='gray'))
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
                
        obj2get = 1
        for ndx, im in enumerate(self.im):
            self.frame[ndx][:,:] = np.frombuffer(self.array[ndx][obj2get][0].get_obj(), self.dtype, self.size).reshape(self.shape)
            im.set_data(self.frame[ndx])
        self.figure.canvas.draw()
                
    def dlcChecked(self, event):
        if self.rtdlc.GetValue():
            self.Enable(False)
            
            self.dlcq = Queue()
            self.dlc2p = Queue()
            autopellet = self.auto_pellet.GetValue()
            autostim = self.auto_stim.GetValue()
            self.rtThread = rtdlc.CLARA_RT(self.dlcq, self.dlc2p, self.array,
                                           self.dlc, self.camaq,
                                           autopellet, autostim, self.pX,
                                           self.pY, self.frmaq, self.recSet.GetValue(),
                                           self.bipolar.GetValue())
            self.rtThread.start()
            self.dlcq.put('initdlc')
            self.dlc2p.get()
            if self.com_ctrl.GetValue():
                self.ser.close()
            else:
                self.com_ctrl.SetValue(True)
                self.com_ctrl.Enable(False)
            self.dlcq.put('initSerial')
            
            self.run_expt.Enable(True)
            self.expt_id.Enable(True)
            self.pause_dlc.Enable(True)
            self.auto_pellet.Enable(False)
            self.auto_stim.Enable(False)
            self.pause_dlc.SetLabel('Pause DLC')
            self.pause_dlc.SetValue(False)
            self.Enable(True)
        else:
            self.dlcq.put('stopSerial')
            if self.com_ctrl.GetValue():
                self.comInit(event)
                self.com_ctrl.Enable(True)
            
            self.dlcq.close()
            self.dlc2p.close()
            self.rtThread.terminate()
            
            self.run_expt.Enable(False)
            self.expt_id.Enable(False)
            self.pause_dlc.Enable(False)
            self.auto_pellet.Enable(True)
            self.auto_stim.Enable(True)
            
    def pauseDLC(self, event):
        if self.pause_dlc.GetValue():
            self.dlcq.put('pause')
            self.pause_dlc.SetLabel('Resume DLC')
        else:
            self.dlcq.put('resume')
            self.pause_dlc.SetLabel('Pause DLC')

    def miniScope(self, event):
        if self.mini_scope.GetValue():
            self.msq = Queue()
            self.ms2p = Queue()
            self.msThread = mscam.CLARA_MS(self.msq, self.ms2p, self.camaq, self.frmaq)
            self.msThread.start()
            self.ms2p.get()
        else:
            self.msq.close()
            self.ms2p.close()
            self.msThread.terminate()

    def saveUser(self, event):
        if self.expSet.GetValue() > 0:
            self.user_cfg['exposure']=self.expSet.GetValue()
            self.user_cfg['framerate']=self.recSet.GetValue()
        clara.write_config(self.user_cfg)
            
    def quitButton(self, event):
        """
        Quits the GUI
        """
        print('Close event called')
        if self.rtdlc.GetValue():
            self.rtdlc.SetValue(False)
            self.dlcChecked(event)
        if self.play.GetValue():
            self.play.SetValue(False)
            self.liveFeed(event)
        if self.rec.GetValue():
            self.rec.SetValue(False)
            self.recordCam(event)
        if self.init.GetValue():
            self.init.SetValue(False)
            self.initCams(event)
        if self.com_ctrl.GetValue():
            self.com_ctrl.SetValue(False)
            self.comInit(event)
        if self.compressThread.is_alive():
            dlg = wx.MessageDialog(parent=None,message="Pausing until previous compression completes!",
                                   caption="Warning!", style=wx.OK|wx.ICON_EXCLAMATION)
            dlg.ShowModal()
            dlg.Destroy()
            while self.compressThread.is_alive():
                time.sleep(10)
        self.compressThread.terminate()   
        
        if self.mini_scope.GetValue():
            self.mini_scope.SetValue(False)
            self.miniScope(event)
        self.statusbar.SetStatusText("")
        self.Destroy()
    
def show():
    app = wx.App()
    MainFrame(None).Show()
    app.MainLoop()

if __name__ == '__main__':
    
    show()
