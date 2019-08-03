"""
CLARA toolbox
https://github.com/wryanw/CLARA
W Williamson, wallace.williamson@ucdenver.edu

"""


from __future__ import print_function
import wx
import wx.lib.dialogs
import cv2
import os
import numpy as np
from pathlib import Path
import pandas as pd
import sched, time
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from deeplabcut.CLARA_DLC_WRW import CLARA_DLC_utils as clara
from matplotlib.animation import FFMpegWriter
import datetime
# ###########################################################################
# Class for GUI MainFrame
# ###########################################################################
class ImagePanel(wx.Panel):

    def __init__(self, parent, gui_size, axesCt, **kwargs):
        h=np.amax(gui_size)/4
        w=np.amax(gui_size)/4
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER,size=(h,w))

        self.figure = Figure()
        self.axes = list()
        for a in range(0,axesCt):
            self.axes.append(self.figure.add_subplot(1,axesCt,a+1, frameon=False))
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

    def getColorIndices(self,img,bodyparts):
        """
        Returns the colormaps ticks and . The order of ticks labels is reversed.
        """
        norm = mcolors.Normalize(vmin=np.min(img), vmax=np.max(img))
        ticks = np.linspace(np.min(img),np.max(img),len(bodyparts))[::-1]
        return norm, ticks

class LabelsPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1,style=wx.SUNKEN_BORDER)
        self.Layout()
    def on_focus(self,event):
        pass

    def addRadioButtons(self,bodyparts,guiDim):
        """
        Adds radio buttons for each bodypart on the right panel
        """
        choices = [l for l in bodyparts]
        if guiDim == 0:
            style=wx.RA_SPECIFY_COLS
            self.choiceBox = wx.BoxSizer(wx.HORIZONTAL)
        else:
            style=wx.RA_SPECIFY_ROWS
            self.choiceBox = wx.BoxSizer(wx.VERTICAL)
        
        self.fieldradiobox = wx.RadioBox(self,label='Select a bodypart to label',
                                    style=style,choices=choices)
        buttSpace = 5
        
        self.label_frames = wx.ToggleButton(self, size=(150, -1), id=wx.ID_ANY, label="Label frames")
        self.choiceBox.Add(self.label_frames, 0, wx.ALL|wx.ALIGN_LEFT, buttSpace)
        self.label_frames.Enable(False)
        self.choiceBox.AddStretchSpacer(1)
        self.choiceBox.Add(self.fieldradiobox, 0, wx.ALL|wx.ALIGN_CENTER, buttSpace)
        self.fieldradiobox.Enable(False)
        
        self.stat = wx.Button(self, id=wx.ID_ANY, label="Stats")
        self.choiceBox.Add(self.stat, 0, wx.ALL|wx.ALIGN_CENTER, buttSpace)
        self.stat.Enable(False)
        
        self.choiceBox.AddStretchSpacer(1)
        
        self.jumpP = wx.Button(self, size=(30, -1), id=wx.ID_ANY, label="<")
        self.choiceBox.Add(self.jumpP, 0, wx.ALL|wx.ALIGN_RIGHT, buttSpace)
        self.jumpP.Enable(False)
        skip_text = wx.StaticText(self,label='Jump')
        self.choiceBox.Add(skip_text, 0, wx.ALL|wx.ALIGN_RIGHT, buttSpace)
        self.jumpN = wx.Button(self, size=(30, -1), id=wx.ID_ANY, label=">")
        self.choiceBox.Add(self.jumpN, 0, wx.ALL|wx.ALIGN_RIGHT, buttSpace)
        self.jumpN.Enable(False)
        
        self.move_label = wx.ToggleButton(self, id=wx.ID_ANY, label="Move")
        self.choiceBox.Add(self.move_label, 0, wx.ALL|wx.ALIGN_RIGHT, buttSpace)
        self.move_label.Enable(False)
        
        self.omit_label = wx.ToggleButton(self, id=wx.ID_ANY, label="Omit")
        self.choiceBox.Add(self.omit_label, 0, wx.ALL|wx.ALIGN_RIGHT, buttSpace)
        self.omit_label.Enable(False)
        self.SetSizerAndFit(self.choiceBox)
        self.Layout()
        
        return(self.stat,self.choiceBox,self.label_frames,self.move_label,self.omit_label,
               self.jumpP,self.jumpN,self.fieldradiobox)

    def replaceRadio(self,bodyparts,guiDim):
        self.choiceBox.Clear(True)

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
        self.choice_panel = LabelsPanel(vSplitter)
        self.widget_panel = WidgetPanel(topSplitter)
        if self.guiDim == 0:
            vSplitter.SplitHorizontally(self.image_panel,self.choice_panel, sashPosition=self.gui_size[1]*0.5)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitVertically(vSplitter, self.widget_panel,sashPosition=self.gui_size[0]*0.8)#0.9
        else:
            vSplitter.SplitVertically(self.image_panel,self.choice_panel, sashPosition=self.gui_size[0]*0.5)
            vSplitter.SetSashGravity(0.5)
            self.widget_panel = WidgetPanel(topSplitter)
            topSplitter.SplitHorizontally(vSplitter, self.widget_panel,sashPosition=self.gui_size[1]*0.8)#0.9
        topSplitter.SetSashGravity(0.5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(topSplitter, 1, wx.EXPAND)
        self.SetSizer(sizer)

###################################################################################################################################################
# Add Buttons to the WidgetPanel and bind them to their respective functions.
        
        self.stat,self.choiceBox,self.label_frames,self.move_label,self.omit_label,self.jumpP,self.jumpN,self.labelselect = self.choice_panel.addRadioButtons(['none'],self.guiDim)
        
        widgetSize = 7
        widgetsizer = wx.WrapSizer(orient=wx.HORIZONTAL)
        
        self.load_vids = wx.Button(self.widget_panel, size=(150, -1), id=wx.ID_ANY, label="Load Videos")
        widgetsizer.Add(self.load_vids, 1, wx.ALL, widgetSize)
        self.load_vids.Bind(wx.EVT_BUTTON, self.loadVids)
        self.load_vids.Enable(False)
        
        widgetsizer.AddStretchSpacer(1)
        
        self.new_config = wx.Button(self.widget_panel, size=(150, -1), id=wx.ID_ANY, label="New Training Set")
        widgetsizer.Add(self.new_config, 1, wx.ALL, widgetSize)
        self.new_config.Bind(wx.EVT_BUTTON, self.newConfig)
        
        self.use_syn = wx.ToggleButton(self.widget_panel, size=(150, -1), id=wx.ID_ANY, label="Access Synology")
        widgetsizer.Add(self.use_syn, 1, wx.ALL, widgetSize)
        
        widgetsizer.AddStretchSpacer(1)
        
        self.load_config = wx.Button(self.widget_panel, size=(150, -1), id=wx.ID_ANY, label="Load Config File")
        widgetsizer.Add(self.load_config, 1, wx.ALL, widgetSize)
        self.load_config.Bind(wx.EVT_BUTTON, self.loadConfig)
        
        self.show_anno = wx.CheckBox(self.widget_panel, id=wx.ID_ANY,label = 'Load labels from analysis')
        widgetsizer.Add(self.show_anno, 0, wx.ALL, widgetSize)
        self.show_anno.Bind(wx.EVT_CHECKBOX, self.loadAnnotations)
        
        self.slider = wx.Slider(self.widget_panel, -1, 0, 0, 100,size=(300, -1), style=wx.SL_HORIZONTAL | wx.SL_AUTOTICKS | wx.SL_LABELS )
        widgetsizer.Add(self.slider, 2, wx.ALL, widgetSize)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        self.slider.Enable(False)
        
        self.play = wx.ToggleButton(self.widget_panel, id=wx.ID_ANY, label="Play")
        widgetsizer.Add(self.play , 1, wx.ALL, widgetSize*1.25)
        self.play.Bind(wx.EVT_TOGGLEBUTTON, self.fwrdPlay)
        self.play.Enable(False)
        
        widgetsizer.AddStretchSpacer(1)
        
        self.grab = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Get Frame")
        widgetsizer.Add(self.grab , 1, wx.ALL , widgetSize)
        self.grab.Bind(wx.EVT_BUTTON, self.chooseFrame)
        self.grab.Enable(False)
        
        widgetsizer.AddStretchSpacer(1)

        self.save = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Make Demo")
        widgetsizer.Add(self.save , 1, wx.ALL , widgetSize)
        self.save.Bind(wx.EVT_BUTTON, self.makeDemoVid)
        self.save.Enable(False)

#        Making radio selection for vid speed
        self.speedOps = [-50,-10,-1,1,10,50]
        viewopts = ['-500','-100','-10 ','10  ','100 ','500']
        choices = [l for l in viewopts]
        self.speedbox = wx.RadioBox(self.widget_panel,label='Playback speed (fps)', majorDimension=1, style=wx.RA_SPECIFY_ROWS,choices=choices)
        widgetsizer.Add(self.speedbox, 1, wx.ALL, widgetSize)
        self.speedbox.Bind(wx.EVT_RADIOBOX, self.playSpeed)
        self.speedbox.SetSelection(3)
        self.speedbox.Enable(False)
        
        self.start_frames_sizer = wx.BoxSizer(wx.VERTICAL)
        self.end_frames_sizer = wx.BoxSizer(wx.VERTICAL)

        self.startFrame = wx.SpinCtrl(self.widget_panel, value='0', size=(150, -1))#,style=wx.SP_VERTICAL)
        self.startFrame.Enable(False)
        self.start_frames_sizer.Add(self.startFrame, 1, wx.EXPAND|wx.ALIGN_LEFT, widgetSize)
        self.startFrame.Bind(wx.EVT_SPINCTRL, self.updateSlider)
        start_text = wx.StaticText(self.widget_panel, label='Start Frame Index')
        self.start_frames_sizer.Add(start_text, 1, wx.EXPAND|wx.ALIGN_LEFT, widgetSize)
         
        self.endFrame = wx.SpinCtrl(self.widget_panel, value='1', size=(150, -1))#, min=1, max=120)
        self.endFrame.Enable(False)
        self.end_frames_sizer.Add(self.endFrame, 1, wx.EXPAND|wx.ALIGN_LEFT, widgetSize)
        self.startFrame.Bind(wx.EVT_SPINCTRL, self.updateSlider)
        end_text = wx.StaticText(self.widget_panel, label='Frames Remaining')
        self.end_frames_sizer.Add(end_text, 1, wx.EXPAND|wx.ALIGN_LEFT, widgetSize)
        
        widgetsizer.Add(self.start_frames_sizer, 1, wx.ALL, widgetSize)
        widgetsizer.AddStretchSpacer(1)
        widgetsizer.Add(self.end_frames_sizer, 1, wx.ALL, widgetSize)
        
        self.train = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Train")
        widgetsizer.Add(self.train , 1, wx.ALL, widgetSize)
        self.train.Bind(wx.EVT_BUTTON, self.trainNetwork)
        
        widgetsizer.AddStretchSpacer(1)
        
        self.anal = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Analyze")
        widgetsizer.Add(self.anal, 1, wx.ALL, widgetSize)
        self.anal.Bind(wx.EVT_BUTTON, self.analyzeVids)
        
        widgetsizer.AddStretchSpacer(1)
        
        self.quit = wx.Button(self.widget_panel, id=wx.ID_ANY, label="Quit")
        widgetsizer.Add(self.quit , 1, wx.ALL, widgetSize)
        self.quit.Bind(wx.EVT_BUTTON, self.quitButton)
        self.Bind(wx.EVT_CLOSE, self.quitButton)
        
        widgetsizer.Add(self, 1, wx.EXPAND)
        self.widget_panel.SetSizer(widgetsizer)
        widgetsizer.Fit(self.widget_panel)
        self.widget_panel.Layout()
        
        self.timer = wx.Timer(self, wx.ID_ANY)
        self.videos = list()
        self.shuffle = 1
        self.trainingsetindex = 0
        self.currAxis = 0
        self.x1 = 0
        self.y1 = 0
        self.vid = list()
        self.videoList = list()
        
        usrdatadir = os.path.dirname(os.path.realpath(__file__))
        _, user = os.path.split(Path.home())
        self.usrdatapath = os.path.join(usrdatadir, '%s_userdata.txt' % user)
        if os.path.isfile(self.usrdatapath):
            usrdata = open(self.usrdatapath, 'r')
            self.config_path = usrdata.readline()
            usrdata.close()
            self.load_vids.Enable(True)
            self.statusbar.SetStatusText('Current config: %s' % self.config_path)

    def OnKeyPressed(self, event):
        
#        print(event.GetKeyCode())
#        print(wx.WXK_RETURN)
        
        if self.play.IsEnabled() == True:
            if event.GetKeyCode() == wx.WXK_UP:
                if self.move_label.GetValue():
                    event.inaxes = self.currAxis
                    event.xdata = self.x1
                    event.ydata = self.y1-1
                    self.newLabelCt-=1
                    self.onClick(event)
                else:
                    if self.play.GetValue() == True:
                        self.play.SetValue(False)
                        self.fwrdPlay(event=None)
                    self.slider.SetValue(self.slider.GetValue()+1)
                    self.OnSliderScroll(event)
            elif event.GetKeyCode() == wx.WXK_DOWN:
                if self.move_label.GetValue():
                    event.inaxes = self.currAxis
                    event.xdata = self.x1
                    event.ydata = self.y1+1
                    self.newLabelCt-=1
                    self.onClick(event)
                else:
                    if self.play.GetValue() == True:
                        self.play.SetValue(False)
                        self.fwrdPlay(event=None)
                    self.slider.SetValue(self.slider.GetValue()-1)
                    self.OnSliderScroll(event)
            elif event.GetKeyCode() == wx.WXK_LEFT:
                if self.move_label.GetValue():
                    event.inaxes = self.currAxis
                    event.xdata = self.x1-1
                    event.ydata = self.y1
                    self.newLabelCt-=1
                    self.onClick(event)
                elif (self.play.GetValue() == False) and (self.labelselect.IsEnabled):
                    if self.labelselect.GetSelection() > 0:
                        self.labelselect.SetSelection(self.labelselect.GetSelection()-1)
                        self.update()
                elif self.speedbox.GetSelection() > 0:
                    self.speedbox.SetSelection(self.speedbox.GetSelection()-1)
                    self.playSpeed(event=None)
            elif event.GetKeyCode() == wx.WXK_RIGHT:
                if self.move_label.GetValue():
                    event.inaxes = self.currAxis
                    event.xdata = self.x1+1
                    event.ydata = self.y1
                    self.newLabelCt-=1
                    self.onClick(event)
                elif (self.play.GetValue() == False) and (self.labelselect.IsEnabled):
                    if self.labelselect.GetSelection() < (self.labelselect.GetCount()-1):
                        self.labelselect.SetSelection(self.labelselect.GetSelection()+1)
                        self.update()
                elif self.speedbox.GetSelection() < (self.speedbox.GetCount()-1):
                    self.speedbox.SetSelection(self.speedbox.GetSelection()+1)
                    self.playSpeed(event=None)
            elif event.GetKeyCode() == wx.WXK_SPACE:
                if self.play.GetValue() == True:
                    self.play.SetValue(False)
                else:   
                    self.play.SetValue(True)
                self.fwrdPlay(event=None)
            elif event.GetKeyCode() == wx.WXK_RETURN:
                if self.move_label.GetValue():
                    self.move_label.SetValue(0)
        event.Skip()

    def fwrdPlay(self, event):
        if self.play.GetValue() == True:
            if not self.timer.IsRunning():
                self.timer.Start(100)
            self.play.SetLabel('Stop')
        else:
            if self.timer.IsRunning():
                self.timer.Stop()
            self.play.SetLabel('Play')
        
    def playSpeed(self, event):
        wasRunning = False
        if self.timer.IsRunning():
            wasRunning = True
            self.timer.Stop()
        self.playSkip = self.speedOps[self.speedbox.GetSelection()]
        self.slider.SetPageSize(pow(5,(self.speedbox.GetSelection()-1)))
        self.play.SetFocus()
        if wasRunning:
            self.timer.Start(100)
            
    def newConfig(self, event):
        userlist = ['WRW','Jordan','Spencer']
        dlgE = wx.SingleChoiceDialog(self, "Select user name",'The Caption',userlist,wx.CHOICEDLG_STYLE)
        if dlgE.ShowModal() == wx.ID_OK:
            experimenter = dlgE.GetStringSelection()
            dlgP = wx.TextEntryDialog(self, 'Enter a project name')
            if dlgP.ShowModal() == wx.ID_OK:
                project = dlgP.GetValue()
                dlgP.Destroy()
                if self.use_syn.GetValue():
                    startDir = '/run/user/1001/gvfs/smb-share:server=synology,share=whsynology/BIOElectricsLab/RAW_DATA/AutomatedBehavior'
                else:
                    startDir = str(Path.home())
                dlgD = wx.DirDialog(self, 'Choose a project directory',startDir)
                if dlgD.ShowModal() == wx.ID_OK:
                    working_directory = dlgD.GetPath()
                    dlgD.Destroy()
                    dlgV = wx.FileDialog(self, 'Select a starting video')
                    wildcard = "Avi files (*.avi)|*.avi"
                    dlgV.SetWildcard(wildcard)
                    dlgV.SetDirectory(startDir)
                    
                    if dlgV.ShowModal() == wx.ID_OK:
                        videoSrc = dlgV.GetPath()
                        vidDir, vidName = os.path.split(videoSrc)
                        vidName, vidExt = os.path.splitext(vidName)
                        self.videoList = list()
                        for vidfile in os.listdir(vidDir):
                            if vidfile.endswith(vidExt):
                                vidParts = vidName.split('_')[0:3]
                                vidBase = '_'.join(vidParts)
                                if vidBase in vidfile:
                                    self.videoList.append(os.path.join(vidDir,vidfile))
                        if len(self.videoList) == 3:
                            self.config_path = clara.create_CLARA_project(self.videoList, project, experimenter, working_directory)
                            usrdatadir = os.path.dirname(os.path.realpath(__file__))
                            _, user = os.path.split(Path.home())
                            usrdatapath = os.path.join(usrdatadir, '%s_userdata.txt' % user)
                            usrdata = open(usrdatapath, 'w')
                            usrdata.write(self.config_path)
                            self.statusbar.SetStatusText('Current config: %s' % self.config_path)
                            self.load_vids.Enable(True)
                        else:
                            print('Not enough videos found!')
                    else:
                        dlgV.Destroy()
                else:
                    dlgD.Destroy()
            else:
                dlgP.Destroy()
        else:
            dlgE.Destroy()
            
        
    def loadConfig(self, event):
        wildcard = "Config files (*.yaml)|*.yaml"
        dlg = wx.FileDialog(self, "Select a config file.")
        dlg.SetWildcard(wildcard)
        if self.use_syn.GetValue():
            startDir = '/run/user/1001/gvfs/smb-share:server=synology,share=whsynology/BIOElectricsLab/RAW_DATA/AutomatedBehavior'
        else:
            startDir = str(Path.home())
        dlg.SetDirectory(startDir)
        if dlg.ShowModal() == wx.ID_OK:
            self.config_path = dlg.GetPath()
            self.load_vids.Enable(True)
            usrdatadir = os.path.dirname(os.path.realpath(__file__))
            _, user = os.path.split(Path.home())
            usrdatapath = os.path.join(usrdatadir, '%s_userdata.txt' % user)
            usrdata = open(usrdatapath, 'w')
            usrdata.write(self.config_path)
            self.statusbar.SetStatusText('Current config: %s' % self.config_path)
            usrdata.close()
        else:
            dlg.Destroy()
        
        
    def loadVids(self, event):
        if self.label_frames.GetValue() == True:
            self.label_frames.SetValue(False)
            self.labelFrames()
        if len(self.vid) > 0:
            for vid in self.vid:
                vid.release()
        
        self.figure,self.axes,self.canvas = self.image_panel.getfigure()
        if len(self.axes[0].get_children()) > 0:
            for hax in self.axes:
                hax.clear()
            self.figure.canvas.draw()
            
# =============================================================================
#        self.videos = ['/home/bioelectrics/Documents/CLARA_RT_DLC/videos/20190611_unit03_session001_frontCamID-0000.avi',
#                       '/home/bioelectrics/Documents/CLARA_RT_DLC/videos/20190611_unit03_session001_sideCamID-0000.avi',
#                       '/home/bioelectrics/Documents/CLARA_RT_DLC/videos/20190611_unit03_session001_topCamID-0000.avi']
#        self.videoList = self.videos
        self.config_path='/home/bioelectrics/Documents/CLARA_RT_DLC-WRW-2019-07-10/config.yaml'
# =============================================================================
        
        
        if not len(self.videoList):
            dlgV = wx.FileDialog(self, 'Select a video')
            wildcard = "Avi files (*.avi)|*.avi"
            dlgV.SetWildcard(wildcard)
            if self.use_syn.GetValue():
                startDir = '/run/user/1001/gvfs/smb-share:server=synology,share=whsynology/BIOElectricsLab/RAW_DATA/AutomatedBehavior'
            else:
                startDir = str(Path.home())
            dlgV.SetDirectory(startDir)
            if dlgV.ShowModal() == wx.ID_OK:
                videoSrc = dlgV.GetPath()
                vidDir, vidName = os.path.split(videoSrc)
                vidName, vidExt = os.path.splitext(vidName)
                self.videoList = list()
                for vidfile in os.listdir(vidDir):
                    if vidfile.endswith(vidExt):
                        vidParts = vidName.split('_')[0:3]
                        vidBase = '_'.join(vidParts)
                        if vidBase in vidfile:
                            self.videoList.append(os.path.join(vidDir,vidfile))
                if len(self.videoList) != 3:
                    print('Not enough videos found!')
                    return
            else:
                dlgV.Destroy()
                return

        self.cfg = auxiliaryfunctions.read_config(self.config_path)
        self.currFrame = 0
        self.bodyparts = self.cfg['bodyparts']
        # checks for unique bodyparts
        if len(self.bodyparts)!=len(set(self.bodyparts)):
            print("Error - bodyparts must have unique labels! Please choose unique bodyparts in config.yaml file and try again.")

        cppos = self.choice_panel.GetPosition()
        cprect = self.choice_panel.GetRect()
        cpsize = self.choice_panel.GetSize()
        self.choice_panel.replaceRadio(self.bodyparts,self.guiDim)
        self.stat,self.choiceBox,self.label_frames,self.move_label,self.omit_label,self.jumpP,self.jumpN,self.labelselect = self.choice_panel.addRadioButtons(self.bodyparts,self.guiDim)
        self.choice_panel.SetPosition(cppos)
        self.choice_panel.SetRect(cprect)
        self.choice_panel.SetSize(cpsize)
        self.labelselect.Bind(wx.EVT_RADIOBOX, self.resetFocus)
        self.label_frames.Bind(wx.EVT_TOGGLEBUTTON, self.labelFrames)
        self.omit_label.Bind(wx.EVT_TOGGLEBUTTON, self.omitLabel)
        self.jumpP.Bind(wx.EVT_BUTTON, self.jumpFrame)
        self.jumpN.Bind(wx.EVT_BUTTON, self.jumpFrame)
        self.stat.Bind(wx.EVT_BUTTON, self.showStats)
        
        self.colormap = plt.get_cmap(self.cfg['colormap'])
        self.colormap = self.colormap.reversed()
        self.markerSize = 6
        self.alpha = self.cfg['alphavalue']
        self.iterationindex=self.cfg['iteration']
        
        self.vid = list()
        self.im = list()
        self.videos = list()
        videoOrder = ['side','front','top']
        for key in enumerate(videoOrder):
            for video in enumerate(self.videoList):
                if key[1] in video[1]:
                    self.videos.append(video[1])
        clara.add_CLARA_videos(self.config_path,self.videos)
        self.frameList = list()
        self.cropPts = list()
        for vndx, video in enumerate(self.videos):
            video_source = Path(video).resolve()
            self.vid.append(cv2.VideoCapture(str(video_source)))
            self.vid[vndx].set(1,self.currFrame)
            ret, frame = self.vid[vndx].read()
            frmW = self.vid[vndx].get(cv2.CAP_PROP_FRAME_WIDTH)
            frmH = self.vid[vndx].get(cv2.CAP_PROP_FRAME_HEIGHT)
            x1 = np.floor(frmW/2-vndx*.25*frmW)
            if vndx == 1:
                x1+=np.round(frmW/8)
            x1 = int(x1)
            xW = int(np.floor(frmW/2))
            y1 = int(70)
            yH = int(frmH)-90
            if frmW == 720.0:
                x1 = int(20)
                xW = int(180)
                y1 = int(220)
                yH = int(180)
            self.cropPts.append([x1,xW,y1,yH])
            frame = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#            cpt = self.cropPts[vndx]
#            frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
            self.im.append(self.axes[vndx].imshow(frame))
            self.frameList.append(frame)
            
            
        self.numberFrames = int(self.vid[vndx].get(cv2.CAP_PROP_FRAME_COUNT))
        self.norm,self.colorIndex = self.image_panel.getColorIndices(frame,self.bodyparts)
        self.strwidth = int(np.ceil(np.log10(self.numberFrames)))
        # Set the values of slider and range of frames
        self.slider.SetMax(self.numberFrames-1)
        self.endFrame.SetMax(self.numberFrames-1)
        self.endFrame.SetValue(self.numberFrames-1)
        self.startFrame.SetValue(0)
        self.endFrame.Bind(wx.EVT_SPINCTRL,self.updateSlider)#wx.EVT_SPIN
        self.startFrame.Bind(wx.EVT_SPINCTRL,self.updateSlider)#wx.EVT_SPIN

        self.grab.Enable(True)
        self.save.Enable(True)
        self.play.Enable(True)
        self.slider.Enable(True)
        self.speedbox.Enable(True)
        self.startFrame.Enable(True)
        self.endFrame.Enable(True)
        self.label_frames.Enable(True)
        self.newLabelCt = 0
        self.videoList = list()
        self.Bind(wx.EVT_TIMER, self.vidPlayer, self.timer)
        self.widget_panel.Layout()
        self.slider.SetValue(self.currFrame)
        self.playSpeed(event)
        self.OnSliderScroll(event)

    def resetFocus(self, event):
        self.update()
        self.play.SetFocus()
        
    def jumpFrame(self, event):
        self.Disable
        if self.jumpP == event.GetEventObject():
            prevFrm = list()
            for ndx in range(0,len(self.axes)):
                df = np.where(self.dataFrame[ndx].count(1).get_values()[:self.currFrame] > 0)
                if not len(df[0]):
                    prevFrm.append(self.currFrame)
                else:
                    prevFrm.append(int(np.amax(df[0])))
            self.currFrame = np.amax(prevFrm)
        else:
            nextFrm = list()
            for ndx in range(0,len(self.axes)):
                df = np.where(self.dataFrame[ndx].count(1).get_values()[self.currFrame:] > 0)
                if not len(df[0]):
                    nextFrm.append(self.currFrame)
                else:
                    nextFrm.append(int(np.amin(df[0]))+self.currFrame)
            self.currFrame = np.amin(nextFrm)
        self.Enable
        self.slider.SetValue(self.currFrame)
        self.OnSliderScroll(event)
        
    def labelFrames(self, event):
        """
        Show the DirDialog and ask the user to change the directory where machine labels are stored
        """
        if self.label_frames.GetValue():
            self.Disable
            data_path = Path(self.config_path).parents[0] / 'labeled-data'
            self.label_dirs = [data_path/Path(i.stem) for i in [Path(vp) for vp in self.videos]]
            self.scorer = self.cfg['scorer']
            self.dataFrame = list()
            self.relativeimagenames = list()
            for frm in range(0,self.numberFrames):
                img_name = 'img'+str(frm).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
                self.relativeimagenames.append(img_name)
            
            for ndx, dirs in enumerate(self.label_dirs):
                dataFilePath = os.path.join(dirs,'CollectedData_'+self.scorer+'.h5')
                
                if os.path.isfile(dataFilePath):
                    # Reading the existing dataset,if already present
                    dataFilePath = os.path.join(dirs,'CollectedData_'+self.scorer+'.h5')
                    self.dataFrame.append(pd.read_hdf(dataFilePath,'df_with_missing'))
                    self.dataFrame[ndx].sort_index(inplace=True)
                else:
                    a = np.empty((len(self.relativeimagenames),2,))
                    a[:] = np.nan
                    df = None
                    for bodypart in self.bodyparts:
                        index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                        frame = pd.DataFrame(a, columns = index, index = self.relativeimagenames)
                        df = pd.concat([df, frame],axis=1)
                    self.dataFrame.append(df)
                    
        # Extracting the list of new labels
                oldBodyParts = self.dataFrame[ndx].columns.get_level_values(1)
                _, idx = np.unique(oldBodyParts, return_index=True)
                oldbodyparts2plot =  list(oldBodyParts[np.sort(idx)])
                self.new_bodyparts =  [x for x in self.bodyparts if x not in oldbodyparts2plot ]
        # Checking if user added a new label
                if self.new_bodyparts != []: # i.e. new labels
                    print('New body parts found!')
                    a = np.empty((len(self.relativeimagenames),2,))
                    a[:] = np.nan
                    for bodypart in self.new_bodyparts:
                        index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],names=['scorer', 'bodyparts', 'coords'])
                        frame = pd.DataFrame(a, columns = index, index = self.relativeimagenames)
                        self.dataFrame[ndx] = pd.concat([self.dataFrame[ndx], frame],axis=1)
                        
            self.cidClick = self.canvas.mpl_connect('button_press_event', self.onClick)
            self.circleH = list()
            self.circleP = list()
            self.croprec = list()
            for hax in self.axes:
                hax.clear()
            
            self.im = list()
            for vndx, video in enumerate(self.videos):
                self.points = [0,0,1.0]
                circ = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=self.alpha)]
                self.circleH.append(self.axes[vndx].add_patch(circ[0]))
                circ = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=self.alpha)]
                self.circleP.append(self.axes[vndx].add_patch(circ[0]))
                cpt = self.cropPts[vndx]
                self.vid[vndx].set(1,self.currFrame)
                ret, frame = self.vid[vndx].read()
                frame = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                self.im.append(self.axes[vndx].imshow(frame))
                
            self.move_label.Enable(True)
            self.omit_label.Enable(True)
            self.labelselect.Enable(True)
            self.jumpP.Enable(True)
            self.jumpN.Enable(True)
            self.stat.Enable(True)
            self.Enable
            self.update()
        else:
            for p in range(0,3):
                self.circleH[p].remove()
                self.circleP[p].remove()
            self.move_label.Enable(False)
            self.omit_label.Enable(False)
            self.labelselect.Enable(False)
            self.jumpP.Enable(False)
            self.jumpN.Enable(False)
            self.stat.Enable(False)
            self.figure.canvas.draw()
            self.saveDataSet()
            
    def showStats(self, event):
        runTot = np.zeros((len(self.bodyparts),1))
        sumry = '---Subtotals---'
        for ndx,df in enumerate(self.dataFrame):
            datastats = df.count().get_values()
            if ndx > 0:
                sumry+='\n'
            sumry+= '\n%s:\n' % os.path.split(self.videos[ndx])[1]
            for n,s in enumerate(range(0,len(datastats),2)):
                if n > 0:
                    sumry+=' - '
                runTot[n]+=datastats[s]
                sumry+='%s: %d' % (self.bodyparts[n],datastats[s])
                
        sumry+='\n\n---Total Counts---\n'
        for ndx, t in enumerate(runTot):
            if n > 0:
                sumry+=' - '
            sumry+='%s: %d' % (self.bodyparts[ndx],t)
        dlg = wx.lib.dialogs.ScrolledMessageDialog(self, sumry, "Summary Stats")
        dlg.ShowModal()
        dlg.Destroy()
        
    def saveDataSet(self):
        """
        Saves the final dataframe
        """
        self.Disable
        for ndx, dirs in enumerate(self.label_dirs):
            self.dataFrame[ndx].sort_index(inplace=True)
            self.dataFrame[ndx].to_csv(os.path.join(dirs,"CollectedData_" + self.scorer + ".csv"))
            self.dataFrame[ndx].to_hdf(os.path.join(dirs,"CollectedData_" + self.scorer + '.h5'),'df_with_missing',format='table', mode='w')
        print('Data saved!')
        self.Enable
        
    def onClick(self,event):
        """
        This function adds labels and auto advances to the next label.
        """
        bp = self.labelselect.GetString(self.labelselect.GetSelection())
        for ndx, hax in enumerate(self.axes):
            if event.inaxes == hax:
                self.x1 = event.xdata
                self.y1 = event.ydata
                if self.omit_label.GetValue():
                    self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'x' ] = np.nan
                    self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'y' ] = np.nan
                    self.omit_label.SetValue(0)
                    self.omitLabel(event=None)
                    self.newLabelCt-=1
                else:
                    img_name = str(self.label_dirs[ndx]) +'/img'+str(self.currFrame).zfill(int(np.ceil(np.log10(self.numberFrames)))) + '.png'
                    if not os.path.isfile(img_name):
                        cv2.imwrite(img_name, cv2.cvtColor(self.frameList[ndx], cv2.COLOR_RGB2BGR))
                    if (bp == 'Hand') or (bp == 'Pellet'):
                        self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'x' ] = self.x1
                        self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'y' ] = self.y1
                        self.move_label.SetValue(1)
                        self.currAxis = hax
                    else:
                        hx = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, 'Hand', 'x' ]
                        hy = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, 'Hand', 'y' ]
                        if not np.isnan(hx):
                            self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'x' ] = hx
                            self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp, 'y' ] = hy
                        else:
                            print('Label hand first')
                    self.newLabelCt+=1
                    
            if self.newLabelCt > 30:
                self.Disable
                self.saveDataSet()
                self.newLabelCt = 0
                self.Enable            
            self.update()
            
    def omitLabel(self, event):
        if self.omit_label.GetValue():
            self.omit_label.SetLabel('Click it')
        else:
            self.omit_label.SetLabel('Omit')
        
    def loadAnnotations(self, event):
        if self.show_anno.GetValue():
            videoSrc = self.videos[0]
            vidDir, vidName = os.path.split(videoSrc)
            vidName, vidExt = os.path.splitext(vidName)
            onlyfiles = [f for f in os.listdir(vidDir) if os.path.isfile(os.path.join(vidDir, f))]
            h5files = [h for h in onlyfiles if '.h5' in h]
            h5parts = [(m.split('DeepCut')[1]) for m in h5files]
            h5unique = [h5parts[0]]
            for h in h5parts:
                found = False
                for u in h5unique:
                    if u == h:
                        found = True
                if not found:
                    h5unique.append(h)
            if len(h5unique) > 1:
                dlgE = wx.SingleChoiceDialog(self, "Select the analysis to use:",'The Caption',h5unique,wx.CHOICEDLG_STYLE)
                if dlgE.ShowModal() == wx.ID_OK:
                    h5tag = dlgE.GetStringSelection()
                else:
                    dlgE.Destroy()
                    return
            else:
                h5tag = h5unique[0]
            self.scorer = 'DeepCut%s' % os.path.splitext(h5tag)[0]
            h5list = ['%s%s.h5' % (j,self.scorer) for j in [os.path.splitext(v)[0] for v in self.videos]]
            self.df_likelihood = list()
            self.df_x = list()
            self.df_y = list()
            self.circleH = list()
            self.circleP = list()
            self.croprec = list()
            self.textH = list()
            for vndx, video in enumerate(self.videos):
                Dataframe = pd.read_hdf(h5list[vndx])
                self.df_likelihood.append(np.empty((len(self.bodyparts),self.numberFrames)))
                self.df_x.append(np.empty((len(self.bodyparts),self.numberFrames)))
                self.df_y.append(np.empty((len(self.bodyparts),self.numberFrames)))
                for bpindex, bp in enumerate(self.bodyparts):
                    self.df_likelihood[vndx][bpindex,:]=Dataframe[self.scorer][bp]['likelihood'].values
                    self.df_x[vndx][bpindex,:]=Dataframe[self.scorer][bp]['x'].values
                    self.df_y[vndx][bpindex,:]=Dataframe[self.scorer][bp]['y'].values
                self.points = [0,0,1.0]
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=self.alpha)]
                self.circleH.append(self.axes[vndx].add_patch(circle[0]))
                circle = [patches.Circle((self.points[0], self.points[1]), radius=self.markerSize, fc = None , alpha=self.alpha)]
                self.circleP.append(self.axes[vndx].add_patch(circle[0]))
                cpt = self.cropPts[vndx]
                croprec = [patches.Rectangle((cpt[0]+1,cpt[2]+1), cpt[1]-3, cpt[3]-3, fill=False, ec = [0.25,0.75,0.25], linewidth=2, linestyle='-',alpha=0.0)]
                self.croprec.append(self.axes[vndx].add_patch(croprec[0]))
                tndcs = [-1,-2,0]
                message = ' '
                for n,t in enumerate(tndcs):
                    test = .5555
                    message+= '%s-%s\n' % (self.bodyparts[t],f'{test:.4f}')
                self.textH.append(self.axes[vndx].text(10,170,message,color=[1,1,1],fontsize=16))
            
            self.update()
        else:
            for p in range(0,3):
                self.circleH[p].remove()
                self.circleP[p].remove()
                self.croprec[p].remove()
            self.figure.canvas.draw()
        
        
    def quitButton(self, event):
        """
        Quits the GUI
        """
        print('Close event called')
        self.statusbar.SetStatusText("")
        if self.label_frames.GetValue():
            self.saveDataSet()
        self.Destroy()
    
    def vidPlayer(self, event):
        deltaF = (self.playSkip)
        newFrame = self.currFrame+deltaF
        endVal = self.endFrame.GetValue()
        if (newFrame < 0) or (deltaF > endVal):
            if self.timer.IsRunning():
                self.timer.Stop()
                self.play.SetValue(False)
        else:
            self.endFrame.SetValue(endVal-deltaF)
            self.slider.SetValue(newFrame)
            self.OnSliderScroll(event)
        
    def updateSlider(self,event):
        self.slider.SetValue(self.startFrame.GetValue())
        self.OnSliderScroll(event)
    
    def OnSliderScroll(self, event):
        """
        Slider to scroll through the video
        """
        self.currFrame = self.slider.GetValue()
        self.endFrame.SetMax(self.numberFrames-self.currFrame)
        if self.endFrame.GetValue() > (self.numberFrames-self.currFrame):
            self.endFrame.SetValue(self.numberFrames-self.currFrame)
        if event.GetEventCategory() == 2 and not (self.endFrame == event.GetEventObject()):
            self.endFrame.SetValue(self.numberFrames-self.currFrame-1)
            
        self.startFrame.SetValue(self.currFrame)
        self.update()
    
    def update(self):
        """
        Updates the image with the current slider index
        """
        for ndx, im in enumerate(self.im):
            # Draw
            self.vid[ndx].set(1,self.currFrame)
            ret, frame = self.vid[ndx].read()
            frame = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cpt = self.cropPts[ndx]
            
            if self.show_anno.GetValue():
                class_test = np.amax(self.df_likelihood[ndx][0:-3,self.currFrame])
                bpindexP = -1
                bpindexH = -2
                pellet_test = np.amax(self.df_likelihood[ndx][bpindexP,self.currFrame])
                hand_test = np.amax(self.df_likelihood[ndx][bpindexH,self.currFrame])
                bpindexC = np.argmax(self.df_likelihood[ndx][0:-3,self.currFrame])
                if class_test > 0.9:
                    self.drawCirc(self.circleH,ndx,bpindexC)
                elif hand_test > 0.9:
                    self.drawCirc(self.circleH,ndx,bpindexH)
                else:
                    self.circleH[ndx].set_alpha(0.0)
                if pellet_test > 0.9:
                    self.drawCirc(self.circleP,ndx,bpindexP)
                else:
                    self.circleP[ndx].set_alpha(0.0)
                message = ''    
                message+= '%s-%s\n' % (self.bodyparts[bpindexP],f'{pellet_test:.4f}')
                message+= '%s-%s\n' % (self.bodyparts[bpindexH],f'{hand_test:.4f}')
                message+= '%s-%s\n' % (self.bodyparts[bpindexC],f'{class_test:.4f}')
                self.textH[ndx].set_text(message)
                
            elif self.label_frames.GetValue():
                frame = frame[cpt[2]:cpt[2]+cpt[3],cpt[0]:cpt[0]+cpt[1]]
                self.frameList[ndx] = frame
                #Hand labels
                bp = self.bodyparts[self.labelselect.GetSelection()]
                color = self.colormap(self.norm(self.colorIndex[self.labelselect.GetSelection()]))
                xpt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'x' ]
                ypt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'y' ]
                if np.isnan(xpt) or bp == 'Pellet':
                    bp = 'Hand'
                    color = self.colormap(self.norm(self.colorIndex[-2]))
                    xpt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'x' ]
                    ypt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'y' ]
                    if np.isnan(xpt):
                        self.circleH[ndx].set_alpha(0.0)
                    else:
                        self.points = [int(xpt),int(ypt),1.0]
                        self.circleH[ndx].set_facecolor(color)
                        self.circleH[ndx].set_center(self.points)
                        self.circleH[ndx].set_alpha(self.alpha)
                else:
                    self.points = [int(xpt),int(ypt),1.0]
                    self.circleH[ndx].set_facecolor(color)
                    self.circleH[ndx].set_center(self.points)
                    self.circleH[ndx].set_alpha(self.alpha)
                #Pellet label
                bp = 'Pellet'
                color = self.colormap(self.norm(self.colorIndex[-1]))
                xpt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'x' ]
                ypt = self.dataFrame[ndx].loc[self.relativeimagenames[self.currFrame]][self.scorer, bp,'y' ]
                if np.isnan(xpt):
                    self.circleP[ndx].set_alpha(0.0)
                else:
                    self.points = [int(xpt),int(ypt),1.0]
                    self.circleP[ndx].set_facecolor(color)
                    self.circleP[ndx].set_center(self.points)
                    self.circleP[ndx].set_alpha(self.alpha)
            im.set_data(frame)
        self.figure.canvas.draw()
        
    def drawCirc(self, handle, ndx, bpndx):
        color = self.colormap(self.norm(self.colorIndex[bpndx]))
        self.points = [int(self.df_x[ndx][bpndx,self.currFrame]),int(self.df_y[ndx][bpndx,self.currFrame]),1.0]
        cpt = self.cropPts[ndx]
        self.points[0] = self.points[0]+cpt[0]
        self.points[1] = self.points[1]+cpt[2]
        handle[ndx].set_facecolor(color)
        handle[ndx].set_center(self.points)
        handle[ndx].set_alpha(self.alpha)
            
            
    def trainNetwork(self,event):
        clara.create_training_dataset_CLARA(self.config_path,num_shuffles=1)
        deeplabcut.train_network(self.config_path)

    def analyzeVids(self,event):
        for ndx,v in enumerate(self.videos):
            cpt = self.cropPts[ndx]
            crp = [cpt[0],cpt[0]+cpt[1],cpt[2],cpt[2]+cpt[3]]
            deeplabcut.analyze_videos(self.config_path,[v], videotype='.mp4',
                                      save_as_csv=True, cropping=crp)
            
    def chooseFrame(self):
        print('need to make')
        
    def makeDemoVid(self,event):
        writer = FFMpegWriter(fps=2)
        dateStr = datetime.datetime.now().strftime("%Y%m%d%H%M")
        base_dir = '/home/bioelectrics/Documents/DemoVids'
        savePath = os.path.join(base_dir,dateStr+"vidExp.mp4")
        with writer.saving(self.figure, savePath):
            while True:
                deltaF = (self.playSkip)
                newFrame = self.currFrame+deltaF
                endVal = self.endFrame.GetValue()
                if (newFrame < 0) or (deltaF > endVal):
                    break
                writer.grab_frame()
                self.vidPlayer(event)
    
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