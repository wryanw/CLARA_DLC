# CLARA_DLC
Adaptation of deeplabcut for video acquisition and analysis using the BioElectrics Lab CLARA (closed-loop automated reaching apparatus)
The installation uses Ubuntu 16.04 and an Anaconda environment using python 3.6
CLARA_DLC_videoAcquisition_v10.py is the main data acquisition GUI.
Three FLIR cameras ( BFS-U3-16S2C-CS USB 3.1 Blackfly® S ) are required.
The main GUI uses CLARA_DLC_PySpin_v7.py to control them.  This requires an installation of Spinview and its Python SDK.
For real-time analysis, deeplabcut must be properly installed.  The GUI uses CLARA_RT_DLC_v5.py for realtime analysis.
