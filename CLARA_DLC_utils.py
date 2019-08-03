"""
DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu

Boilerplate project creation inspired from DeepLabChop
by Ronny Eichler
"""
import os, socket
from pathlib import Path
from deeplabcut import DEBUG
import cv2
import numpy as np
import pandas as pd
import os.path
import yaml
import ruamel.yaml
from deeplabcut.utils import auxiliaryfunctions, auxfun_models

def create_CLARA_project(videos, project, experimenter, working_directory):
    """Creates a new project directory, sub-directories and a basic configuration file. The configuration file is loaded with the default values. Change its parameters to your projects need.

    """
    from datetime import datetime as dt
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return
    video_path = project_path / 'videos'
    data_path = project_path / 'labeled-data'
    shuffles_path = project_path / 'training-datasets'
    results_path = project_path / 'dlc-models'
    for p in [video_path, data_path, shuffles_path, results_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    # adds the video list to the config.yaml file
    video_sets = {}
    for video in videos:
        print(video)
        try:
           # For windows os.path.realpath does not work and does not link to the real video. [old: rel_video_path = os.path.realpath(video)]
           rel_video_path = str(Path.resolve(Path(video)))
        except:
           rel_video_path = os.readlink(str(video))

        vcap = cv2.VideoCapture(rel_video_path)
        if vcap.isOpened():
           width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           video_sets[rel_video_path] = {'crop': ', '.join(map(str, [0, width, 0, height]))}
        else:
           print("Cannot open the video file!")
           video_sets=None
           
    #        Set values to config file:
    cfg_file,ruamelFile = auxiliaryfunctions.create_config_template()
    cfg_file
    cfg_file['Task']=project
    cfg_file['scorer']=experimenter
    cfg_file['video_sets']=video_sets
    cfg_file['project_path']=str(project_path)
    cfg_file['date']=d
    cfg_file['bodyparts']=['Reach','Grasp','Ret_NP','Ret_WP','Hand','Pellet']
    cfg_file['cropping']=False
    cfg_file['start']=0
    cfg_file['stop']=1
    cfg_file['numframes2pick']=20
    cfg_file['TrainingFraction']=[0.95]
    cfg_file['iteration']=0
    cfg_file['resnet']=152
    cfg_file['snapshotindex']=-1
    cfg_file['x1']=0
    cfg_file['x2']=640
    cfg_file['y1']=277
    cfg_file['y2']=624
    cfg_file['batch_size']=15 #batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
    cfg_file['corner2move2']=(50,50)
    cfg_file['move2corner']=True
    cfg_file['pcutoff']=0.1
    cfg_file['dotsize']=8 #for plots size of dots
    cfg_file['alphavalue']=0.7 #for plots transparency of markers
    cfg_file['colormap']='jet' #for plots type of colormap

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile,cfg_file)

    print('Generated "{}"'.format(project_path / 'config.yaml'))
    print("\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)." %(project_name,str(wd)))
    return projconfigfile

def add_CLARA_videos(config,videos):
    """
    Add new videos to the config file at any stage of the project.

    """

    # Read the config file
    cfg = auxiliaryfunctions.read_config(config)
    video_path = Path(config).parents[0] / 'videos'
    data_path = Path(config).parents[0] / 'labeled-data'
    videos = [Path(vp) for vp in videos]
    dirs = [data_path/Path(i.stem) for i in videos]
    for p in dirs:
        """
        Creates directory under data & perhaps copies videos (to /video)
        """
        p.mkdir(parents = True, exist_ok = True)

    for idx,video in enumerate(videos):
        try:
           video_path = str(Path.resolve(Path(video)))
        except:
           video_path = os.readlink(video)

        vcap = cv2.VideoCapture(video_path)
        if vcap.isOpened():
            # get vcap property
           width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
           height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
           cfg['video_sets'].update({video_path : {'crop': ', '.join(map(str, [0, width, 0, height]))}})
        else:
           print("Cannot open the video file!")
    auxiliaryfunctions.write_config(config,cfg)

def create_training_dataset_CLARA(config,num_shuffles=1,Shuffles=None,windows2linux=False,trainIndexes=None,testIndexes=None):
    """
    Creates a training dataset. Labels from all the extracted frames are merged into a single .h5 file.\n
    Only the videos included in the config file are used to create this dataset.\n
    
    [OPTIONAL] Use the function 'add_new_video' at any stage of the project to add more videos to the project.

    Parameter
    ----------
    config : string
        Full path of the config.yaml file as a string.

    num_shuffles : int, optional
        Number of shuffles of training dataset to create, i.e. [1,2,3] for num_shuffles=3. Default is set to 1.

    Shuffles: list of shuffles.
        Alternatively the user can also give a list of shuffles (integers!).

    windows2linux: bool.
        The annotation files contain path formated according to your operating system. If you label on windows 
        but train & evaluate on a unix system (e.g. ubunt, colab, Mac) set this variable to True to convert the paths. 
    
    Example
    --------
    >>> deeplabcut.create_training_dataset('/analysis/project/reaching-task/config.yaml',num_shuffles=1)
    Windows:
    >>> deeplabcut.create_training_dataset('C:\\Users\\Ulf\\looming-task\\config.yaml',Shuffles=[3,17,5])
    --------
    """
    from skimage import io
    import scipy.io as sio
    
    # Loading metadata from config file:
    cfg = auxiliaryfunctions.read_config(config)
    scorer = cfg['scorer']
    project_path = cfg['project_path']
    # Create path for training sets & store data there
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg) #Path concatenation OS platform independent
    auxiliaryfunctions.attempttomakefolder(Path(os.path.join(project_path,str(trainingsetfolder))),recursive=True)
    
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).
    """
    AnnotationData=None
    data_path = Path(os.path.join(project_path , 'labeled-data'))
    videos = cfg['video_sets'].keys()
    video_names = [Path(i).stem for i in videos]
    for i in video_names:
        try:
            data = pd.read_hdf((str(data_path / Path(i))+'/CollectedData_'+cfg['scorer']+'.h5'),'df_with_missing')
            smlData = data.dropna(how='all')
            smlKeys = list(smlData.index.values)
            smlKeyLong = list()
            for sk in smlKeys:
                smlKeyLong.append('labeled-data/'+str(Path(i))+'/'+sk)
            smlData.index = smlKeyLong
            data = smlData
            if AnnotationData is None:
                AnnotationData=data
            else:
                AnnotationData=pd.concat([AnnotationData, data])

        except FileNotFoundError:
            print((str(data_path / Path(i))+'/CollectedData_'+cfg['scorer']+'.h5'), " not found (perhaps not annotated)")

    trainingsetfolder_full = Path(os.path.join(project_path,trainingsetfolder))
    filename=str(str(trainingsetfolder_full)+'/'+'/CollectedData_'+cfg['scorer'])
    AnnotationData.to_hdf(filename+'.h5', key='df_with_missing', mode='w')
    AnnotationData.to_csv(filename+'.csv') #human readable.
    Data = AnnotationData

    Data = Data[scorer] #extract labeled data

    #loading & linking pretrained models
    net_type ='resnet_'+str(cfg['resnet'])
    import deeplabcut
    parent_path = Path(os.path.dirname(deeplabcut.__file__))
    defaultconfigfile = str(parent_path / 'pose_cfg.yaml')
    
    model_path,num_shuffles=auxfun_models.Check4weights(net_type,parent_path,num_shuffles)
    
    if Shuffles==None:
        Shuffles=range(1,num_shuffles+1,1)
    else:
        Shuffles=[i for i in Shuffles if isinstance(i,int)]

    bodyparts = cfg['bodyparts']
    TrainingFraction = cfg['TrainingFraction']
    for shuffle in Shuffles: # Creating shuffles starting from 1
        for trainFraction in TrainingFraction:
            #trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)
            if trainIndexes is None and testIndexes is None:
                trainIndexes, testIndexes = SplitTrials_CLARA(range(len(Data.index)), trainFraction)
            else:
                print("You passed a split with the following fraction:", len(trainIndexes)*1./(len(testIndexes)+len(trainIndexes))*100)
            
            ####################################################
            # Generating data structure with labeled information & frame metadata (for deep cut)
            ####################################################

            # Make training file!
            data = []
            for jj in trainIndexes:
                H = {}
                # load image to get dimensions:
                filename = Data.index[jj]
                im = io.imread(os.path.join(cfg['project_path'],filename))
                H['image'] = filename

                if np.ndim(im)==3:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]])
                else:
                    # print "Grayscale!"
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                indexjoints=0
                joints=np.zeros((len(bodyparts),3))*np.nan
                for bpindex,bodypart in enumerate(bodyparts):
                    if Data[bodypart]['x'][jj]<np.shape(im)[1] and Data[bodypart]['y'][jj]<np.shape(im)[0]: #are labels in image?
                        joints[indexjoints,0]=int(bpindex)
                        joints[indexjoints,1]=Data[bodypart]['x'][jj]
                        joints[indexjoints,2]=Data[bodypart]['y'][jj]
                        indexjoints+=1

                joints = joints[np.where(
                    np.prod(np.isfinite(joints),
                            1))[0], :]  # drop NaN, i.e. lines for missing body parts

                assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0])
                        )  # y coordinate within image?
                assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1])
                        )  # x coordinate within image?

                H['joints'] = np.array(joints, dtype=int)
                if np.size(joints)>0: #exclude images without labels
                        data.append(H)

            if len(trainIndexes)>0:
                datafilename,metadatafilename=auxiliaryfunctions.GetDataandMetaDataFilenames(trainingsetfolder,trainFraction,shuffle,cfg)
                ################################################################################
                # Saving metadata (Pickle file)
                ################################################################################
                auxiliaryfunctions.SaveMetadata(os.path.join(project_path,metadatafilename),data, trainIndexes, testIndexes, trainFraction)
                ################################################################################
                # Saving data file (convert to training file for deeper cut (*.mat))
                ################################################################################

                DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
                MatlabData = np.array(
                    [(np.array([data[item]['image']], dtype='U'),
                      np.array([data[item]['size']]),
                      boxitintoacell_CLARA(data[item]['joints']))
                     for item in range(len(data))],
                    dtype=DTYPE)

                sio.savemat(os.path.join(project_path,datafilename), {'dataset': MatlabData})

                ################################################################################
                # Creating file structure for training &
                # Test files as well as pose_yaml files (containing training and testing information)
                #################################################################################

                modelfoldername=auxiliaryfunctions.GetModelFolder(trainFraction,shuffle,cfg)
                auxiliaryfunctions.attempttomakefolder(Path(config).parents[0] / modelfoldername,recursive=True)
                auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/'+ '/train')
                auxiliaryfunctions.attempttomakefolder(str(Path(config).parents[0] / modelfoldername)+ '/'+ '/test')

                path_train_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'train','pose_cfg.yaml'))
                path_test_config = str(os.path.join(cfg['project_path'],Path(modelfoldername),'test','pose_cfg.yaml'))
                #str(cfg['proj_path']+'/'+Path(modelfoldername) / 'test'  /  'pose_cfg.yaml')

                items2change = {
                    "dataset": datafilename,
                    "metadataset": metadatafilename,
                    "num_joints": len(bodyparts),
                    "all_joints": [[i] for i in range(len(bodyparts))],
                    "all_joints_names": [str(bpt) for bpt in bodyparts],
                    "init_weights": model_path,
                    "project_path": str(cfg['project_path']),
                    "net_type": net_type,
                    "crop": 'False'
                }
                trainingdata = MakeTrain_pose_yaml_CLARA(items2change,path_train_config,defaultconfigfile)
                keys2save = [
                    "dataset", "num_joints", "all_joints", "all_joints_names",
                    "net_type", 'init_weights', 'global_scale', 'location_refinement',
                    'locref_stdev'
                ]
                MakeTest_pose_yaml_CLARA(trainingdata, keys2save,path_test_config)
                
def MakeTest_pose_yaml_CLARA(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)

def MakeTrain_pose_yaml_CLARA(itemstochange,saveasconfigfile,defaultconfigfile):
    raw = open(defaultconfigfile).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc,Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasconfigfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]

def boxitintoacell_CLARA(joints):
    ''' Auxiliary function for creating matfile.'''
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer

def SplitTrials_CLARA(trialindex, trainFraction=0.8):
    ''' Split a trial index into train and test sets. Also checks that the trainFraction is a two digit number between 0 an 1. The reason
    is that the folders contain the trainfraction as int(100*trainFraction). '''
    if trainFraction>1 or trainFraction<0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])

    if abs(trainFraction-round(trainFraction,2))>0:
        print("The training fraction should be a two digit number between 0 and 1; i.e. 0.95. Please change accordingly.")
        return ([],[])
    else:
        trainsetsize = int(len(trialindex) * round(trainFraction,2))
        shuffle = np.random.permutation(trialindex)
        testIndexes = shuffle[trainsetsize:]
        trainIndexes = shuffle[:trainsetsize]
        
        return (trainIndexes, testIndexes)
    
def cam_config_template():
    """
    Creates a template for config.yaml file. This specific order is preserved while saving as yaml file.
    """
    yaml_str = """\
# Camera reference (enter serial numbers for each)
    frontCam:
    sideCam:
    topCam:
    masterCam:
    \n
# Camera settings
    frontCrop:
    sideCrop:
    topCrop:
    exposure:
    framerate:
    bin:
    \n
# DLC settings
    config_path:
    trainingsetindex:
    shuffle:
    \n
# User information
    unitRef:
    raw_data_dir:
    COM:
    default_video_dir:
    \n
    """
    ruamelFile = ruamel.yaml.YAML()
    cfg_file = ruamelFile.load(yaml_str)
    return cfg_file, ruamelFile

def read_config():
    """
    Reads structured config file

    """
    usrdatadir = os.path.dirname(os.path.realpath(__file__))
    _, user = os.path.split(Path.home())
    configname = os.path.join(usrdatadir, '%s_userdata.yaml' % user)
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = ruamelFile.load(f)
        except Exception as err:
            if err.args[2] == "could not determine a constructor for the tag '!!python/tuple'":
                with open(path, 'r') as ymlfile:
                  cfg = yaml.load(ymlfile,Loader=yaml.SafeLoader)
                  write_config(cfg)
    else:
        cfg,ruamelFile = cam_config_template()
        cfg['frontCrop']=[135, 180, 70, 180]
        cfg['sideCrop']=[180, 180, 70, 180]
        cfg['topCrop']=[0, 180, 70, 180]
        cfg['exposure']=1000
        cfg['framerate']=120
        cfg['bin']=4
        hostname = socket.gethostname()
        cfg['unitRef']='unit' + hostname[-2:]
        write_config(cfg)
    return(cfg)

def write_config(cfg):
    """
    Write structured config file.
    """
    usrdatadir = os.path.dirname(os.path.realpath(__file__))
    _, user = os.path.split(Path.home())
    configname = os.path.join(usrdatadir, '%s_userdata.yaml' % user)
    
    with open(configname, 'w') as cf:
        ruamelFile = ruamel.yaml.YAML()
        cfg_file,ruamelFile = cam_config_template()
        for key in cfg.keys():
            cfg_file[key]=cfg[key]
        
        ruamelFile.dump(cfg_file, cf)