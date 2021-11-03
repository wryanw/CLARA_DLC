#!/home/bioelectrics/anaconda3/envs/dlc/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 11:18:04 2019

@author: bioelectrics
"""

debugFileName = '/home/bioelectrics/Desktop/overnight_results.txt'
syntag = '/gvfs/smb-share:server=synology,share=whsynology'

import os, sys
import ruamel.yaml
from pathlib import Path
import glob
import shutil
from pathlib import PurePath
from datetime import timedelta, date
import cv2
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from skimage.util import img_as_ubyte
import os.path
import pprint
import logging
import pickle, yaml
from easydict import EasyDict as edict
import re
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import datetime

# =============================================================================
useFullFrame = False
# =============================================================================
overwriteOldData = True
# =============================================================================

flog = open(debugFileName, 'w')
flog.write('Overnight analysis is starting \t%s \n' % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
flog.close()

cfg = edict()
cfg.stride = 8.0
cfg.weigh_part_predictions = False
cfg.weigh_negatives = False
cfg.fg_fraction = 0.25
cfg.weigh_only_present_joints = False
cfg.mean_pixel = [123.68, 116.779, 103.939]
cfg.shuffle = True
cfg.snapshot_prefix = "./snapshot"
cfg.log_dir = "log"
cfg.global_scale = 1.0
cfg.location_refinement = False
cfg.locref_stdev = 7.2801
cfg.locref_loss_weight = 1.0
cfg.locref_huber_loss = True
cfg.optimizer = "sgd"
cfg.intermediate_supervision = False
cfg.intermediate_supervision_layer = 12
cfg.regularize = False
cfg.weight_decay = 0.0001
cfg.mirror = False

cfg.crop_pad = 0
cfg.scoremap_dir = "test"
cfg.dataset_type = "default"
cfg.use_gt_segm = False
cfg.batch_size = 1
cfg.video = False
cfg.video_batch = False

# Parameters for augmentation with regard to cropping
cfg.crop = False
cfg.cropratio= 0.25 #what is the fraction of training samples with cropping?
cfg.minsize= 100 #what is the minimal frames size for cropping plus/minus ie.. [-100,100]^2 for an arb. joint
cfg.leftwidth= 400
#limit width  [-leftwidth*u-100,100+u*rightwidth] x [-bottomwith*u-100,100+u*topwidth] where u is always a (different) random number in unit interval
cfg.rightwidth= 400
cfg.topheight= 400
cfg.bottomheight= 400

net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101,
             'resnet_152': resnet_v1.resnet_v1_152}


def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=2,
                                         scope='block4')
            return pred

class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg

    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg.net_type]

        mean = tf.constant(self.cfg.mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        # The next part of the code depends upon which tensorflow version you have.
        vers = tf.__version__
        vers = vers.split(".") #Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44
        if int(vers[0])==1 and int(vers[1])<4: #check if lower than version 1.4.
            with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16)
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=16,is_training=False)

        return net,end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg

        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with tf.variable_scope('pose', reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision:
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(inputs)
        prob = tf.sigmoid(heads['part_pred'])
        return {'part_prob': prob, 'locref': heads['locref']}

    
def pose_net(cfg):
    cls = PoseNet
    return cls(cfg)

def _setup_pose_prediction(cfg):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size   , None, None, 3])
    net_heads = pose_net(cfg).test(inputs)
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
        outputs.append(net_heads['locref'])

    restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    return sess, inputs, outputs
    
def extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network '''
    scmap = outputs_np[0]
    scmap = np.squeeze(scmap)
    locref = None
    if cfg.location_refinement:
        locref = np.squeeze(outputs_np[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref

def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return np.array(pose)

def getpose(image, cfg, sess, inputs, outputs, outall=False):
    ''' Extract pose '''
    im=np.expand_dims(image, axis=0).astype(float)
    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref = extract_cnn_output(outputs_np, cfg)
    pose = argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose

## Functions below implement are for batch sizes > 1:
def extract_cnn_outputmulti(outputs_np, cfg):
    ''' extract locref + scmap from network 
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart'''
    scmap = outputs_np[0]
    locref = None
    if cfg.location_refinement:
        locref =outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1],shape[2], -1, 2))
        locref *= cfg.locref_stdev
    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref


def getposeNP(image, cfg, sess, inputs, outputs, outall=False):
    ''' Adapted from DeeperCut, performs numpy-based faster inference on batches'''
    outputs_np = sess.run(outputs, feed_dict={inputs: image})
    
    scmap, locref = extract_cnn_outputmulti(outputs_np, cfg) #processes image batch.
    batchsize,ny,nx,num_joints = scmap.shape
    
    #Combine scoremat and offsets to the final pose.
    LOCREF=locref.reshape(batchsize,nx*ny,num_joints,2)
    MAXLOC=np.argmax(scmap.reshape(batchsize,nx*ny,num_joints),axis=1)
    Y,X=np.unravel_index(MAXLOC,dims=(ny,nx))
    DZ=np.zeros((batchsize,num_joints,3))
    for l in range(batchsize):
        for k in range(num_joints):
            DZ[l,k,:2]=LOCREF[l,MAXLOC[l,k],k,:]
            DZ[l,k,2]=scmap[l,Y[l,k],X[l,k],k]
            
    X=X.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,0]
    Y=Y.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,1]
    pose = np.empty((cfg['batch_size'], cfg['num_joints']*3), dtype=X.dtype) 
    pose[:,0::3] = X
    pose[:,1::3] = Y
    pose[:,2::3] = DZ[:,:,2] #P
    if outall:
        return scmap, locref, pose
    else:
        return pose

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return
    for k, v in a.items():
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def _load_dlc_config(filename):
    """Load a config from file filename and merge it into the default options.
    """
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f,Loader=yaml.SafeLoader))

    #Update the snapshot path to the corresponding path!
    trainpath=str(filename).split('pose_cfg.yaml')[0]
    yaml_cfg['snapshot_prefix']=trainpath+'snapshot'
    #the default is: "./snapshot"
    _merge_a_into_b(yaml_cfg, cfg)

    logging.info("Config:\n"+pprint.pformat(cfg))
    return cfg

def _GetScorerName(cfg,shuffle,trainFraction,trainingsiterations='unknown'):
    ''' Extract the scorer/network name for a particular shuffle, training fraction, etc. '''
    Task = cfg['Task']
    date = cfg['date']
    if trainingsiterations=='unknown':
        snapshotindex=cfg['snapshotindex']
        if cfg['snapshotindex'] == 'all':
            print("Changing snapshotindext to the last one -- plotting, videomaking, etc. should not be performed for all indices. For more selectivity enter the ordinal number of the snapshot you want (ie. 4 for the fifth) in the config file.")
            snapshotindex = -1
        else:
            snapshotindex=cfg['snapshotindex']

        modelfolder=os.path.join(cfg["project_path"],str(_GetModelFolder(trainFraction,shuffle,cfg)),'train')
        Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(modelfolder) if "index" in fn])
        increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
        Snapshots = Snapshots[increasing_indices]
        #dlc_cfg = read_config(os.path.join(modelfolder,'pose_cfg.yaml'))
        #dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
        SNP=Snapshots[snapshotindex]
        trainingsiterations = (SNP.split(os.sep)[-1]).split('-')[-1]

    scorer = 'DeepCut' + "_resnet" + str(cfg['resnet']) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)
    return scorer

def _GetModelFolder(trainFraction,shuffle,cfg):
    Task = cfg['Task']
    date = cfg['date']
    iterate = 'iteration-'+str(cfg['iteration'])
    return Path('dlc-models/'+ iterate+'/'+Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle))

def _Getlistofvideos(videos,videotype):
    from random import sample
    #checks if input is a directory
    if [os.path.isdir(i) for i in videos] == [True]:#os.path.isdir(video)==True:
        """
        Analyzes all the videos in the directory.
        """
        
        print("Analyzing all the videos in the directory")
        videofolder= videos[0]
        os.chdir(videofolder)
        videolist=[fn for fn in os.listdir(os.curdir) if (videotype in fn) and ('labeled.mp4' not in fn)] #exclude labeled-videos!
        Videos = sample(videolist,len(videolist)) # this is useful so multiple nets can be used to analzye simultanously
    else:
        if isinstance(videos,str):
            if os.path.isfile(videos): # #or just one direct path!
                Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
            else:
                Videos=[]
        else:
            Videos=[v for v in videos if os.path.isfile(v) and ('labeled.mp4' not in v)]
    return Videos

def GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,batchsize):
    ''' Batchwise prediction of pose '''
    
    PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0 # keeps track of which image within a batch should be written to
    batch_num = 0 # keeps track of which batch you are at
    ny,nx=int(cap.get(4)),int(cap.get(3))
    if cfg['cropping']:
        print("Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." %(cfg['x1'], cfg['x2'],cfg['y1'], cfg['y2']))
        nx=cfg['x2']-cfg['x1']
        ny=cfg['y2']-cfg['y1']
        if nx>0 and ny>0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1']>=0 and cfg['x2']<int(cap.get(3)+1) and cfg['y1']>=0 and cfg['y2']<int(cap.get(4)+1):
            pass #good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')
            
    frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte') # this keeps all frames in a batch
    pbar=tqdm(total=nframes)
    counter=0
    step=max(10,int(nframes/100))
    while(cap.isOpened()):
            if counter%step==0:
                pbar.update(step)
            ret, frame = cap.read()
            if ret:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if cfg['cropping']:
                    frames[batch_ind] = img_as_ubyte(frame[cfg['y1']:cfg['y2'],cfg['x1']:cfg['x2']])
                else:
                    frames[batch_ind] = img_as_ubyte(frame)
                    
                if batch_ind==batchsize-1:
                    pose = getposeNP(frames,dlc_cfg, sess, inputs, outputs)
                    PredicteData[batch_num*batchsize:(batch_num+1)*batchsize, :] = pose
                    batch_ind = 0
                    batch_num += 1
                else:
                   batch_ind+=1
            else:
                nframes = counter
                print("Detected frames: ", nframes)
                if batch_ind>0:
                    pose = getposeNP(frames, dlc_cfg, sess, inputs, outputs) #process the whole batch (some frames might be from previous batch!)
                    PredicteData[batch_num*batchsize:batch_num*batchsize+batch_ind, :] = pose[:batch_ind,:]
                break
            counter+=1

    pbar.close()
    return PredicteData,nframes

def _AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,sess,inputs, outputs,pdindex,save_as_csv, destfolder=None):
    ''' Helper function for analyzing a video '''
    
    vname = Path(video).stem
    if destfolder is None:
        destfolder = str(Path(video).parents[0])
    dataname = os.path.join(destfolder,vname + DLCscorer + '.h5')
    try:
        # Attempt to load data...
        pd.read_hdf(dataname)
        if not overwriteOldData:
            return
    except:
        pass
    
    flog = open(debugFileName, 'a')
    flog.write('%s\t%s\n' % (vname, datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    flog.close()
    cap=cv2.VideoCapture(video)
    
    fps = cap.get(5) #https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    nframes = int(cap.get(7))
    size=(int(cap.get(4)),int(cap.get(3)))
    
    ny,nx=size
    start = time.time()

    if int(dlc_cfg["batch_size"])>1:
        PredicteData,nframes=GetPoseF(cfg,dlc_cfg, sess, inputs, outputs,cap,nframes,int(dlc_cfg["batch_size"]))
    else:
        flog = open(debugFileName, 'a')
        flog.write('Batch size must be greater than 1 \n')
        flog.close()
        return

    stop = time.time()
    
    if cfg['cropping']==True:
        coords=[cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']]
    else:
        coords=[0, nx, 0, ny] 
        
    dictionary = {
        "start": start,
        "stop": stop,
        "run_duration": stop - start,
        "Scorer": DLCscorer,
        "DLC-model-config file": dlc_cfg,
        "fps": fps,
        "batch_size": dlc_cfg["batch_size"],
        "frame_dimensions": (ny, nx),
        "nframes": nframes,
        "iteration (active-learning)": cfg["iteration"],
        "training set fraction": trainFraction,
        "cropping": cfg['cropping'],
        "cropping_parameters": coords
    }
    metadata = {'data': dictionary}

    print("Saving results in %s..." %(Path(video).parents[0]))
    _SaveData(PredicteData[:nframes,:], metadata, dataname, pdindex, range(nframes),save_as_csv)

def _SaveData(PredicteData, metadata, dataname, pdindex, imagenames,save_as_csv):
    ''' Save predicted data as h5 file and metadata as pickle file; created by predict_videos.py '''
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([dataname, DataMachine], f)
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    if save_as_csv:
        print("Saving csv poses!")
        DataMachine.to_csv(dataname.split('.h5')[0]+'.csv')
    with open(dataname.split('.h5')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def read_config_dlc(configname):
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    try:
        with open(path, 'r') as f:
            cfg = ruamelFile.load(f)
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s \t%s \n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.write('\n')
    return(cfg)
        
def read_config():
    usrdatadir = os.path.dirname(os.path.realpath(__file__))
    _, user = os.path.split(Path.home())
    configname = os.path.join(usrdatadir, '%s_userdata.yaml' % user)
    ruamelFile = ruamel.yaml.YAML()
    path = Path(configname)
    try:
        with open(path, 'r') as f:
            cfg = ruamelFile.load(f)
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.close()
    return(cfg)

def _analyze_videos(config,videos,videotype='avi',shuffle=1,trainingsetindex=0,gputouse=None,save_as_csv=False, destfolder=None,cropping=None):
    flog = open(debugFileName, 'a')
    flog.write('Starting GPU \t%s \n' % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.close()
    
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE'] #was potentially set during training
    
    if gputouse is not None: #gpu selection
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
            
    tf.reset_default_graph()
    start_path=os.getcwd() #record cwd to return to this directory in the end
    flog = open(debugFileName, 'a')
    flog.write('GPU initialized \t%s \n' % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.close()

    cfg = read_config_dlc(config)
    
    
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    
    modelfolder=os.path.join(cfg["project_path"],str(_GetModelFolder(trainFraction,shuffle,cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    print(str(path_test_config))
    try:
        dlc_cfg = _load_dlc_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError("It seems the model for shuffle %s and trainFraction %s does not exist."%(shuffle,trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
      Snapshots = np.array([fn.split('.')[0]for fn in os.listdir(os.path.join(modelfolder , 'train'))if "index" in fn])
    except FileNotFoundError:
      raise FileNotFoundError("Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."%(shuffle,shuffle))

    snapshotindex=cfg['snapshotindex']
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    
    dlc_cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]
    
    #update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size']=cfg['batch_size']
    # Name for scorer:
    DLCscorer = _GetScorerName(cfg,shuffle,trainFraction,trainingsiterations=trainingsiterations)
    
    sess, inputs, outputs = _setup_pose_prediction(dlc_cfg)
    pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],names=['scorer', 'bodyparts', 'coords'])
    ##################################################
    # Datafolder
    ##################################################
    Videos=_Getlistofvideos(videos,videotype)
    if len(Videos)>0:
        #looping over videos
        for n, video in enumerate(Videos):
            if cropping is not None:
                cfg['cropping']=True
                cfg['x1'],cfg['x2'],cfg['y1'],cfg['y2']=cropping[n]
            else:
                cfg['cropping']=False
            _AnalyzeVideo(video,DLCscorer,trainFraction,cfg,dlc_cfg,sess,inputs, outputs,pdindex, save_as_csv, destfolder)
    os.chdir(str(start_path))


def read_metadata(path):
    ruamelFile = ruamel.yaml.YAML()
    if os.path.exists(path):
        with open(path, 'r') as f:
            cfg = ruamelFile.load(f)
    return(cfg)

def track_all():
    dirlist = list()
    destlist = list()
    user_cfg = read_config()
    write_dir = user_cfg['compressed_video_dir']
    read_dir = write_dir
    prev_date_list = [name for name in os.listdir(read_dir)]
    for f in prev_date_list:
        unit_dirR = os.path.join(read_dir, f, user_cfg['unitRef'])
        unit_dirW = os.path.join(write_dir, f, user_cfg['unitRef'])
        if os.path.exists(unit_dirR):
            prev_expt_list = [name for name in os.listdir(unit_dirR)]
            for s in prev_expt_list:
                dirlist.append(os.path.join(unit_dirR, s))
                destlist.append(os.path.join(unit_dirW, s))
    fullVidList = list()
    cropList = list()
    for ndx, s in enumerate(dirlist):
        avi_list = os.path.join(s, '*.mp4')
        vid_list = glob.glob(avi_list)
        if len(vid_list):
            vidpathpts = os.path.split(vid_list[0])
            namepts = vidpathpts[1].split('_')
            metaname = namepts[0]+'_'+namepts[1]+'_'+namepts[2]+'_metadata.yaml'
            metapath = os.path.join(vidpathpts[0],metaname)
            metadata = read_metadata(metapath)
        for v in vid_list:
            vpt = v.split('Cam')[0]
            cpt = metadata[vpt.split('_')[-1]+'Crop']
            if 'unit00' in v and useFullFrame:
                config_path = '/home/bioelectrics/Documents/FullFrame_CLARA-WRW-2020-03-06/config.yaml'
                cropList = None
            else:
                config_path = user_cfg['config_path']
                crp = [cpt[0],cpt[0]+cpt[1],cpt[2],cpt[2]+cpt[3]]
                if crp[3] > 268:
                    crp[3] = 268
                if crp[1] > 360:
                    crp[1] = 360
                cropList.append(crp)
            
            fullVidList.append(v)
            
    try:
        _analyze_videos(config_path,fullVidList, videotype='.mp4', save_as_csv=False, cropping=cropList)
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.close()
                
def copy2server():
    prev_date_list = list()
    start_dt = date(2019, 9, 11)
    end_dt = date.today()
    for n in range(int ((end_dt - start_dt).days)+1):
        dt = start_dt+ timedelta(n)
        prev_date_list.append(dt.strftime("%Y%m%d"))
        
    
    # =============================================================================
    overwrite = False
    # =============================================================================
    
    server_dir = '/run/user/10*'
    mounted_serv = glob.glob(server_dir)
    if not len(mounted_serv):
        flog = open(debugFileName, 'a')
        flog.write('No server found')
        flog.close()
        return
    
    synology = PurePath(mounted_serv[0] + syntag + '/BIOElectricsLab/RAW_DATA/AutomatedBehavior')
    if not os.path.isdir(synology):
        flog = open(debugFileName, 'a')
        flog.write('Synology not found')
        flog.close()
        return
    user_cfg = read_config()
    start_date = '20200201'
    local_dir = user_cfg['compressed_video_dir']
    #synology = 'media/nvme2/unit00_AnalyzedData'
    #synology = '/home/bioelectrics/Documents/unit00_AnalyzedData'
    
    read_list = list()
    write_list = list()
    try:
        for f in prev_date_list:
            if f < start_date:
                continue
            date_dir = PurePath(os.path.join(local_dir, f))
            if not os.path.isdir(date_dir):
                continue
            unit_list = [name for name in os.listdir(date_dir)]
            for u in unit_list:
                if not u == user_cfg['unitRef']:
                    continue
                unit_dir = PurePath(os.path.join(local_dir, f, u))
                sess_list = [name for name in os.listdir(unit_dir)]
                for s in sess_list:
                    sess_dir = os.path.join(unit_dir,s)
                    read_list.append(sess_dir)
                    write_list.append(os.path.join(synology,f,u,s))
                    
        for ndx, r in enumerate(read_list):
            dest_dir = write_list[ndx]
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            all_files = glob.glob(os.path.join(r, '*'))    
            for f in all_files:
                mname = PurePath(f).name
                if '.avi' in mname:
                    continue
                mdest = os.path.join(dest_dir,mname)
                
                if overwrite:
                    shutil.copyfile(f,mdest)
                if not os.path.isfile(mdest):
                    shutil.copyfile(f,mdest)
        #            print('copyingA\n')
                try:
                    szA = os.path.getsize(mdest)
                    szB = os.path.getsize(f)
                    if not szA == szB:
                        shutil.copyfile(f,mdest)
                except Exception as ex:
                    flog = open(debugFileName, 'a')
                    flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
                    flog.close()
        
        #            shutil.copyfile(f,mdest)
            vid_list = glob.glob(os.path.join(r, '*.mp4'))
            for v in vid_list:
                vid_name = PurePath(v)
                dest_path = os.path.join(dest_dir, vid_name.stem+'.mp4')
                vid = cv2.VideoCapture(dest_path)
                numberFramesA = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            if '.avi' in mname:
                continue
        #    break
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.close()
    
if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath('/home/bioelectrics/anaconda3/envs/dlc/lib/python3.6/site-packages/deeplabcut/CLARA_DLC'))
    sys.path.insert(0, os.path.abspath('/home/bioelectrics/anaconda3/envs/dlc/lib/python3.6/site-packages/deeplabcut'))
    flog = open(debugFileName, 'a')
    flog.write("Tracking initiated\t%s\n" % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.close()
    
    try:
        server_dir = '/run/user/10*'
        mounted_serv = glob.glob(server_dir)
        tracking_complete_txt = PurePath(mounted_serv[0] + syntag + '/BIOElectricsLab/ANALYZED_DATA/TrackingStatus/unit00.txt')
        f = open(tracking_complete_txt, 'w')
        f.write("in progress\r\n")
        f.close()
        track_all()
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.close()
            
    flog = open(debugFileName, 'a')
    flog.write("Tracking complete\t%s\n" % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.write("Copying files to server\t%s\n" % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.close()
    
    try:
        copy2server()
        server_dir = '/run/user/10*'
        mounted_serv = glob.glob(server_dir)
        tracking_complete_txt = PurePath(mounted_serv[0] + syntag + '/BIOElectricsLab/ANALYZED_DATA/TrackingStatus/unit00.txt')
        f = open(tracking_complete_txt, 'w')
        f.write("complete\r\n")
        f.close()
    except Exception as ex:
        flog = open(debugFileName, 'a')
        flog.write("%s\t%s\n" % (str(ex), datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
        flog.close()
    
    flog = open(debugFileName, 'a')
    flog.write("Copying complete\t%s\n" % datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    flog.write("Overnight analysis complete!!")
    flog.close()
    
    
