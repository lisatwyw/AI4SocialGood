'''
!wget -O persist_install.py https://raw.githubusercontent.com/lisatwyw/CS4SocialGood/main/common/persist_install.py
exec( open('persist_install.py','r').read() )
'''

'''
cmds= ['pip install pytorch-tabnet --target=/kaggle/working/mypackages']
cmds+=['pip install lifelines --target=/kaggle/working/mypackages']
cmds+=['pip install sentence-transformers --target=/kaggle/working/mypackages']
#cmds=['pip install scikit-survival --target=/kaggle/working/mypackages']
cmds+=['pip install git+https://github.com/sebp/scikit-survival.git --target=/kaggle/working/mypackages']
cmds+=['pip install hdbscan --no-deps --target=/kaggle/working/mypackages']
cmds=['pip install webdriver-manager selenium']
'''
import pandas as pd
import polars as pol
import numpy as np
import subprocess, sys, os
import tensorflow as tf
from time import time

gpus = tf.config.list_physical_devices('GPU');

def install_packages( cmds, INTERACTIVE=False ):
    if INTERACTIVE:
        package_dir = '/kaggle/working/mypackages' 
    else:
        package_dir = None 
    if package_dir is not None:
        sys.path.append( package_dir )    
    if gpus:    
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)          
    try:
        if package_dir is not None:
            sys.mkdir( package_dir )
    except:
        pass 
    
    for cmd in cmds:
        if package_dir is not None:
            cmd +=' --target=' + package_dir
        print( 'Running:\n\t', cmd ) 
        cmd = cmd.split(' ')
        subprocess.run(cmd, shell=False)         
        
    if package_dir is not None:
        sys.path.append( package_dir )      
    return gpus 

import torch, random
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    tf.set_random_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('random, torch, tf, os packages seeded') 
          
seed_everything(1119)
print(  '\n\n- torch, tf, os, sys, subprocess loaded \n- np, pol, pd, time, loaded')     
print( '\n\nngpus = install_packages() \n\nseed_everything(1119) ')
 
 
