'''
!wget -O k_persist_install.py https://raw.githubusercontent.com/lisatwyw/CS4SocialGood/main/common/k_persist_install.py
exec( open('k_persist_install.py','r').read() )
'''

import sys 
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU'); 

package_dir = '/kaggle/working/mypackages'

sys.path.append( package_dir )
if gpus:
    package_dir += '_gpu'
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
      
    
'''
cmds= ['pip install pytorch-tabnet --target=/kaggle/working/mypackages']
cmds+=['pip install lifelines --target=/kaggle/working/mypackages']
cmds+=['pip install sentence-transformers --target=/kaggle/working/mypackages']
#cmds=['pip install scikit-survival --target=/kaggle/working/mypackages']
cmds+=['pip install git+https://github.com/sebp/scikit-survival.git --target=/kaggle/working/mypackages']
cmds+=['pip install hdbscan --no-deps --target=/kaggle/working/mypackages']
cmds=['pip install webdriver-manager selenium']
'''


import subprocess, sys
def install( cmds ): 
    print( len(cmds) )
    
    for cmd in cmds:
        cmd +=' '+ package_dir
        print(cmd) 
        cmd = cmd.split(' ')
        subprocess.run(cmd, shell=False)              

    sys.path.append( package_dir )      
 
