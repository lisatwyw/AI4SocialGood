import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU'); 

print(gpus,'gpus?')

package_dir = '/kaggle/working/mypackages'
import sys; sys.path.append( package_dir )
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
        
    package_dir += '_gpu'
package_dir    


cmds= ['pip install pytorch-tabnet --target=/kaggle/working/mypackages']
cmds+=['pip install lifelines --target=/kaggle/working/mypackages']
cmds+=['pip install sentence-transformers --target=/kaggle/working/mypackages']

#cmds=['pip install scikit-survival --target=/kaggle/working/mypackages']
cmds+=['pip install git+https://github.com/sebp/scikit-survival.git --target=/kaggle/working/mypackages']
cmds+=['pip install hdbscan --no-deps --target=/kaggle/working/mypackages']

cmds=['pip install webdriver-manager selenium']

import subprocess, sys


try:
    import lifelines
except:
    print('installing pytorch-tabnet, etc...')
    if len(gpus)>0:        
        for cmd in cmds:
            cmd = cmd.split(' ') # essential that arguments be a list
            subprocess.run(cmd+'_gpu', shell=False)
    else:
        for cmd in cmds:
            cmd = cmd.split(' ')
            subprocess.run(cmd, shell=False)              

sys.path.append( package_dir )      

import lifelines
import time 

from selenium import webdriver 
from selenium.webdriver import Chrome 
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.common.by import By 
from webdriver_manager.chrome import ChromeDriverManager


url = 'https://www.doaj.org/search/journals?source=%7B%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22term%22%3A%7B%22bibjson.apc.has_apc%22%3Afalse%7D%7D%2C%7B%22term%22%3A%7B%22bibjson.other_charges.has_other_charges%22%3Afalse%7D%7D%5D%7D%7D%2C%22track_total_hits%22%3Atrue%7D'
    
    
