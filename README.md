# AI4SocialGood

1. Humanity issues
   - AI in human trafficking         /6
   - Web-scraping data on suicide    /6
   - ASL recognition                 /8 [[private]](https://colab.research.google.com/drive/1HZkHXPkgasQ7OJSyMHTsdlp33y6TYkpw?authuser=2#scrollTo=OToQM-BWQ9T2)
1. Climate research
   - AI in climate research          /6
   - Contrail segmentation           /7
1. Healthcare
   - Groundglass opacity severity rating in CT  /3
   - In-vitro cell research          /7
   - Fall prevention in seniors      /8-9 [[private]](https://docs.google.com/document/d/1dtgnINC1BMY-YDRbd82jQMkAmXDht9QX8xd9DuGvCVw/)
   - Age regression using meth-data  /8-9
   - Indigenous Peoples              /8
   - Climate changes & elders        /9

  

```
# Below works only in Linux OS
import subprocess
from ast import literal_eval

def run(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out, err = process.communicate()
    print(out.decode('utf-8').strip())

print('# CPU')
run('cat /proc/cpuinfo | egrep -m 1 "^model name"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu MHz"')
run('cat /proc/cpuinfo | egrep -m 1 "^cpu cores"')

# CPU
print('# RAM')
run('cat /proc/meminfo | egrep "^MemTotal"')

# RAM
print('# OS')
run('uname -a')

# OS
print('# GPU')
run('lspci | grep VGA')
```
