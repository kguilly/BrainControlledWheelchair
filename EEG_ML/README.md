# Brain Controlled Wheelchair Machine Learning Repo
### To activate environment: 
`$ conda env create -f conda_eegnet_env.yml`

If having trouble importing EEGNet, append the line: 
```
# EEGNet specific imports
import sys
sys.path.append('/path/to/repository/')
from EEG_ML.EEGModels import EEGNet
```
