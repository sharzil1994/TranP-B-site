# TranP-B-site
# Intro
TranP-B-site is a Transformer embedding based CNN model that predicts binding sites of protien protein interactions. It utilizes transformer embedding information extracted from protein sequences. 
# System Requirment
This model has been developed in Linux environment with:
* python 3.8.10
* numpy 1.21.5
* pandas 1.4.2
* tensor flow 2.3.0
* # Software and Database requirments
To run TranP-B-site  
Run following commands in the terminal  
pip install torch transformers sentencepiece h5py  
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  
# Run TranP-B-site for prediction
For prediction of binding site in a protein run following command:  
``` python prediction.py -s  'EDRLKIDVIDWLVFDPAQRAE'  
