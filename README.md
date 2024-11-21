# TranP-B-site
# Intro
TranP-B-site is a Transformer embedding based CNN model that predicts binding sites of protien protein interactions. It utilizes transformer embedding information extracted from protein sequences.

* # System Requirment 
```
* NVIDIA A100 80GB has been used for training and testing process
```  
* # Software and Database requirments  
To run TranP-B-site   
```
This model has been developed in Linux environment with: 
  * python 3.8.10
  * numpy 1.21.5
  * pandas 1.4.2
  * tensor flow 2.3.0
Run following commands in the terminal 
  * pip install torch transformers sentencepiece h5py  
  * pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```  
* # Run TranP-B-site for prediction
For prediction of binding site in a protein run following command:  
  *``` python prediction.py -s  'EDRLKIDVIDWLVFDPAQRAE' ```  
* # Google Colab TranP-B-site for prediction
Google Colab link  
*complete code
https://colab.research.google.com/drive/1hJa3IytWhD_Vi8MSmb_ICeMwPKHCn7qy#scrollTo=PRM2mkRFQd1M   
*feature extraction   
https://colab.research.google.com/drive/1bAcL_z5nfeN8Eobv9RUbYDnQB6nBGWVR?usp=sharing   
*training model   
https://colab.research.google.com/drive/1d9ZDRnNogZ2uoUnSHm4KaQJ7VXZHWF6i?usp=sharing
   
