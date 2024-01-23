#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 01:33:14 2023

@author: sharzil
"""
import argparse
import  pickle
import numpy as np
import random
from windowing import * 
import sys,os

import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D,LayerNormalization, Dropout, Flatten, Dense, concatenate ,AveragePooling1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, roc_auc_score, auc
import math
import tensorflow as tf
from tensorflow.keras import losses
from sklearn.utils import compute_class_weight
import random

import itertools
import pickle
np.random.seed(seed=21)





from transformers import T5EncoderModel, T5Tokenizer
import torch
import h5py
import time



def predict(prot_test_31,prot_test_p_31):
    gpu=0
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    mix2=0
    win_size=7
    emb_type_no=0
    folder_num=22
    
    f_name=f'{str(folder_num)}_{str(emb_type_no)}_{str(win_size)}__{str(mix2)}'
    new_model = tf.keras.models.load_model(f"model_7.h5")


    pred_y = new_model.predict([prot_test_31,prot_test_p_31])
    for i in range(len(pred_y)):
        if pred_y[i] < 0.5:
            pred_y[i] = 0;
        else:
            pred_y[i] = 1;
    pred_y=np.squeeze(pred_y)        
    pred_y=pred_y.astype('int')

    return(pred_y)




def embedding(input_seq):

    
    seq_path = "./protT5/example_seqs.fasta"
    

    per_residue = True 
    per_residue_path = "./protT5/output/per_residue_embeddings.h5" # where to store the embeddings

    per_protein = True
    per_protein_path = "./protT5/output/per_protein_embeddings.h5" # where to store the embeddings

    sec_struct = False
    sec_struct_path = "./protT5/output/ss3_preds.fasta" # file for storing predictions
    
    assert per_protein is True or per_residue is True or sec_struct is True, print(
        "Minimally, you need to active per_residue, per_protein or sec_struct. (or any combination)")
    
    
    #@title Import dependencies and check whether GPU is available. { display-mode: "form" }

    # device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    # print("Using {}".format(device))
    
    device = torch.device('cpu')
    
    def get_T5_model():
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        model = model.to(device) # move model to GPU
        model = model.eval() # set model to evaluation model
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    
        return model, tokenizer

    
    def get_embeddings( model, tokenizer, seqs, per_residue, per_protein, sec_struct, 
                       max_residues=4000, max_seq_len=1000, max_batch=100 ):
    
        if sec_struct:
          sec_struct_model = load_sec_struct_model()
    
        results = {"residue_embs" : dict(), 
                   "protein_embs" : dict(),
                   "sec_structs" : dict() 
                   }
    
        # sort sequences according to length (reduces unnecessary padding --> speeds up embedding)
        seq_dict   = sorted( seqs.items(), key=lambda kv: len( seqs[kv[0]] ), reverse=True )
        start = time.time()
        batch = list()
        for seq_idx, (pdb_id, seq) in enumerate(seq_dict,1):
            seq = seq
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            batch.append((pdb_id,seq,seq_len))
    
            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed 
            n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
            if len(batch) >= max_batch or n_res_batch>=max_residues or seq_idx==len(seq_dict) or seq_len>max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()
    
                # add_special_tokens adds extra token at the end of each sequence
                token_encoding = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
                input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
                embedding_repr = model(input_ids, attention_mask=attention_mask)
                #try:
                #    with torch.no_grad():
                        # returns: ( batch-size x max_seq_len_in_minibatch x embedding_dim )
                #        embedding_repr = model(input_ids, attention_mask=attention_mask)
                #except RuntimeError:
                #    print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                #    continue
    
                for batch_idx, identifier in enumerate(pdb_ids): # for each protein in the current mini-batch
                    s_len = seq_lens[batch_idx]
                    # slice off padding --> batch-size x seq_len x embedding_dim  
                    emb = embedding_repr.last_hidden_state[batch_idx,:s_len]
                    if per_residue: # store per-residue embeddings (Lx1024)
                        results["residue_embs"][ identifier ] = emb.detach().cpu().numpy().squeeze()
                    if per_protein: # apply average-pooling to derive per-protein embeddings (1024-d)
                        protein_emb = emb.mean(dim=0)
                        results["protein_embs"][identifier] = protein_emb.detach().cpu().numpy().squeeze()
    
    
        passed_time=time.time()-start
        print(passed_time)
        print(len(results["residue_embs"]))
        print(per_residue)
        print(len(results["protein_embs"]))
        
        
        
        
        avg_time = passed_time/len(results["residue_embs"]) if per_residue else passed_time/len(results["protein_embs"])
        print('\n############# EMBEDDING STATS #############')
        print('Total number of per-residue embeddings: {}'.format(len(results["residue_embs"])))
        print('Total number of per-protein embeddings: {}'.format(len(results["protein_embs"])))
        print("Time for generating embeddings: {:.1f}[m] ({:.3f}[s/protein])".format(
            passed_time/60, avg_time ))
        print('\n############# END #############')
        return results
    
    model, tokenizer = get_T5_model()
    # test=dict()
    input_seqence=dict()
    # test['test']='DVSGTVCLSALPPEATDTLNLIASDGPFPYSQDGVVFQNRESVLPTQSYGYYHEYTVITPGARTRGTRRIITGEATQEDYYTGDHYATFSLIDQTC'
    # test['test']='AQVQLVESGGGLVQAGGSLRLSCAVSGRPFSEYNLGWFRQAPGKEREFVARIRSSGTTVYTDSVKGRFSASRDNAKNMGYLQLNSLEPEDTAVYYCAMSRVDTDSPAFYDYWGQGTQVTVSTPR'
    input_seqence['input_seq']=input_seq
    # print(len(test['test']))
    
    results=get_embeddings( model, tokenizer, input_seqence,
                         per_residue, per_protein, sec_struct)
    
    ppi_embd=dict()
    ppi_embd_2=dict()
    for index1 in results['residue_embs'].keys():
        ppi_embd[index1]=results['residue_embs'][index1]
        
    for index1 in results['residue_embs'].keys():
        ppi_embd_2[index1]=results['protein_embs'][index1]  
    

    win_size=7
    
    seqs3=ppi_embd
    seqs6=ppi_embd_2
    
    
    
    seqs6=Protein_seq_feature(seqs3,seqs6)
    
    test_315_index=dict()
    for index in seqs3.keys():
        test_315_index[index]=0
    
    prot_d=seqs3
    prot_d_p=seqs6
    
    
    prot_test_3d=dict()
    prot_test_3d_p=dict()
    
    for index in prot_d.keys():
        prot_test_3d[index]=prot_d[index]
        prot_test_3d_p[index]=prot_d_p[index]
    
    
    prot_test_3=np.array(list(prot_test_3d.items()),dtype=object)[:,1]
    
    prot_test_3=windowing(prot_test_3,win_size)
    
    prot_test_p_3=np.array(list(prot_test_3d_p.items()),dtype=object)[:,1]
    
    prot_test_p_3=windowing(prot_test_p_3,1)
    
    return prot_test_3,prot_test_p_3
  
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--protein_sequence", type = str, help = "PDBID (e.g. EDRLKIDVIDWLVFDPAQRAE)")
    args = parser.parse_args()    
    if args.protein_sequence == None or len(args.protein_sequence) <= 10:
        print("Invalid protein sequence!")
    else:

        input_seq=args.protein_sequence
        [f1,f2]=embedding(input_seq)
        pred=predict(f1,f2)
        pred2=''
        for index in pred:
            pred2=pred2+str(index)
        print(input_seq)
        print(pred2)
