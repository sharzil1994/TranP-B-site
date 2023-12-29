#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:52:00 2023

@author: sharzil
"""
import numpy as np

def windowing(features,w_size):   
    to=0
    for index in features:
        b=np.shape(index)[0]
        to=to+b
    
        
    
    a=features[0]
    fea_len=np.shape(a)[1]
    finalout1=np.zeros([to,w_size,fea_len],'float')
    
    l=0
    for i in range(0,len(features)):
        temp_features=features[i]
        
        for j in range( 0, np.shape(temp_features)[0]):
            
            
            for k in range(0,w_size):
                
                k1=int(j+k-((w_size-1)/2))
                
                if k1<0 or k1 > np.shape(temp_features)[0]-1:
                    pass
                else:
                    finalout1[l,k,:]=temp_features[k1,:]
            l=l+1
    finalout1=finalout1.reshape((finalout1.shape[0], finalout1.shape[1],finalout1.shape[2], 1))
    return finalout1

def windowing_avg(features,w_size):   
    to=0
    for index in features:
        b=np.shape(index)[0]
        to=to+b
    
        
    
    a=features[0]
    fea_len=np.shape(a)[1]
    finalout1=np.zeros([to,fea_len],'float')
    
    l=0
    for i in range(0,len(features)):
        temp_features=features[i]
        
        for j in range( 0, np.shape(temp_features)[0]):
            
            div=0
            for k in range(0,w_size):
                
                k1=int(j+k-((w_size-1)/2))
                
                if k1<0 or k1 > np.shape(temp_features)[0]-1:
                    pass
                else:
                    finalout1[l,:]=finalout1[l,:]+temp_features[k1,:]
                    div=div+1
            
            finalout1[l,:]=finalout1[l,:]/div
            l=l+1         
    return finalout1



    
def label( labels):   
    final_label=labels
    
    to=0
    for i in range(0,len(final_label)):
        temp_label=final_label[i]
        for j in range(0,len(temp_label)):
            to=to+1
    
    
    finallabel=np.zeros([to],'int')
    
    l=0
    for i in range(0,len(final_label)):
        # print(i)

        temp_label=final_label[i]
        
        for j in range( 0, len(temp_label)):
            

            finallabel[l]=temp_label[j]
            l=l+1
            
    return  finallabel 

def Protein_seq_feature(seqs1,seqs4):
    
    for index in seqs1.keys():
        a=seqs1[index]
        b=np.shape(a)[0]
        c=seqs4[index]
        c=c.reshape((1,len(c)))
        d=seqs4[index]
        d=d.reshape((1,len(d)))
        for index_1 in range(b-1):
            d=np.concatenate((d,c),axis=0)
        seqs4[index]=d
    return seqs4
    
    
    