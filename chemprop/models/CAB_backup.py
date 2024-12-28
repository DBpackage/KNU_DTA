from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights

#class GlobalAttentionPooli(nn.Module):
#    def __init__(self, 


class ori_AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, 
                 hid_dim, 
                 n_heads,
                 batch_size,
                 dropout,
                 args : TrainArgs):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.batch_size = batch_size

        assert hid_dim % n_heads == 0

        self.W_q = nn.Linear(hid_dim, hid_dim)
        self.W_k = nn.Linear(hid_dim, hid_dim)
        self.W_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def split_heads(self, x):
        # print("CAB, Attention x size", x.size())
        x = x.view(self.batch_size, -1, self.n_heads, self.head_dim) # [Bs, seq_len, n, H/n]
        return x.permute(0, 2, 1, 3) # [Bs, n, seq_len, H/n]

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : SMILES Transformer output [args.batch_size, args.smiles_length, args.hidden_size]
        :Key   : SMILES D-MPNN output [args.batch_size, 1, args.hidden_size]
        :Value : SMILES D-MPNN output [args.batch_size, 1, args.hidden_size]
        args.batch_size    = bs
        args.smiles_length = Sz
        args.hidden_size   = H
        Cross-Att: Key and Value should always come from the same source (Aiming to forcus on), Query comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """
        #print("query size :",query.size())
        #print("key size : ",key.size())
        #print("value size :", value.size())

        Q = self.W_q(query) # [Bs, Sz, H]
        K = self.W_k(key)   # [Bs, 1, H]
        V = self.W_v(value) # [Bs, 1, H]
        #print("Before split heads")
        #print("Q size :",Q.size())
        #print("K size :",K.size())
        #print("V size :",V.size())

        Q = self.split_heads(Q)                     # [Bs, n, Sz, H/n]
        K_T = self.split_heads(K).permute(0,1,3,2)  # [Bs, n, H/n, 1]
        V = self.split_heads(V)                     # [Bs, n, 1, H/n]
        #print("After split heads ")
        #print("Q size   :",Q.size())
        #print("K_T size :",K_T.size())
        #print("V size   :",V.size())
        #input()


        """
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        print("Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3) size : ",Q.size())
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        print("K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3) size : ", K_T.size())
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        print("V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3) size : ", V.size())
        input()
        """

        energy = torch.matmul(Q, K_T) / self.scale # [Bs, n, Sz, 1]
        #print("energy size :", energy.size())

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1)) # [Bs, n, Sz, 1]
        #print("attention size:", attention.size())
        #input()

        weighter_matrix = torch.matmul(attention, V) # [Bs, n, Sz, H]
        #print("weighter matrix size1 : ",weighter_matrix.size())
        #input()

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous() # [Bs, Sz, n, H]
        #print("weighter matrix size2 : ",weighter_matrix.size())
        #input()

        weighter_matrix = weighter_matrix.view(self.batch_size, -1, self.hid_dim) # [Bs, Sz, H]
        #print("weighter matrix size3 : ",weighter_matrix.size())
        #input()

        weighter_matrix = self.do(self.fc(weighter_matrix)) # [Bs, Sz, H]
        #print("weighter matrix size4 : ",weighter_matrix.size())
        #input()

        return weighter_matrix

class ori_CrossAttentionBlock(nn.Module):
    """
        The main idea of Perceiver CPI (cross attention block + self attention block).
    """

    def __init__(self, args: TrainArgs):

        super(ori_CrossAttentionBlock, self).__init__()
        self.att = ori_AttentionBlock(hid_dim = args.hidden_size, n_heads = 1, batch_size=args.batch_size, dropout=args.dropout, args=args)
        #print("IN THE CAB ,args", args); input()
        #print("IN THE CAB prot, sps, smiles len", args.sequence_length, args.sps_length, args.smiles_length);input()
    
    def forward(self, graph_feature, trn_smiles, trn_prot_sps):
        """
            :graph_feature : [Bs, 1, 300]
            :trn_smiles    : [Bs, Sz, 300]
            :trn_prot_sps  : [Bs, Sy, 300]
        """

        #print("IN THE CAB")
        #print("graph_feature size : ",graph_feature.size())
        #print("trn_smiles size : ",trn_smiles.size())
        #print("trn_prot_sps : ",trn_prot_sps.size());input()
        # cross attention for compound information enrichment 
        # print("I'm coming into DRUG CROSS_ATTENTION BLOCK!!!")
        # self.att output [Bs, Sy, 300] 
        # graph_feature   [Bs, 1, 300]
        #print("FIRST GRAPH FEATURE STARTS!!!!!"); input()
        graph_feature = self.att(trn_smiles, graph_feature, graph_feature) # [Bs, Sy, H]
        #print("FIRST GRAPH FEATRUE ENDS!!!!!!!"); input()
        #print("graph_feature 1 ", graph_feature.size()); input() 
        
        # self-attention
        #print("SECOND GRAPH FEATURE STARTS!!!!!"); input()
        graph_feature = self.att(graph_feature,graph_feature,graph_feature) # [Bs, Sy, H]
        #print("SECOND GRAPH FEATRUE ENDS!!!!!!!"); input()

        #print("graph_feature 2: ", graph_feature.size()); input()
        # cross-attention for interaction
        #print("LAST GRAPH FEATRUE STARTS!!!!!!!"); input()
        output = self.att(graph_feature, trn_prot_sps, trn_prot_sps)
        #print("LAST GRAPH FEATRUE ENDS!!!!!!!"); input()
        #print("final output size :", output.size()); input()
        #print("FINAL OUTPUT SHAPE :", output.size())
        #input()

        return output



class perceiver_AttentionBlock(nn.Module):
    """ A class for attention mechanisn with QKV attention """
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        """ 
        :Query : A projection function
        :Key : A projection function
        :Value : A projection function
        Cross-Att: Query and Value should always come from the same source (Aiming to forcus on), Key comes from the other source
        Self-Att : Both three Query, Key, Value come form the same source (For refining purpose)
        """

        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)

        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2,3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class perceiver_CrossAttentionBlock(nn.Module):
    """
        The main idea of Perceiver CPI (cross attention block + self attention block).
    """

    def __init__(self, args: TrainArgs):

        super(perceiver_CrossAttentionBlock, self).__init__()
        self.att = perceiver_AttentionBlock(hid_dim = args.hidden_size, n_heads = 1, dropout=args.dropout)

    
    def forward(self,graph_feature,morgan_feature,sequence_feature):
        """
            :graph_feature : A batch of 1D tensor for representing the Graph information from compound
            :morgan_feature: A batch of 1D tensor for representing the ECFP information from compound
            :sequence_feature: A batch of 1D tensor for representing the information from protein sequence
        """
        # cross attention for compound information enrichment
        graph_feature = graph_feature + self.att(morgan_feature,graph_feature,graph_feature)
        # self-attention
        graph_feature = self.att(graph_feature,graph_feature,graph_feature)
        # cross-attention for interaction
        output = self.att(graph_feature, sequence_feature,sequence_feature)

        return output
