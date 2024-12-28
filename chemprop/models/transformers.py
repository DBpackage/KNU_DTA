import json
from sklearn.preprocessing import OneHotEncoder

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
import sys

import pandas as pd
import numpy as np
import math
import collections
import copy

from time import time
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from sklearn.metrics import mean_squared_error, log_loss, r2_score
from scipy.stats import pearsonr
import statistics
from typing import List, Union, Tuple
from chemprop.args import TrainArgs
from chemprop.data.utils import get_vocabs

import warnings
warnings.filterwarnings(action='ignore')

#args = TirainArgs
#words2idx_p, words2idx_s, words2idx_d, _, _ = get_vocabs(TrainArgs)

#print("len p dict :",len(words2idx_p))
#print("len s dict :",len(words2idx_s))
#print("len d dict :",len(words2idx_d))
#input()

class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_model : int,
                 max_len : int = 2000,
                 dropout : float = 0.1):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p = dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x): # x.size [batch size, sequence_len]

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Protein_SPS_Transformer(nn.Module):

    def __init__(self,
                 args : TrainArgs):
        super(Protein_SPS_Transformer, self).__init__()
        
        words2idx_p, words2idx_s, words2idx_d, _, _ = get_vocabs(args)

        #print("IN THE TRANSFORMER args : ",args); input()
        #print("args.smiles_length :", args.smiles_length)
        #print("args.sps_legnth :",args.sps_length);input()
        # 1. Initialization
        self.protein_vocab_size = len(words2idx_p) # 31
        self.sps_vocab_size     = len(words2idx_s) # 77?
        self.d_model            = args.transformer_d_model
        self.d_ff               = args.transformer_d_ff
        self.num_heads          = args.transformer_nheads
        self.nlayers            = args.transformer_nlayers
        
        self.pe = PositionalEncoding(d_model = self.d_model, dropout=args.dropout)
        
        self.protein_embedding = nn.Embedding(self.protein_vocab_size, self.d_model)
        self.sps_embedding = nn.Embedding(self.sps_vocab_size, self.d_model)

        self.input_dim  = self.d_model
        self.output_dim = args.hidden_size
        
        self.lin = nn.Linear(self.input_dim, self.output_dim)

        self.LN = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(args.dropout)

        self.bs = args.batch_size

        # 2. Make nn.Transformer
        self.transformer = nn.Transformer(d_model = self.d_model,
                                          nhead = self.num_heads,
                                          num_encoder_layers = self.nlayers,
                                          num_decoder_layers = self.nlayers,
                                          dim_feedforward = self.d_ff,
                                          batch_first = True)
        
        # 3. Make transformer Encoder block
        self.transformer_encoderlayer = nn.TransformerEncoderLayer(d_model = self.d_model,
                                                                   nhead = self.num_heads,
                                                                   dim_feedforward = self.d_ff,
                                                                   batch_first = True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.transformer_encoderlayer,
                                                                 num_layers = self.nlayers,
                                                                enable_nested_tensor=False) # 이거바꿈
        
        # 4. Make transformer Decoder block
        self.transformer_decoderlayer = torch.nn.TransformerDecoderLayer(d_model = self.d_model,
                                                                         nhead = self.num_heads,
                                                                         dim_feedforward = self.d_ff,
                                                                         batch_first = True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer = self.transformer_decoderlayer,
                                                         num_layers = self.nlayers)
        
        # 5. Make Fc layer for decoder output to string
        # self.lin = nn.Linear(self.d_model, self.sps_vocab_size)
        # self.lin = nn.Linear(self.d_model, self.concat_dim)
        #print("self.transformer_encoder:",self.transformer_encoder)
        #input()
    
    def Encoding(self, src, pad_id = 0):
        """
        args
        src : here src means protein, src shape is [Bs, seq_len_protein(Sx)]
        pad_id : padding token index
        """
        #torch.set_printoptions(threshold=sys.maxsize)

        # 1. Embedding
        x = self.protein_embedding(src) * math.sqrt(self.d_model) # x shape [Bs, Sx, d_model]
        #print("Encoder After embedding x :",x.size()); input()

        # 2. Positional Encoding
        x = self.pe(x) # x shape [Bs, Sx, d_model]
        #print("Encoder After Positional Encoding x :", x.size()); input()
        
        # 3. padding mask for encoder
        src_padding_mask = (src == pad_id)
        #print("Encoder src_padding_mask.size :",src_padding_mask.size())
        #print("Encoder src_padding_amks :",src_padding_mask)
        #input()
        
        # 4. forward Encoder
        memory = self.transformer_encoder(x, 
                                          src_key_padding_mask = src_padding_mask)
        #print("Encoder After Encoder x :", memory,); input()
        
        #print("PROTEIN ENCODER memory size : ",memory.size())
        #input()
        #torch.set_printoptions(threshold=1000)
        return memory, src_padding_mask # [Bs, Sx, d_model] [Sx, Sx]

    def Decoding(self, tgt, memory, memory_mask, pad_id=0): 
        """
        args
        tgt : here tgt means SPS, tgt shape is [Bs, seq_len_sps(Sy)]
        memory : flows from Encoder shape is [Bs, Sx, d_model]
        memory_mask : same mask with Encoding layer src_mask shape is [Bs, Sx] ? 확인필요
        pad_id : padding token index
        """
        # 1. Embedding
        x = self.sps_embedding(tgt) * math.sqrt(self.d_model) # x shape : [Bs, Sy, d_model]

        # 2. Positional Encoding
        x = self.pe(x)
        
        # 3. MAKE MASK FOR "Masked Multi-head attention" (only in decoder)
        tgt_mask = self.transformer.generate_square_subsequent_mask(sz = x.size(1)) # [Sy, Sy]
        tgt_mask = tgt_mask.to(x.device)
        #print("SPS DECODER tgt_mask.size() : ",tgt_mask.size())
        #input()
        
        # 4. Make tgt padding mask
        tgt_padding_mask = (tgt == pad_id)
        #print("SPS DECODER tgt_padding_mask size : ",tgt_padding_mask.size())
        #input()

        # 5. forward decoder
        out = self.transformer_decoder(x, 
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        tgt_key_padding_mask = tgt_padding_mask,
                                        memory_key_padding_mask = memory_mask)
        """
        tgt_mask (Optional[Tensor]) – the mask for the tgt sequence (optional).
        memory_mask (Optional[Tensor]) – the mask for the memory sequence (optional).
        tgt_key_padding_mask (Optional[Tensor]) – the mask for the tgt keys per batch (optional).
        """
        
        #print("SPS Decoder out size : ", out.size()) # out size :  torch.Size([4, 206, 128])
        #input()

        return out # shape [Bs, Sy, d_model]

    def forward(self, src, tgt, pad_id=0):
        #print("PS_transformer src :", src)
        #input()
        #print("PS_transformer tgt :", tgt)
        #input()

        memory, mask = self.Encoding(src, pad_id) # memory [bs, Sx, d_model] mask [Sx, Sx]
        #print("protein sps Encoder memory",memory)
        #input()
        # print("protein Encoder mask.size", mask.size())
        # input()

        out = self.Decoding(tgt, memory, mask, pad_id) # output shape [Bs, Sy, d_model]
        #print("protein sps Decoding output",out)
        #input()
        #out = out.view(self.bs, -1) # [Bs, Sy * d_model]

        out = F.relu(self.dropout(self.lin(out))) # [Bs, Sy, args.hidden_size]
        #print("protein sps after FC out", out)
        #input()
        out = self.LN(out)
        #print("protein sps after LN out", out)
        #input()
        #print("Protein_SPS transformer Final output shape :", out.size())
        #input()
        return out # [Bs, Sy, args.hidden_size]

class Drug_Encoder(nn.Module):

    def __init__(self, args : TrainArgs):
        super(Drug_Encoder, self).__init__()
        
        words2idx_p, words2idx_s, words2idx_d, _, _ = get_vocabs(args)

        # 1. Initialization
        self.drug_vocab_size = len(words2idx_d)

        self.d_model    = args.transformer_d_model
        self.d_ff       = args.transformer_d_ff
        self.nlayers    = args.transformer_nlayers
        self.num_heads  = args.transformer_nheads

        self.pe = PositionalEncoding(d_model = self.d_model)

        self.drug_embedding = nn.Embedding(self.drug_vocab_size, self.d_model)

        self.input_dim  = self.d_model
        self.output_dim = args.hidden_size
        self.lin = nn.Linear(self.input_dim, self.output_dim)

        self.LN = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(args.dropout)

        self.bs = args.batch_size

        # 2. Transformer Encoder
        self.transformer_encoderlayer = nn.TransformerEncoderLayer(d_model = self.d_model,
                                                                   nhead = self.num_heads,
                                                                   dim_feedforward = self.d_ff,
                                                                   batch_first = True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.transformer_encoderlayer,
                                                         num_layers = self.nlayers,
                                                         enable_nested_tensor=False)

    def Encoding(self, src, pad_id = 0):
        """
        args
        src : here src means protein after DeepConV, src shape is [Bs, Sz, d_model]
        so, you don't need to do embedding for protein in this structure.
        pad_id : padding token index
        """
        #print("DRUG Encoder before embedding: ", src.size(), src)
        # 1. Embedding
        x = self.drug_embedding(src) * math.sqrt(self.d_model)
        #print("Encoder after embedding: ",x.size(), x)

        # 2. Positional Encoding
        x = self.pe(x) # x shape [Bs, Sz, d_model]

        src_padding_mask = (src == pad_id)  # [Sz, Sz]
        #print("drug encoder x ",x.size(), x)
        
        # 4. Forward Encoder with padding mask
        memory = self.transformer_encoder(x, 
                                          src_key_padding_mask=src_padding_mask)

        return memory  # memory shape [Bs, Sz, d_model]

    def forward(self, src, pad_id = 0):
        memory = self.Encoding(src, pad_id)

        memory = F.relu(self.dropout(self.lin(memory))) # [Bs, Sz, args.hidden_size]
        memory = self.LN(memory)

        return memory # [Bs, Sz, args.hidden_size]


class Protein_Encoder(nn.Module):

    def __init__(self, args : TrainArgs):
        super(Protein_Encoder, self).__init__()

        words2idx_p, words2idx_s, words2idx_d, _, _ = get_vocabs(args)

        # 1. Initialization
        self.prot_vocab_size = len(words2idx_p)

        self.d_model    = args.transformer_d_model
        self.d_ff       = args.transformer_d_ff
        self.nlayers    = args.transformer_nlayers
        self.num_heads  = args.transformer_nheads

        self.pe = PositionalEncoding(d_model = self.d_model)

        self.prot_embedding = nn.Embedding(self.prot_vocab_size, self.d_model)

        self.input_dim  = self.d_model
        self.output_dim = args.hidden_size
        self.lin = nn.Linear(self.input_dim, self.output_dim)

        self.LN = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(args.dropout)

        self.bs = args.batch_size

        # 2. Transformer Encoder
        self.transformer_encoderlayer = nn.TransformerEncoderLayer(d_model = self.d_model,
                                                                   nhead = self.num_heads,
                                                                   dim_feedforward = self.d_ff,
                                                                   batch_first = True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = self.transformer_encoderlayer,
                                                         num_layers = self.nlayers,
                                                         enable_nested_tensor=False)

    def Encoding(self, src, pad_id = 0):
        """
        args
        src : here src means protein after DeepConV, src shape is [Bs, Sz, d_model]
        so, you don't need to do embedding for protein in this structure.
        pad_id : padding token index
        """
        #print("DRUG Encoder before embedding: ", src.size(), src)
        # 1. Embedding
        x = self.prot_embedding(src) * math.sqrt(self.d_model)
        #print("Encoder after embedding: ",x.size(), x)

        # 2. Positional Encoding
        x = self.pe(x) # x shape [Bs, Sz, d_model]

        src_padding_mask = (src == pad_id)  # [Sz, Sz]
        #print("drug encoder x ",x.size(), x)

        # 4. Forward Encoder with padding mask
        memory = self.transformer_encoder(x,
                                          src_key_padding_mask=src_padding_mask)

        return memory  # memory shape [Bs, Sz, d_model]

    def forward(self, src, pad_id = 0):
        memory = self.Encoding(src, pad_id)

        memory = F.relu(self.dropout(self.lin(memory))) # [Bs, Sz, args.hidden_size]
        memory = self.LN(memory)

        return memory # [Bs, Sz, args.hidden_size]

