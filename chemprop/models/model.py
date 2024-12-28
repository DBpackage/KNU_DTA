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
from .CAB import CrossAttentionBlock as CAB
#from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer
#from tape import ProteinBertModel
from .transformers import Protein_SPS_Transformer, Drug_Encoder, Protein_Encoder
from chemprop.data.utils import get_vocabs

class InteractionModel(nn.Module):
    """A :class:`InteractionNet` is a model which contains a D-MPNN and MPL and 1DCNN following by Cross attention Block"""


    def __init__(self, args: TrainArgs, featurizer: bool = False):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param featurizer: Whether the model should act as a featurizer, i.e., outputting the
                           learned features from the last layer prior to prediction rather than
                           outputting the actual property predictions.
        """
        super(InteractionModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.featurizer = featurizer
        
        self.sps_cnn    = args.sps_cnn
        self.D_MPNN     = args.D_MPNN
        self.ECFP       = args.ECFP
        self.prot_cnn   = args.prot_cnn
        
        words2idx_p, words2idx_s, words2idx_d, _, _ = get_vocabs(args)

        # smiles transformer (Need)
        self.smiles_transformer = Drug_Encoder(args) # 그냥 args로 전달해야하는걸 TrainArgs로 전달하니까 바뀌는 거였음

        # protein transformer (Need)
        self.prot_transformer = Protein_Encoder(args)

        self.scale = torch.sqrt(torch.FloatTensor([args.alpha])).cuda()
        self.relu = nn.ReLU()
        self.hidden_norm = nn.LayerNorm(args.hidden_size)
        self.do = nn.Dropout(args.dropout)

        self.output_size = args.num_tasks

        if self.multiclass:
            self.output_size *= args.multiclass_num_classes

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
    
        if self.ECFP: 
            self.fc_mg = nn.Linear(2048, args.hidden_size)
        
        if self.D_MPNN:
            self.create_encoder(args)

        if self.prot_cnn:
            #self.sps_embedding = nn.Embedding(len(words2idx_s), args.sps_hidden_size) # [Bs, sps_len, args.sps_hidden_size]
            self.protein_embedding = nn.Embedding(31, args.sps_hidden_size) # [Bs, ]
            self.conv_in = nn.Conv1d(in_channels=args.sequence_length, out_channels=args.sps_1d_out, kernel_size=1)
            self.convs = nn.ModuleList([nn.Conv1d(args.sps_hidden_size, 2*args.sps_hidden_size, args.kernel_size, padding=args.kernel_size//2) for _ in range(args.sps_1dcnn_numlayers)])
            self.cnn_fc = nn.Linear(args.sps_hidden_size*args.sps_1d_out, args.hidden_size)
            self.norm = nn.LayerNorm(args.sps_1d_out)
        if self.sps_cnn:
            self.sps_embedding = nn.Embedding(len(words2idx_s), args.sps_hidden_size) # [Bs, sps_len, args.sps_hidden_size]
            self.conv_in = nn.Conv1d(in_channels=args.sps_length, out_channels=args.sps_1d_out, kernel_size=1) 
            self.convs = nn.ModuleList([nn.Conv1d(args.sps_hidden_size, 2*args.sps_hidden_size, args.kernel_size, padding=args.kernel_size//2) for _ in range(args.sps_1dcnn_numlayers)])
            self.cnn_fc = nn.Linear(args.sps_hidden_size*args.sps_1d_out, args.hidden_size)
            self.norm = nn.LayerNorm(args.sps_1d_out)

        ################
        #self.D_FC = nn.Linear(args.hidden_size+args.smiles_length, args.sequence_length)
        input_dim = [2 * args.hidden_size]

        cls_dims = [128, 64, 32, 1]

        dims = input_dim + cls_dims
        nlayers = len(dims) -1

        self.Final_FC = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(nlayers)])
        ################
        
        initialize_weights(self)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)
              
        if args.checkpoint_frzn is not None:
            if args.freeze_first_only: # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad=False
            else: # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad=False                   
                        
    def create_ffn(self, 
                   args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
            #print("self.multiclass")

        if args.features_only:
            first_linear_dim = args.features_size
            #print("args.features_only")
        else:
            first_linear_dim = args.hidden_size * args.number_of_molecules
            #print("args.features_only else")

            if args.use_input_features:
                first_linear_dim += args.features_size
                #print("ASDF")

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size
            #print("atom_descriptors")

        #print("IN THE MODEL args.smiles_legnth",args.smiles_length); input()
        #print("sps length , protein length :", args.sps_length, args.sequence_length);input()
        #print("IN THE MODEL args.hidden_size", args.hidden_size); input()
        #print("multiple : ", args.smiles_length * args.hidden_size);input()
        input_dim = args.smiles_length * args.hidden_size
        
        dims = [input_dim] + args.Final_cls_hidden_dim
        layers = len(dims) -1
        
        self.ffn = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layers)])

        #print("input_dim:", input_dim)
        #input()

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN model
        #self.ffn = nn.Sequential(*ffn)
        
        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers >0:
                for param in list(self.ffn.parameters())[0:2*args.frzn_ffn_layers]: # Freeze weights and bias for given number of layers
                    param.requires_grad=False


    def featurize(self,
                  batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                  features_batch: List[np.ndarray] = None,
                  atom_descriptors_batch: List[np.ndarray] = None,
                  atom_features_batch: List[np.ndarray] = None,
                  bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Computes feature vectors of the input by running the model except for the last layer.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The feature vectors computed by the :class:`InteractionModel`.
        """
        return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch,
                                          atom_features_batch, bond_features_batch))

    def fingerprint(self,
                  batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                  features_batch: List[np.ndarray] = None,
                  atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes the fingerprint vectors of the input molecules by passing the inputs through the MPNN and returning
        the latent representation before the FFNN.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :return: The fingerprint vectors calculated through the MPNN.
        """
        return self.encoder(batch, features_batch, atom_descriptors_batch)


    def normalization(self,vector_present,threshold=0.1):
        
        vector_present_clone = vector_present.clone()
        num = vector_present_clone - vector_present_clone.min(1,keepdim = True)[0]
        de = vector_present_clone.max(1,keepdim = True)[0] - vector_present_clone.min(1,keepdim = True)[0]

        return num / de

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                smiles_tensor : List[np.ndarray] = None,
                sequence_tensor: List[np.ndarray] = None,
                sps_tensor : List[np.ndarray] = None,
                add_feature: List[np.ndarray] = None,
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`InteractionNet` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :sequence_tensor: A list of numpy arrays contraning Protein Encoding vectors
        :add_feature: A list of numpy arrays containing additional features (Morgan' Fingerprint).
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The output of the :class:`InteractionNet`, which is either property predictions
                 or molecule features if :code:`self.featurizer=True`.
        """
        #print("HERE BEFORE MPN CODE, start point of InteractionModel forward")
        #print("batch : ", batch); 
        #print("smiles_tensor :", smiles_tensor, smiles_tensor.size()); input()
        #print("smiles_mask :", smiles_mask, smiles_mask.size()); 
        #print("sequence_tensor :", sequence_tensor, sequence_tensor.size()); input()
        #print("sps_tensor :", sps_tensor, sps_tensor.size()); input()
        #print("add_feature :" , add_feature, add_feature.size()); input()
       
        smiles_tensor, sequence_tensor, sps_tensor, add_feature = smiles_tensor.cuda(), sequence_tensor.cuda(), sps_tensor.cuda(), add_feature.cuda()

        if self.featurizer:
            return self.featurize(batch, features_batch, atom_descriptors_batch,
                                  atom_features_batch, bond_features_batch)

        # SMILES transformer part
        smiles_outputs = self.smiles_transformer(smiles_tensor)     # [Bs, smiles_len, H]
        smiles_outputs = torch.mean(smiles_outputs, dim=1)          # [Bs, H]

        # 1D Graph feature
        if self.D_MPNN:
            #mpnn_out = self.normalization(self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)) # [Bs, H]
            mpnn_out = self.encoder(batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch) # [Bs, H]
        # Chemical Fingerprint feature
        if self.ECFP:
            add_feature = self.do(self.relu(self.hidden_norm(self.fc_mg(add_feature)))) # [Bs, H]

        if self.D_MPNN and self.ECFP:
            smiles_final = smiles_outputs + mpnn_out + add_feature  # [Bs, H] A + B + C
        elif self.D_MPNN==True and self.ECFP==False:
            smiles_final = smiles_outputs + mpnn_out                # [Bs, H] A + B
        elif self.D_MPNN==False and self.ECFP==True:
            smiles_final = smiles_outputs + add_feature             # [Bs, H] A + C
        else:
            smiles_final = smiles_outputs                           # [Bs, H] A 
        #print("temp size :", temp.size())
        
        # Protein transformer part
        # print("asdfadfsadfasd:", sequence_tensor[0], sequence_tensor.size()); input()
        prot_outputs = self.prot_transformer(sequence_tensor)   # [Bs, prot_len, H]
        prot_outputs = torch.mean(prot_outputs, dim=1)          # [Bs, H]
        
        ######
        # if Sequence GCNN
        if self.prot_cnn:
            prot_tensor = self.protein_embedding(sequence_tensor) # [Bs, prot_len, args.sps_hidden_size(C)]
            input_nn = self.conv_in(prot_tensor)         # [Bs, sps_out_channels, args.sps_hidden_size(C)]
            conv_input = input_nn.permute(0, 2, 1)      # [Bs, args.sps_hidden_size(C), sps_out_channels]

            for i, conv in enumerate(self.convs):
                # input conv shape [Bs, C, sps_out_channels]

                conved = self.norm(conv(conv_input))            # [Bs, 2*C, sps_out_channels]
                #print(f"conv{i}: conved1 :", conved.size());input()

                conved = F.glu(conved, dim=1)                   # [Bs, C, sps_out_channels]
                #print(f"conv{i}: conved2 :", conved.size());input()

                conved = conved + self.scale*conv_input         # [Bs, C, sps_out_channels]
                #print(f"conv{i}: conved3 :", conved.size());input()

                conv_input = conved # [Bs, C, sps_out_channels]

            #print("conved size: ",conved.size()); input()
            conved = conved.view(conved.size(0), -1)            # [Bs, C*sps_out_channels]
            out_conv = self.do(self.relu(self.hidden_norm(self.cnn_fc(conved))))  # [Bs, H]

            prot_sps_final = prot_outputs + out_conv            # [Bs, H]
            output = torch.cat((smiles_final, prot_sps_final), dim=1) # [Bs, 2H]
        #else:
        #    output = torch.cat((smiles_final, prot_outputs), dim=1) # [Bs, 2H]
        ######
        
        # elif SPS GCNN
        elif self.sps_cnn:
            sps_tensor = self.sps_embedding(sps_tensor) # [Bs, sps_len, args.sps_hidden_size(C)]
            input_nn = self.conv_in(sps_tensor)         # [Bs, sps_out_channels, args.sps_hidden_size(C)]
            conv_input = input_nn.permute(0, 2, 1)      # [Bs, args.sps_hidden_size(C), sps_out_channels]

            for i, conv in enumerate(self.convs):
                # input conv shape [Bs, C, sps_out_channels]

                conved = self.norm(conv(conv_input))            # [Bs, 2*C, sps_out_channels]
                #print(f"conv{i}: conved1 :", conved.size());input()

                conved = F.glu(conved, dim=1)                   # [Bs, C, sps_out_channels]
                #print(f"conv{i}: conved2 :", conved.size());input()

                conved = conved + self.scale*conv_input         # [Bs, C, sps_out_channels]
                #print(f"conv{i}: conved3 :", conved.size());input()

                conv_input = conved # [Bs, C, sps_out_channels] 

            #print("conved size: ",conved.size()); input() 
            conved = conved.view(conved.size(0), -1)            # [Bs, C*sps_out_channels]
            out_conv = self.do(self.relu(self.hidden_norm(self.cnn_fc(conved))))  # [Bs, H]

            prot_sps_final = prot_outputs + out_conv            # [Bs, H]
            output = torch.cat((smiles_final, prot_sps_final), dim=1) # [Bs, 2H]

        # else : GCNN is None
        else:
            output = torch.cat((smiles_final, prot_outputs), dim=1) # [Bs, 2H]

        # PROTEIN  : prot_outputs, out_conv
        # COMPOUND : smiles_outputs, mpnn_out, add_feature 

        # Output
        #print("output size: ", output.size()); input()

        for i, layer in enumerate(self.Final_FC):
            if i == len(self.Final_FC)-1:
                output = layer(output)
            else:
                output = self.do(self.relu(layer(output)))

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))  # batch size x num targets x num classes per target
            if not self.training:
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training as we're using CrossEntropyLoss
        #print("FINAL output :", output)
        
        return output
