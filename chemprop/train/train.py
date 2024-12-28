import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import InteractionModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR
from chemprop.data.utils import protein2emb_encoder, drug2emb_encoder, SPS2emb_encoder
import numpy as np

def train(model: InteractionModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None,
          tokenizer = None,
          sps_tokenizer= None,
          smiles_tokenizer = None,
          pbpe = None,
          dbpe = None) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~chemprop.models.model.InteractionModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, smiles_batch, target_batch, protein_sequence_batch, sps_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.smiles(), batch.targets(), batch.sequences(), batch.sps(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights(), batch.add_features()
        
        #print("smiles_batch :", smiles_batch); input()
        #print("protein_seq_batch :",protein_sequence_batch); input()
        #print("sps_sequence_batch :", sps_sequence_batch); input()
        #input()
        # add_feature : 2048 dimension Morgan circular fingerprint
        #print(sps_sequence_batch)
        #print(sps_tokenizer)
        #print("sps_sequence_batch :", sps_sequence_batch)
        #for arr in sps_sequence_batch:
        #    print(arr)
        #    temp = SPS2emb_encoder(arr[0], sps_tokenizer, args=TrainArgs)
        #    print(temp)
        #    input()
        # print("train.py : batch.targets()[:10] ", target_batch[:10]); input()
        #print("train.py : batch.smiles()[:10] ", smiles_batch[:10]); input()
        #print("train.py : batch.sequences()[:10] ", protein_sequence_batch[:10]); input()

        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]) # [bs, 1]
        mask_weight = torch.Tensor([[args.alpha if list(args.tau)[0]<=x<= list(args.tau)[1] else args.beta for x in tb] for tb in target_batch]) # [bs, 1]
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]) # [bs, 1]
        #print("targets size :", targets.size())
        #print("mask_wieght.size :", mask_weight.size())

        if args.target_weights is not None:
            target_weights = torch.Tensor(args.target_weights)
        else:
            target_weights = torch.ones_like(targets)
        data_weights = torch.Tensor(data_weights_batch).unsqueeze(1)
        
        # Run model
        model.zero_grad()
        
        dummy_array          = [0] * args.sequence_length # 단백질 토큰화 args.sequence_length만큼의 0 리스트 더미를 만듬
        smiles_dummy_array   = [0] * args.smiles_length
        sps_dummy_array      = [0] * args.sps_length

        #input()
        new_ar          = []
        sps_new_ar      = []
        smiles_new_ar   = []

        if args.prot_cnn: # character wise tokenizer
            sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
            #print("sequence_2_ar , : ", sequence_2_ar); input()
            #print("sequence_2_ar length : ", len(sequence_2_ar[0])); input()
            for arr in sequence_2_ar:
                while len(arr)>args.sequence_length:
                    arr.pop(len(arr)-1)
                # print(len(arr)
                new_ar.append(np.zeros(args.sequence_length)+np.array(arr))
        else: # subword_tokenizer
            for arr in protein_sequence_batch:
                temp = list(protein2emb_encoder(arr[0], tokenizer, pbpe, args)) # 여기도 args= TrainArgs를 전달하고 있어서 엉뚱한 길이로 잘리던 거였음
                new_ar.append(temp)

        #for arr in sequence_2_ar: # Truncation of protein sequecnes upto args.sequence_length
        #    while len(arr)>args.sequence_length:
        #        arr.pop(len(arr)-1)
            # print(len(arr)
        #    new_ar.append(np.zeros(args.sequence_length)+np.array(arr))
        #print("len new_ar", len(new_ar)) # bs
        #print("len new_ar[0]", len(new_ar[0])) # args.sequence_length
        #print("len new_ar[1]", len(new_ar[1])) # args.sequence_length
        #input()

        #for arr in protein_sequence_batch:
        #    temp = list(protein2emb_encoder(arr[0], tokenizer, pbpe, args)) # 여기도 args= TrainArgs를 전달하고 있어서 엉뚱한 길이로 잘리던 거였음
        #    new_ar.append(temp)

        for arr in smiles_batch:
            temp = list(drug2emb_encoder(arr[0], smiles_tokenizer, dbpe, args))
            smiles_new_ar.append(temp)

        for arr in sps_sequence_batch:
            temp = list(SPS2emb_encoder(arr[0], sps_tokenizer, args))
            sps_new_ar.append(temp)
        #print(sps_new_ar)
        #input()
        #print("After protein embedding :", new_ar); input()
        #print("After sps embedding     :",sps_new_ar); input()
        #print("After smiles embedding  :",smiles_new_ar); input()

        # convert list_sequence to tensor
        sequence_tensor = torch.LongTensor(new_ar) # [bs, args.sequence_length]
        sps_tensor = torch.LongTensor(sps_new_ar) # [bs, args.sps_length]
        smiles_tensor = torch.LongTensor(smiles_new_ar) # [bs, args.smiles_length]
        #smiles_mask = torch.LongTensor(tokenized_smiles.attention_mask) # [bs, args.smiles_length]
        #print(smiles_tensor.size())
        #print(smiles_mask.size())
        #input()
        #print("sps_tensor size:", sps_tensor.size())
        #input()
        add_feature = torch.Tensor(add_feature) # [bs, 2048] 
        #print("add_feature size:", add_feature.size()); input()
        #input()
        preds = model(mol_batch, smiles_tensor, sequence_tensor, sps_tensor, add_feature, features_batch=None, atom_descriptors_batch=None, atom_features_batch=None, bond_features_batch=None)
        #print("preds :", preds); 
        #print("target :", targets); input()
        # Move tensors to correct device
        mask = mask.to(preds.device)
        mask_weight = mask_weight.to(preds.device)
        targets = targets.to(preds.device)

        target_weights = target_weights.to(preds.device)
        data_weights = data_weights.to(preds.device)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask_weight
        loss = loss.sum() / mask.sum()
        #print("loss : ", loss.item())

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n_iter += len(batch)
    return n_iter
