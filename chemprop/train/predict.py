from typing import List

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import InteractionModel
import numpy as np
from chemprop.args import TrainArgs
from chemprop.data.utils import protein2emb_encoder, drug2emb_encoder, SPS2emb_encoder

def predict(model: InteractionModel,
            data_loader: MoleculeDataLoader,
            args: TrainArgs,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            tokenizer=None,
            sps_tokenizer=None,
            smiles_tokenizer=None,
            pbpe = None,
            dbpe = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.InteractionModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    print("From Here, prediction")
    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, smiles_batch, protein_sequence_batch, sps_sequence_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, add_feature = \
            batch.batch_graph(), batch.features(), batch.smiles(), batch.sequences(), batch.sps(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features(), batch.add_features()

        
        # print("predict.py : batch.targets()[:10] ", target_batch[:10]); input()
        # print("predict.py : batch.smiles()[:10] ", smiles_batch[:10]); input()

        dummy_array          = [0] * args.sequence_length
        smiles_dummy_array   = [0] * args.smiles_length
        sps_dummy_array      = [0] * args.sps_length

        new_ar          = []
        sps_new_ar      = []
        smiles_new_ar   = []

        if args.prot_cnn:
            #print("prot cnn !!!!!!!!!!!"); input()
            #print("tokenizer : ", tokenizer); input()
            sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
            #print("sequence_2_ar , : ", sequence_2_ar); input()
            #print("sequence_2_ar length : ", len(sequence_2_ar[0])); input()
            for arr in sequence_2_ar:
                while len(arr)>args.sequence_length:
                    arr.pop(len(arr)-1)
                # print(len(arr)
                new_ar.append(np.zeros(args.sequence_length)+np.array(arr))
        else:
            for arr in protein_sequence_batch:
                temp = list(protein2emb_encoder(arr[0], tokenizer, pbpe, args)) # 여기도 args= TrainArgs를 전달하고 있어서 엉뚱한 길이로 잘리던 거였음
                new_ar.append(temp)

        
        #for arr in protein_sequence_batch:
        #    temp = list(protein2emb_encoder(arr[0], tokenizer, pbpe, args))
        #    new_ar.append(temp)

        for arr in smiles_batch:
            temp = list(drug2emb_encoder(arr[0], smiles_tokenizer, dbpe, args))
            smiles_new_ar.append(temp)

        for arr in sps_sequence_batch:
            temp = list(SPS2emb_encoder(arr[0], sps_tokenizer, args))
            sps_new_ar.append(temp)
        

        sequence_tensor = torch.LongTensor(new_ar) # [bs, args.sequence_length]
        sps_tensor = torch.LongTensor(sps_new_ar) # [bs, args.sps_length]
        smiles_tensor = torch.LongTensor(smiles_new_ar) # [bs, args.smiles_length]

        #print("predict.py smiles_tensor:",smiles_tensor, smiles_tensor.size())
        #input()
        #sequence_2_ar = [list(tokenizer.encode(list(t[0]))) + dummy_array for t in protein_sequence_batch]
        #new_ar = []

        #for arr in sequence_2_ar:
        #    while len(arr)>args.sequence_length:
        #        arr.pop(len(arr)-1)
        #    new_ar.append(np.zeros(args.sequence_length)+np.array(arr))
        
        # convert list_sequence to tensor
        # sequence_tensor = torch.LongTensor(new_ar)
        add_feature = torch.Tensor(add_feature)

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, smiles_tensor, sequence_tensor, sps_tensor, add_feature, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        batch_preds = batch_preds.data.cpu().numpy()
        #print("predict.py : batch_preds len:", len(batch_preds))
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds
