import json
from logging import Logger
import os
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import InteractionModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model
from .lamb import Lamb
from lifelines.utils import concordance_index

def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None,
                 tokenizer = None,
                 sps_tokenizer = None,
                 smiles_tokenizer = None,
                 pbpe = None,
                 dbpe = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """

    #print("run_training.py : data", data._targets); input()
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    # 현재 시각을 가져오기
    current_time = datetime.now()

    # 현재 시각을 debug 로그로 기록
    debug(f"Current execution time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_features_path=args.separate_test_bond_features_path,
                             smiles_columns=args.smiles_columns,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_features_path=args.separate_val_bond_features_path,
                            smiles_columns = args.smiles_columns,
                            logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=(0.8, 0.0, 0.2),
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=(0.95, 0.05, 0.0),
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)


    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = None
    else:
        scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_sequences, test_sps, test_targets = test_data.smiles(), test_data.sequences(), test_data.sps(), test_data.targets()
    #print("Drop_Last test size :",int(len(test_smiles)//args.batch_size * args.batch_size)); input()
    if args.drop_last:
        test_len = int(len(test_smiles)//args.batch_size * args.batch_size)
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((test_len, args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((test_len, args.num_tasks))
    else:
        if args.dataset_type == 'multiclass':
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
        else:
            sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))
    
    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=False,
        seed=args.seed,
        drop_last=args.drop_last
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=args.drop_last
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=args.drop_last
    )
    
    #print("run_trainings, len_data_loaders [train/valid/test]:")
    #print(len(train_data_loader.dataset))
    #print(len(val_data_loader.dataset))
    #print(len(test_data_loader.dataset)); input()
    #input()
    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        ####
        #if args.sps_cnn==True and args.D_MPNN==True and args.ECFP==True:   ##
        #    model_name = 'Full_'                                           ##
        if args.sps_cnn==True and args.D_MPNN==True and args.ECFP==True:
            model_name = 'Full_sps_'
        elif args.prot_cnn==True and args.D_MPNN==True and args.ECFP==True:
            model_name = 'Full_prot_'
        elif args.sps_cnn==True and args.D_MPNN==True and args.ECFP==False:
            model_name = 'ablation_ECFP_'
        elif args.sps_cnn==True and args.D_MPNN==False and args.ECFP==True:
            model_name = 'ablation_MPNN_'
        elif args.sps_cnn==False and args.D_MPNN==True and args.ECFP==True:
            model_name = 'ablation_sps_'
        elif args.sps_cnn==True and args.D_MPNN==False and args.ECFP==False:
            model_name = 'only_sps_'
        elif args.sps_cnn==False and args.D_MPNN==True and args.ECFP==False:
            model_name = 'only_MPNN_'
        elif args.sps_cnn==False and args.D_MPNN==False and args.ECFP==True:
            model_name = 'only_ECFP_'
        else:
            model_name = 'Baseline_'

        ####
        save_dir = os.path.join(args.save_dir, f'{model_name}model_{model_idx}')
        makedirs(save_dir)
        print(f"Save to Dir {save_dir}!")
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = InteractionModel(args)
            
        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
            model = load_frzn_model(model=model,path=args.checkpoint_frzn, current_args=args, logger=logger)     
        
        debug(model)
        
        if args.checkpoint_frzn is not None:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')
        
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)


        # # Learning rate schedulers
        # optimizer = Lamb(model.parameters(), lr=args.lamp_lr, weight_decay=0.01, betas=(.9, .999), adam= True)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
        scheduler = build_lr_scheduler(optimizer, args)
        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0

        HIGHER_IS_BEST = ['auc', 'prc-auc', 'accuracy', 'r2', 'ci'] # You must add your own metric if you want to add your own earlystop_metric 
        LOWER_IS_BEST= ['rmse', 'mse', 'cross_entropy' ,'binary_cross_entropy']
        
        if args.earlystop_metric:
            if args.earlystop_metric in HIGHER_IS_BEST:
                cnt , best = 0 , 0
            elif args.earlystop_metric in LOWER_IS_BEST:
                cnt , best = 0, 9999
            else:
                ValueError(f"Invalid earlystop_metric : {args.earlystop_metric}!")

        for epoch in trange(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer,
                tokenizer= tokenizer,
                sps_tokenizer = sps_tokenizer,
                smiles_tokenizer = smiles_tokenizer,
                pbpe = pbpe,
                dbpe = dbpe
            )

            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                args = args,
                scaler=scaler,
                logger=logger,
                tokenizer= tokenizer,
                sps_tokenizer=sps_tokenizer,
                smiles_tokenizer =smiles_tokenizer,
                pbpe = pbpe,
                dbpe = dbpe
            )
            
            for metric, scores in val_scores.items():

                # Average validation score
                avg_val_score = np.nanmean(scores)
                if args.earlystop_metric == metric:
                    if args.earlystop_metric in HIGHER_IS_BEST:
                        if avg_val_score > best:
                            best = avg_val_score
                            cnt = 0
                        else:
                            cnt +=1
                            print("Early Stop counting : ",cnt)
                    elif args.earlystop_metric in LOWER_IS_BEST:
                        if avg_val_score < best:
                            best = avg_val_score
                            cnt = 0
                        else:
                            cnt +=1
                            print("Early Stop counting : ",cnt)
                
                debug(f'Model {model_idx} Validation {metric} = {avg_val_score:.6f}')
                # print(name, param)
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)

                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)
            
            # Save model checkpoint if improved validation score
            avg_val_score = np.nanmean(val_scores[args.metric])
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler, features_scaler,
                                atom_descriptor_scaler, bond_feature_scaler, args)

            if cnt == args.earlystop:
                print(f"Train Early stopped! Metric by {args.earlystop_metric}, No improvement during {args.earlystop}steps!")
                break
        
        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), device=args.device, logger=logger)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler,
            tokenizer= tokenizer,
            sps_tokenizer = sps_tokenizer,
            smiles_tokenizer = smiles_tokenizer,
            args = args,
            pbpe = pbpe,
            dbpe = dbpe
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            logger=logger
        )
        
        """
        #sns.set_style('whitegrid')  # 그래프 스타일 설
        #print("prot_outputs", prot_outputs, type(prot_outputs)); input()
        #print("out_conv", out_conv, type(out_conv)); input()
        #print("smiles_outputs", smiles_outputs, type(smiles_outputs)); input()
        #print("mpnn_out", mpnn_out, type(mpnn_out)); input()
        #print("add_feature", add_feature, type(add_feature)); input()
        #sns.kdeplot(prot_outputs.cpu().flatten(), cmap='viridis', shade=True, label='prot_outputs')
        #sns.kdeplot(out_conv.cpu().flatten(), cmap='viridis', shade=True, label='out_conv')
        #plt.legend()
        #plt.show()
        
        sns.set_style('whitegrid')
        sns.kdeplot(smiles_outputs.cpu().flatten(), cmap='viridis', shade=True, label='smiles_outputs')
        sns.kdeplot(add_feature.cpu().flatten(), cmap='viridis', shade=True, label='add_feature')
        sns.kdeplot(mpnn_out.cpu().flatten(), cmap='viridis', shade=True, label='mpnn_out')
        plt.legend()
        plt.show()

        #prot = prot_outputs + out_conv
        #compound = smiles_outputs + mpnn_out + add_feature
        test_preds = np.array(test_preds)
        
        sns.set_style('whitegrid')
        #sns.kdeplot(cab.cpu().flatten(), cmap='viridis', shade=True, label='cab')
        sns.kdeplot(torch.from_numpy(test_preds).flatten(), cmap='viridis', shade=True, label='test_preds')
        plt.legend()
        plt.show()

        #print("run_training : sum_test_preds : ",sum_test_preds, type(sum_test_preds), len(sum_test_preds)); input()
        #print("run_training : test_preds : ",test_preds, type(test_preds), len(test_preds));input()
        #print("run_training : test_scores : ", test_scores, type(test_scores), len(test_scores));input()
        """

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)
            #print("runt_trianing :sum_test_preds :",sum_test_preds); input()

        # Average test score
        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, scores):
                    info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    #print("run_training, avg_test_preds :",avg_test_preds); input()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        logger=logger
    )
    #print("run_training.py : ensemble_scores :", ensemble_scores); input()

    prediction = []
    label = []
    for i in range(args.num_tasks):
        for j in range(len(avg_test_preds)):
            if test_targets[j][i] is not None:  # Skip those without targets
                prediction.append(avg_test_preds[j][i])
                label.append(float(test_targets[j][i]))
                
    cindex = concordance_index(label,prediction)
    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores, cindex
