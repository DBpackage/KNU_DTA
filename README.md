# KNU_DTI/DTA (Version 1.0)
A Pytorch Implementation of paper:
**KNU_DTI/DTA: KNowledge United Drug-Target Interaction predcition** (In the reivision step)

Ryong Heo1,3, Dahyeon Lee2, Byung Ju Kim3, Sangmin Seo4, Sanghyun Park4, and Chihyun Park1,2,3,5*

1 Department of Medical Bigdata Convergence, Kangwon National University, Chuncheon-si, 24341, Gangwon-do, Republic of Korea, 2 Department of Data Science, Kangwon National University, 3 UBLBio Corporation, Yeongtong-ro 237, Suwon, 16679, Gyeonggi-do, Republic of Korea, 4 Department of Computer Science, Yonsei University, Yonsei-ro 50, Seodaemun-gu, 03722, Seoul, Republic of Korea, 5Department of Computer Science and Engineering, Kangwon National University

*correspondent author

Our reposistory uses:

https://github.com/chemprop/chemprop as a backbone for compound information extraction.

https://github.com/dmis-lab/PerceiverCPI as a backbone for model structure.

We highly recommend researchers read both papers 
[D-MPNN](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and [PerceiverCPI](https://doi.org/10.1093/bioinformatics/btac731) to better understand how it was used. 

We express our sincere gratitude to the PerceiverCPI team for providing the primary inspiration for our study.


## 1. **Overview of KNU_DTI/DTA**

The KNU-DTI model integrates structural and contextual features of proteins and compounds to predict drug-target interactions effectively. It learns structural properties from proteins using Structural Property Sequences (SPS) and from compounds through molecular graphs and ECFP fingerprints. Independent representation vectors are generated via GCNN, D-MPNN, MLP, and transformer-based modules to capture diverse and complementary feature spaces. These vectors are combined through element-wise addition, leveraging orthogonality to integrate independent information without redundancy. The model outperforms existing methods in predictive performance and generalizability across multiple datasets. Its efficient design balances simplicity and scalability, making it practical for real-world drug discovery applications.

Set up the environment:

In our experiment we use, Python 3.9.12 with PyTorch 2.0.1 + CUDA 11.3 and rdkit==2023.3.2.
For more detailed environment configurations, please refer to the environments.yml file.

```bash
git clone [https://github.com/DBpackage/KNU_DTA.git]
conda env create -f environment.yml
```

## 2. **Preparing Data**
The toy_data directory includes the toy dataset required to run classification or regression tasks using the model.

The full dataset used in the model, due to its size, has been uploaded separately and can be accessed [https://drive.google.com/drive/folders/1oUrTDG0l11baqCLAi2VsnS-vjAooceZS?usp=sharing].

You can make your own dataset. Make the csv file with this format.

| smiles  | sequence | pka | sps | label |
| ------------- | ------------- |------------- |------------- |------------- | 
| COc1cc(CCCOC(=O)  | MDVLSPGQGNNTTS  | 10.34969248 | CEDL,BNGM,CEKM | 1 |
| OC(=O)C=C | MSWATRPPF  | 5.568636236 | AEKL,CETS,AEKM | 0 |

You can make Label column from pka column by setting any threshold value.
In our dataset, except for Davis, class labels were assigned based on a pKa threshold of 6.0, with values of 6.0 or higher labeled as 1 and those below 6.0 labeled as 0.

The smiles data is automatically converted into a 2D graph input for D-MPNN during the training and inference processes using RDKit. 
However, any smiles entries that do not comply with the 2D graph conversion algorithm may be dropped from the input dataset during this process.

* Please ensure that the column name and order are maintained.
* We recommend referring to PubChem for 'canonical SMILEs' for compounds.
* Data with a smiles column length of 5 or less cannot be processed by this model.
* If the protein sequences are not preprocessed according to the standards provided by DeepAffinity, issues may arise during the tokenization process.

## How to prepare SPS from your own dataset

The protein-SPS pair data for human proteins provided by DeepAffinity can be accessed at

[https://github.com/Shen-Lab/DeepAffinity/blob/master/data/dataset/uniprot.human.scratch_outputs.w_sps.tab_corrected.zip.]

Additionally, to support training and testing in this study, we provide an extended protein-SPS pair dataset that includes SPS data for proteins not covered in the original dataset. 

This extended dataset is available at [https://github.com/DBpackage/KNU_DTA/blob/main/toy_dataset/Human_SPS_ver5.csv.]

If you want to generate new SPS sequences corresponding to the unique proteins in your dataset, please refer to the following page for instructions: [https://github.com/DBpackage/SPS].

## 3.**To train the model:**

Please modify line 406 of chemprop/args.py to match your specific path.

#vocab_path = '/data/knu_hr/project/final/vocab/'

```bash
python train.py 
--data_path "dataset path"
--metric "metric"
--extra_metrics "extra_metrics" 
--dataset_type "regression or classification"
--target_columns "your target column name"
--save_dir "save directory"
--epochs "epoch times"
--ensemble_size "ensemble size"
--num_folds "k-fold cross validation"
--batch_size "batch size"
--transformer_d_model "transformer embedding dim"
--transformer_d_ff "transformer FFNN dim"
--transformer_nlayers "transformer layers"
--sequence_length "MAX sub-word tokenized protein sequence length"
--sps_length "MAX sps length"
--smiles_length "MAX sub-word tokenized smiles length"
--sps_cnn "sps learning module TRUE"
--D_MPNN "D-MPNN learning module TRUE"
--ECFP "ECFP learning module TRUE"
--earlystop "earlystopping step by [--metric] option"
--seed "training data shuffling seed"
--tau "tau"
--alpha "alpha" 
--beta "beta"
```
_Usage Example:_ For Regression
~~~
python train.py 
--data_path ./toy_dataset/toy_BindingDB_class.csv
--metric mse 
--extra_metrics rmse r2 ci 
--dataset_type regression 
--target_columns pka 
--save_dir ./toy_save
--epochs 120 
--ensemble_size 3 
--num_folds 1 
--batch_size 50 
--transformer_d_model 64 
--transformer_d_ff 512 
--transformer_nlayers 2 
--sequence_length 500 
--sps_length 70 
--smiles_length 50 
--sps_cnn 
--D_MPNN 
--ECFP 
--earlystop 50 
--seed 42 
--tau 0.0 5.0 
--alpha 0.5 
--beta 5.0
~~~

_Usage Example:_ For Classification
~~~
python train.py 
--data_path ./toy_dataset/toy_BindingDB_class.csv
--metric auc
--extra_metrics prc-auc accuracy
--dataset_type classification 
--target_columns label 
--save_dir ./toy_save
--epochs 120 
--ensemble_size 3 
--num_folds 1 
--batch_size 50 
--transformer_d_model 64 
--transformer_d_ff 512 
--transformer_nlayers 2 
--sequence_length 500 
--sps_length 70 
--smiles_length 50 
--sps_cnn 
--D_MPNN 
--ECFP 
--earlystop 50 
--seed 42 
~~~

## 4.**To take the inferrence:**
```bash
python predict.py 
--test_path "test dataset path"
--checkpoint_path "saved trained model path"
--preds_path "prediction result file name with save path"
===Below args must be same with training condition===
--batch_size 50 
--transformer_d_model 64 
--transformer_d_ff 512 
--transformer_nlayers 2 
--sequence_length 500 
--sps_length 70 
--smiles_length 50 
--sps_cnn 
--D_MPNN 
--ECFP 
======================================

```
_Usage Example:_
~~~
python predict.py 
--test_path ./toy_dataset/protein_class/test_GPCR_df.csv
--checkpoint_path ./save/fold_0/Full_sps_model_0/model.pt
--preds_path ./pred/GPCR_inference_results.csv 
===Below args must be same with training condition===
--batch_size 50 
--transformer_d_model 64 
--transformer_d_ff 512 
--transformer_nlayers 2 
--sequence_length 500 
--sps_length 70 
--smiles_length 50 
--sps_cnn 
--D_MPNN 
--ECFP
======================================================
~~~

# citation
~~~
@article{RyongHeo2024KNUDTI,
  title={KNU_DTI: KNowledge United Drug-Target Interaction predcition},
  author={Ryong Heo, Dahyeon Lee, Byung Ju Kim, Sangmin Seo, Sanghyun Park, and Chihyun Park},
  journal={submitted to Computers in Biology and Medicine (CIBM)}, 'In the reivision step'
}
~~~
