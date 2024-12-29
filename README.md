# KNU_DTI/DTA (Version 1.0)
A Pytorch Implementation of paper:
**KNU_DTI/DTA: KNowledge United Drug-Target Interaction predcition** (Not published yet)

Ryong Heo1,3, Dahyeon Lee2, Byung Ju Kim3, Sangmin Seo4, Sanghyun Park4, and Chihyun Park1,2,3,5*

1Department of Medical Bigdata Convergence, Kangwon National University, Chuncheon-si, 24341, Gangwon-do, Republic of Korea, 2Department of Data Science, Kangwon National University, 3UBLBio Corporation, Yeongtong-ro 237, Suwon, 16679, Gyeonggi-do, Republic of Korea, 4Department of Computer Science, Yonsei University, Yonsei-ro 50, Seodaemun-gu, 03722, Seoul, Republic of Korea, 5Department of Computer Science and Engineering, Kangwon National University
*correspondent author

Our reposistory uses:

https://github.com/chemprop/chemprop as a backbone for compound information extraction.

https://github.com/dmis-lab/PerceiverCPI as a backbone for model structure.

We highly recommend researchers read both papers 
[D-MPNN](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) and [PerceiverCPI](https://doi.org/10.1093/bioinformatics/btac731) to better understand how it was used. 

We express our sincere gratitude to the PerceiverCPI team for providing the primary inspiration for our study.


# 0.**Overview of KNU_DTI/DTA**
![image]([KNU_DTI_Figure_1.pdf](https://github.com/user-attachments/files/18266073/KNU_DTI_Figure_1.pdf))

The KNU-DTI model integrates structural and contextual features of proteins and compounds to predict drug-target interactions effectively. It learns structural properties from proteins using Structural Property Sequences (SPS) and from compounds through molecular graphs and ECFP fingerprints. Independent representation vectors are generated via GCNN, D-MPNN, MLP, and transformer-based modules to capture diverse and complementary feature spaces. These vectors are combined through element-wise addition, leveraging orthogonality to integrate independent information without redundancy. The model outperforms existing methods in predictive performance and generalizability across multiple datasets. Its efficient design balances simplicity and scalability, making it practical for real-world drug discovery applications.

Set up the environment:

In our experiment we use, Python 3.9.12 with PyTorch 2.0.1 + CUDA 11.3 and rdkit==2023.3.2.
For more detailed environment configurations, please refer to the environments.yml file.

```bash
git clone [https://github.com/DBpackage/KNU_DTA.git]
conda env create -f environment.yml
```

# 1. **Preparing Data**
The toy_data directory includes the toy dataset required to run classification or regression tasks using the model.

The full dataset used in the model, due to its size, has been uploaded separately and can be accessed [https://drive.google.com/drive/folders/1oUrTDG0l11baqCLAi2VsnS-vjAooceZS?usp=sharing].

You can make your own dataset. Make the csv file with this format.

| smiles  | sequence | pka | sps | label |
| ------------- | ------------- |------------- |------------- |------------- | 
| COc1cc(CCCOC(=O)  | MDVLSPGQGNNTTS  | 10.34969248 | CEDL,BNGM,CEKM | 1 |
| OC(=O)C=C | MSWATRPPF  | 5.568636236 | AEKL,CETS,AEKM | 0 |

You can make Label column from pka column by setting any threshold value.
In our dataset, except for Davis, class labels were assigned based on a pKa threshold of 6.0, with values of 6.0 or higher labeled as 1 and those below 6.0 labeled as 0.

* We recommend referring to PubChem for 'canonical SMILES' for compounds.
* Data with a smiles column length of 5 or less cannot be processed by this model.
* If the protein sequences are not preprocessed according to the standards provided by DeepAffinity, issues may arise during the tokenization process.

## How to prepare SPS from your own dataset

The protein-SPS pair data for human proteins provided by DeepAffinity can be accessed at

[https://github.com/Shen-Lab/DeepAffinity/blob/master/data/dataset/uniprot.human.scratch_outputs.w_sps.tab_corrected.zip.]

Additionally, to support training and testing in this study, we provide an extended protein-SPS pair dataset that includes SPS data for proteins not covered in the original dataset. 

This extended dataset is available at [https://github.com/DBpackage/KNU_DTA/blob/main/toy_dataset/Human_SPS_ver5.csv.]

If you want to generate new SPS sequences corresponding to the unique proteins in your dataset, please refer to the following page for instructions: [https://github.com/DBpackage/SPS].

# 2.**To train the model:**
```bash
python train.py --data_path "datasetpath" --separate_val_path "validationpath" --separate_test_path "testpath" --metric mse --dataset_type regression --save_dir "checkpointpath" --target_columns label
```
_Usage Example:_
~~~
python train.py --data_path ./toy_dataset/novel_pair_0_train.csv --separate_val_path ./toy_dataset/novel_pair_0_val.csv --separate_test_path ./toy_dataset/novel_pair_0_test.csv --metric mse --dataset_type regression --save_dir regression_150_newprot_pre --target_columns label --epochs 150 --ensemble_size 3 --num_folds 1 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds
~~~
# 3.**To take the inferrence:**
```bash
python predict.py --test_path "testdatapath" --checkpoint_dir "checkpointpath" --preds_path "predictionpath.csv"
```
_Usage Example:_
~~~
python predict.py --test_path ./toy_dataset/novel_pair_0_test.csv --checkpoint_dir regression_150_newprot_pre --preds_path newnew_fold0.csv
~~~
# 4.**To train YOUR model:**

Your data should be in the format csv, and the column names are: 'smiles','sequences','label'.

You can freely tune the hyperparameter for your best performance (but highly recommend using the Bayesian optimization package).


~~~
@article{10.1093/bioinformatics/btac731,
    author = {Nguyen, Ngoc-Quang and Jang, Gwanghoon and Kim, Hajung and Kang, Jaewoo},
    title = "{Perceiver CPI: A nested cross-attention network for compound-protein interaction prediction}",
    journal = {Bioinformatics},
    year = {2022},
    month = {11},
    abstract = "{Compound-protein interaction (CPI) plays an essential role in drug discovery and is performed via expensive molecular docking simulations. Many artificial intelligence-based approaches have been proposed in this regard. Recently, two types of models have accomplished promising results in exploiting molecular information: graph convolutional neural networks that construct a learned molecular representation from a graph structure (atoms and bonds), and neural networks that can be applied to compute on descriptors or fingerprints of molecules. However, the superiority of one method over the other is yet to be determined. Modern studies have endeavored to aggregate information that is extracted from compounds and proteins to form the CPI task. Nonetheless, these approaches have used a simple concatenation to combine them, which cannot fully capture the interaction between such information.We propose the Perceiver CPI network, which adopts a cross-attention mechanism to improve the learning ability of the representation of drug and target interactions and exploits the rich information obtained from extended-connectivity fingerprints to improve the performance. We evaluated Perceiver CPI on three main datasets, Davis, KIBA, and Metz, to compare the performance of our proposed model with that of state-of-the-art methods. The proposed method achieved satisfactory performance and exhibited significant improvements over previous approaches in all experiments.Perceiver CPI is available at https://github.com/dmis-lab/PerceiverCPISupplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac731},
    url = {https://doi.org/10.1093/bioinformatics/btac731},
    note = {btac731},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac731/47214739/btac731.pdf},
}
~~~
