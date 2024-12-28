"""Loads a trained chemprop model checkpoint and makes predictions on a dataset."""

from chemprop.train import chemprop_predict
#from chemprop.data.utils import get_vocabs
if __name__ == '__main__':
    #tokenizer, sps_tokenizer, smiles_tokenizer, pbpe, dbpe = get_vocabs() 
    chemprop_predict()
