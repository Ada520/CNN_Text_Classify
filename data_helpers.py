import numpy as np
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string=re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)