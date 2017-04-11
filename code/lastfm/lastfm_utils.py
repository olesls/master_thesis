import math
import numpy as np
import os
import pickle
import time

class PlainRNNDataHandler:
    
    def __init__(self, dataset_path, batch_size, sequence_length):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        print("Loading dataset")
        load_time = time.time()
        self.dataset = pickle.load(open(self.dataset_path, 'rb'))
        print("|- dataset loaded in", str(time.time()-load_time), "s")
        self.current_index = 0
        
        self.split_dataset()

    def split_dataset(self):

