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
        dataset = pickle.load(open(self.dataset_path, 'rb'))
        print("|- dataset loaded in", str(time.time()-load_time), "s")
        
        self.trainset = dataset['trainset']
        setf.testset = dataset['testset']
        self.train_session_lengths = dataset['train_session_lengths']
        self.test_session_lengths = dataset['test_session_lengths']
    
    def get_num_sessions(self, dataset):
        session_count = 0
        for k, v in dataset.items():
            session_count += len(v)
        return session_count

    def get_num_batches(self, dataset):
        num_sessions = self.get_num_sessions(dataset)
        return math.ceil(num_sessions/self.batch_size)

    def get_num_training_batches():
        self.get_num_batches(self.trainset)

    def get_num_test_batches():
        self.get_num_batches(self.testset)


