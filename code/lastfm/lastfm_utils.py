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

        self.train_current_session_index = 0
        self.train_current_user = 0
        self.test_current_session_index = 0
        self.test_current_user = 0
        
        self.num_users = len(self.trainset)
        if len(self.trainset) != len(self.testset):
            raise Exception("""Testset and trainset have different 
                    amount of users.""")

    
    def get_num_sessions(self, dataset):
        session_count = 0
        for k, v in dataset.items():
            session_count += len(v)
        return session_count

    def get_num_batches(self, dataset):
        num_sessions = self.get_num_sessions(dataset)
        return math.ceil(num_sessions/self.batch_size)

    def get_num_training_batches(self):
        self.get_num_batches(self.trainset)

    def get_num_test_batches(self):
        self.get_num_batches(self.testset)
    
    def get_subbatch(self, dataset, current_user, current_index, max_size):
        num_remaining_sessions_for_user = len(dataset[current_user]) - current_index
        b, sl, cu, ci = 0
        if num_remaining_sessions_for_user < max_size:
            # can use the whole of the remaining sessions for current user
            b  = dataset[current_user][current_index:]
            sl = dataset_session_lengths[current_user][current_index:]
            cu = current_user+1
            ci = 0
        else:
            # can only use up to max_size of remaining sessions for current user
            end_index = current_index + max_size
            b  = dataset[current_user][current_index:end_index]
            sl = dataset_session_lengths[current_user][current_index:end_index]
            cu = current_user
            ci = end_index
        
        return b, sl, cu, ci

    def process_batch(self, batch):
        batch = [[event[1] for event in session] for session in batch]

        x = batch[:-1]
        y = batch[1:]

        return x, y

    def get_next_batch(self, dataset, dataset_session_lengths, current_index, current_user):
        batch = []
        session_lengths = []

        while len(batch) < self.batch_size and current_user < self.num_users:
            num_remaining_sessions = self.batch_size - len(batch)
            sessions, sl, current_user, current_index = get_subbatch(dataset, 
                    dataset_session_lengths, current_user, current_index, num_remaining_sessions)

            batches += sessions
            session_lengths += sl
        
        session_lengths = session_lengths[:-1]  # only for X
        x, y = self.process_batch(batch)

        return x, y, session_lengths, current_user, current_index

    def get_next_train_batch(self):
        x, y, sl, cu, ci = self.get_next_batch(self.trainset, self.train_session_lengths, 
                self.train_current_user, self.train_current_session_index)
        self.train_current_user = cu
        self.train_current_session_index = ci

        return x, y, sl

    def get_next_test_batch(self):
        x, y, sl, cu, ci = self.get_next_batch(self.testset, self.test_session_lengths, 
                self.test_current_user, self.test_current_session_index)
        self.test_current_user = cu
        self.test_current_session_index = ci

        return x, y, sl
