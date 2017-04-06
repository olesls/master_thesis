import math
import numpy as np
import os
import pickle
import time

class PlainRNNDataHandler:
    
    def __init__(self, dataset_path, batch_size):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        print("Loading dataset")
        load_time = time.time()
        self.dataset = pickle.load(open(self.dataset_path, 'rb'))
        print("|- dataset loaded in", str(time.time()-load_time), "s")
        self.current_index = 0

    def get_num_sequences(self):
        return len(self.dataset)

    def get_num_batches(self):
        ''' Returns how many batches the dataset contains, given the batch_size
        '''
        return math.floor(self.get_num_sequences() / self.batch_size)
    
    def get_next_batch(self):
        ''' Returns a slice of the dataset, which is the next batch.
        '''
        start = self.current_index
        end = start + self.batch_size+1
        batch = self.dataset[start:end]

        sl = self.sequence_lengths[start:end]

        self.current_index = end

        # Handle wraparound of the list of samples
        if end >= self.get_num_sequences():
            diff = end - self.num_samples
            batch = batch + self.dataset[0:diff]
            sl = sl + self.sequence_lengths[0:diff]

            self.current_index = diff
        
        x = batch[:-1]
        y = batch[1:]
        
        return x, y, sl
    
    def reset_batches(self):
        self.current_index = 0

    def get_num_items(self):
        ''' Find out the number of items/labels in the dataset.
        '''
        labels = {}

        for session in self.dataset:
            for label in session:
                if label not in labels:
                    labels[label] = True

        return len(labels)

    def pad_seq(self, seq, pad_len):
        return np.pad(seq, (0, pad_len - len(seq)), 'constant', constant_values=-1)

    def set_max_seq_len(self,max_seq_len):
        ''' Split sequences that are too long into shorter sequences based on 
            the specified maximum sequence length
        '''
        print("Padding sequences")
        padding_time = time.time()
        self.max_seq_len = max_seq_len

        # check if we already have a stored version of the requested data
        filename = "padded_" + str(max_seq_len)+".pickle"
        if os.path.isfile(filename):
            print("|- found existing version of data. Loading existing version")
            self.dataset = pickle.load(open(filename, 'rb'))
        else:
            new_dataset = []
            for sequence in self.dataset:
                if len(sequence) < max_seq_len:
                    appended_sequence = self.pad_seq(sequence, max_seq_len)
                    new_dataset.append(appended_sequence)
                else:
                    # The sequence is too long. Split it
                    splits = [sequence[i:i+max_seq_len] for i in range(0, len(sequence), max_seq_len)]
                    # The last sequence split might be too short (only one element), and should be removed in that case
                    if len(splits[-1]) == 1:
                        del splits[-1]

                    splits = [self.pad_seq(s, max_seq_len) for s in splits]
                    new_dataset += splits

            self.dataset = new_dataset
            pickle.dump(self.dataset, open(filename, 'wb'))

        self.sequence_lengths = []
        for s in self.dataset:
            if -1 in s:
                index = 0
                for i in range(len(s)):
                    if s[i] == -1:
                        index = i
                        break
                self.sequence_lengths.append(index-1)
            else:
                self.sequence_lengths.append(len(s)-1)
        

        print("|- sequences padded in", str(time.time()-padding_time), "s")

#    def split_to_test_and_training(self, training_ratio=0.8):
