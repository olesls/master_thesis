import pickle
import math

class PlainRNNDataHandler:
    
    def __init__(self, dataset_path, batch_size=50):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset = pickle.load(open(self.dataset_path, 'rb'))
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
        end = start + self.batch_size
        batch = self.dataset[start:end]

        self.current_index = end

        # Handle wraparound of the list of samples
        if end >= self.num_samples:
            diff = end - self.num_samples
            batch = batch + self.dataset[0:diff]

            self.current_index = diff
        
        x = batch[:-1]
        y = batch[1:]

        return x, y
    
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

    def set_max_seq_len(max_seq_len):
        ''' Split sequences that are too long into shorter sequences based on 
            the specified maximum sequence length
        '''
        new_dataset = []
        for sequence in self.dataset:
            if len(sequence) < max_seq_len:
                new_dataset.append(sequence)
                continue

            # The sequence is too long. Split it
            splits = (sequence[i:i+max_seq_len] for i in range(0, len(sequence), max_seq_len))
            # The last sequence split might be too short (only one element), and should be removed in that case
            if len(splits[-1]) == 1:
                del splits[-1]

            new_dataset += splits

        self.dataset = new_dataset
