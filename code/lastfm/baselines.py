import collections
import datetime
import os
import pickle
import random
import time
from lastfm_utils import PlainRNNDataHandler
from test_util import Tester

reddit = "subreddit"
lastfm = "lastfm"

dataset = lastfm

dataset_path = os.path.expanduser('~') + '/datasets/'+dataset+'/4_train_test_split.pickle'

date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-testing'
BATCHSIZE = 2
datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE, log_file)
num_train_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
num_items = datahandler.get_num_items()
num_predictions = 19

def log_config(baseline):
    message =    "------------------------------------------------------------------------"
    message += "\nDATASET: "+dataset
    message += "\nBASELINE: "+baseline
    datahandler.log_config(message)
    print(message)

def most_recent_sequence_predicions(sequence, sequence_length):
    full_prediction_sequence = random.sample(range(1, num_items), num_predictions)
    predictions = []
    for i in range(sequence_length):
        current_item = sequence[i]
        if current_item in full_prediction_sequence:
            index = full_prediction_sequence.index(current_item)
            del(full_prediction_sequence[index])
        full_prediction_sequence.insert(0, current_item)
        predictions.append(full_prediction_sequence[:num_predictions])
    return predictions

def most_recent():
    log_config("most_recent")
    datahandler.reset_batches()
    tester = Tester()
    for batch_number in range(num_test_batches):
        x, y, sl = datahandler.get_next_test_batch()
        prediction_batch = []

        for i in range(len(x)):
            prediction_batch.append(most_recent_sequence_predicions(x[i], sl[i]))

        tester.evaluate_batch(prediction_batch, y, sl)

    test_stats = tester.get_stats_and_reset()
    print(test_stats)
    datahandler.log_test_stats(0, 0, test_stats)


def most_popular():
    log_config("most_popular")
    datahandler.reset_batches()
    popularity_count = [0]*(num_items+1)
    tester = Tester()

    # Training
    for _ in range(num_train_batches):
        x, y, sl = datahandler.get_next_train_batch()
        for i in range(len(x)):
            sequence_length = sl[i]+1
            items = x[i][:sequence_length]
            
            for item in items:
                popularity_count[item] += 1
    
    top_k = sorted(range(len(popularity_count)), key=lambda i:popularity_count[i])
    top_k = top_k[-num_predictions:]
    top_k = list(reversed(top_k))

    # Testing
    for _ in range(num_test_batches):
        x, y, sl = datahandler.get_next_test_batch()
        prediction_batch = []

        for i in range(len(x)):
            sequence_predictions = []
            for j in range(sl[i]):
                sequence_predictions.append(top_k)
            prediction_batch.append(sequence_predictions)

        tester.evaluate_batch(prediction_batch, y, sl)
    
    test_stats = tester.get_stats_and_reset()
    print(test_stats)
    datahandler.log_test_stats(0, 0, test_stats)
    

most_recent()
most_popular()

