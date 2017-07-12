from theano_bpr.utils import load_data_from_array
from theano_bpr import BPR
from test_util import Tester
from lastfm_utils import PlainRNNDataHandler #for logging of score
import pickle
import time
import datetime
import os

reddit = "subreddit"
lastfm = "lastfm"

max_epochs = 20
dataset = lastfm

if dataset == lastfm:
    embedding_size = 100
    learning_rate = 0.01
elif dataset == reddit:
    embedding_size = 50
    learning_rate = 0.01

dataset_path = os.path.expanduser('~') + '/datasets/'+dataset+'/bpr-mf_train_test_split.pickle'

date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-BPR-MF-testing'
BATCHSIZE = 2 # dummy variable
datahandler = PlainRNNDataHandler("", BATCHSIZE, log_file)

message =    "------------------------------------------------------------------------"
message += "\nDATASET: "+dataset
message += "\nBASELINE: BPR-MF"
datahandler.log_config(message)
print(message)
print()


print("loading data with pickle")
p = pickle.load(open(dataset_path, 'rb'))

train = p['trainset']
train_sl = p['train_session_lengths']
test = p['testset']
test_sl = p['test_session_lengths']

def convert_data(dataset, session_lengths):
    a = []
    for user, sessions in list(dataset.items()):
        usl = session_lengths[user]
        for s in range(len(sessions)):
            session = sessions[s]
            ssl = usl[s] + 1
            for i in range(ssl):
                a.append((user, session[i][1]))
    return a

print("converting training data to array")
a = convert_data(train, train_sl)
training_array, uti, iti = load_data_from_array(a)
a = convert_data(test, test_sl)
testing_array, uti, iti = load_data_from_array(a, uti, iti)

print("creating BPR model")
bpr = BPR(embedding_size, len(list(uti.keys())), len(list(iti.keys())), learning_rate=learning_rate)


print("training model")
split2 = int(len(training_array)/2)
split1 = int(split2/2)
split3 = split1+split2


users = train.keys()
session_batch = []
sl = []
for user in users:
    session_batch.append(test[user][0])
    sl.append(test_sl[user][0])
session_batch = [[event[1] for event in session] for session in session_batch]
for s in range(len(session_batch)):
    session = session_batch[s]
    for i in range(len(session)):
        item = session[i]
        if item != 0:
            item = iti[item]
            session_batch[s][i] = item

x = [session[:-1] for session in session_batch]
y = [session[1:] for session in session_batch]

for epoch in range(max_epochs):
    runtime = time.time()

    bpr.train(training_array[:split1], epochs=1)
    bpr.train(training_array[split1:split2], epochs=1)
    bpr.train(training_array[split2:split3], epochs=1)
    bpr.train(training_array[split3:], epochs=1)
    #bpr.train(training_array, epochs=4)


    print("  epoch", epoch, " took ", str(time.time()-runtime),"seconds to run..")


    print("testing")
    tester = Tester()

    prediction_batch = []

    print("getting user predictions for testing")
    runtime = time.time()
    user_predictions = []
    for user in users:
        user_predictions.append(bpr.top_predictions(user, topn=20))
    print("  predictions found in", str(time.time()-runtime), "seconds")

    print(user_predictions[1])
    print(user_predictions[100])
    print(user_predictions[300])
    print(user_predictions[600])
    print(user_predictions[800])

    print("calculating test scores")
    for i in range(len(x)):
        sequence_predictions = []
        for j in range(sl[i]):
            sequence_predictions.append(user_predictions[i])
        prediction_batch.append(sequence_predictions)

    tester.evaluate_batch(prediction_batch, y, sl)

    test_stats, _1, _2 = tester.get_stats_and_reset()
    print(test_stats)
    datahandler.log_test_stats(epoch, -1, test_stats)


