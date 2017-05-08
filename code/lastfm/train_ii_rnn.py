# Code to train a plain RNN for prediction on the lastfm dataset

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # will probably be moved to code in TF 1.1. Keep it imported as rnn to make the rest of the code independent of this.
import datetime
import os
import time
import math
import numpy as np
from lastfm_utils_ii_rnn import IIRNNDataHandler
from test_util import Tester

reddit = "subreddit"
lastfm = "lastfm"

dataset = lastfm

home = os.path.expanduser('~')
if home == '/root':
    home = '/notebooks'

dataset_path = home + '/datasets/'+dataset+'/4_train_test_split.pickle'
epoch_file = './epoch_file-iirnn-'+dataset+'.pickle'
checkpoint_file = './checkpoints/ii-rnn-'+dataset+'-'
checkpoint_file_ending = '.ckpt'
date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-testing-ii-rnn.txt'

seed = 0
tf.set_random_seed(seed)

N_ITEMS      = -1       # number of items (size of 1-hot vector) #labels
BATCHSIZE    = 100      #
if dataset == reddit:
    ST_INTERNALSIZE = 50
    LT_INTERNALSIZE = ST_INTERNALSIZE
elif dataset == lastfm:
    ST_INTERNALSIZE = 150   # size of internal vectors/states in the rnn
    LT_INTERNALSIZE = ST_INTERNALSIZE
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 20-1     # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions. This is important for training, but when running, we can have as long sequences as we want! Just need to keep the hidden state and compute the next action)
EMBEDDING_SIZE = ST_INTERNALSIZE
TOP_K = 20
MAX_EPOCHS = 100
MAX_SESSION_REPRESENTATIONS = 15

learning_rate = 0.0007   # fixed learning rate
dropout_pkeep = 0.7     # no dropout

# Load training data
datahandler = IIRNNDataHandler(dataset_path, BATCHSIZE, log_file, 
        MAX_SESSION_REPRESENTATIONS, LT_INTERNALSIZE)
N_ITEMS = datahandler.get_num_items()
N_SESSIONS = datahandler.get_num_training_sessions()

message = "------------------------------------------------------------------------\n"
message += "DATASET: "+dataset+" MODEL: II-RNN"
message += "\nCONFIG: N_ITEMS="+str(N_ITEMS)+" BATCHSIZE="+str(BATCHSIZE)
message += "\nST_INTERNALSIZE="+str(ST_INTERNALSIZE)+" LT_INTERNALSIZE="+str(LT_INTERNALSIZE)
message += "\nN_LAYERS="+str(N_LAYERS)+" SEQLEN="+str(SEQLEN)+" EMBEDDING_SIZE="+str(EMBEDDING_SIZE)
message += "\nN_SESSIONS="+str(N_SESSIONS)+" SEED="+str(seed)
message += "\nMAX_SESSION_REPRESENTATIONS="+str(MAX_SESSION_REPRESENTATIONS)
datahandler.log_config(message)
print(message)


##
## The model
##
print("Creating model")
cpu = ['/cpu:0']
gpu = ['/gpu:0', '/gpu:1']

with tf.device(cpu[0]):
    # Inputs
    X = tf.placeholder(tf.int32, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
    Y_ = tf.placeholder(tf.int32, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]

    W_embed = tf.Variable(tf.random_uniform([N_ITEMS, EMBEDDING_SIZE], -1.0, 1.0), name='embeddings')
    X_embed = tf.nn.embedding_lookup(W_embed, X)

with tf.device(gpu[0]):
    seq_len = tf.placeholder(tf.int32, [None], name='seqlen')
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    lr = tf.placeholder(tf.float32, name='lr')              # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')        # dropout parameter

    X_lt = tf.placeholder(tf.float32, [None, None, LT_INTERNALSIZE], name='X_lt') #[BATCHSIZE, LT_INTERNALSIZE]
    seq_len_lt = tf.placeholder(tf.int32, [None], name='lt_seqlen')

    # Longterm RNN
    lt_cell = rnn.GRUCell(LT_INTERNALSIZE)
    lt_rnn_outputs, lt_rnn_states = tf.nn.dynamic_rnn(lt_cell, X_lt,
            sequence_length=seq_len_lt, dtype=tf.float32)

    # Get the correct outputs (depends on session_lengths)
    last_lt_rnn_output = tf.gather_nd(lt_rnn_outputs, tf.stack([tf.range(batchsize), seq_len_lt-1], axis=1))

    # Shortterm RNN
    onecell = rnn.GRUCell(ST_INTERNALSIZE)
    dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = rnn.MultiRNNCell([dropcell]*N_LAYERS, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    Yr, H = tf.nn.dynamic_rnn(multicell, X_embed, 
            sequence_length=seq_len, dtype=tf.float32, initial_state=last_lt_rnn_output)

    H = tf.identity(H, name='H') # just to give it a name

    # Apply softmax to the output
    # Flatten the RNN output first, to share weights across the unrolled time steps
    Yflat = tf.reshape(Yr, [-1, ST_INTERNALSIZE])         # [ BATCHSIZE x SEQLEN, ST_INTERNALSIZE ]
    # Change from internal size (from RNNCell) to N_ITEMS size
    Ylogits = layers.linear(Yflat, N_ITEMS)                     # [ BATCHSIZE x SEQLEN, N_ITEMS ]

#with tf.device(cpu[0]):
    # Flatten expected outputs to match actual outputs
    Y_flat_target = tf.reshape(Y_, [-1])    # [ BATCHSIZE x SEQLEN ]

    # Calculate loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_flat_target)    # [ BATCHSIZE x SEQLEN ]
    
    # Mask the losses (so we don't train in padded values)
    mask = tf.sign(tf.to_float(Y_flat_target))
    masked_loss = mask * loss

    # Unflatten loss
    loss = tf.reshape(masked_loss, [batchsize, -1])            # [ BATCHSIZE, SEQLEN ]

    # Get the index of the highest scoring prediction through Y
    Y = tf.argmax(Ylogits, 1)   # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name='Y')        # [ BATCHSIZE, SEQLEN ]
    
    # Get prediction
    top_k_values, top_k_predictions = tf.nn.top_k(Ylogits, k=TOP_K)        # [BATCHSIZE x SEQLEN, TOP_K]
    Y_prediction = tf.reshape(top_k_predictions, [batchsize, -1, TOP_K], name='YTopKPred')

    # Training
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    # Stats
    # Average sequence loss
    seqloss = tf.reduce_mean(loss, 1)
    # Average batchloss
    batchloss = tf.reduce_mean(seqloss)

# Average number of correct predictions
accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.int32)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init to save models
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)



# Initialization
# istate = np.zeros([BATCHSIZE, ST_INTERNALSIZE*N_LAYERS])    # initial zero input state
init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True      # be nice and don't use more memory than necessary
sess = tf.Session(config=config)
saver = tf.train.Saver()

# Tensorboard stuff
# Saves Tensorboard information into a different folder at each run
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training", sess.graph)
#validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

##
##  TRAINING
##

print("Starting training.")
epoch = datahandler.get_latest_epoch(epoch_file)
print("|-Starting on epoch", epoch+1)
if epoch > 0:
    print("|--Restoring model.")
    save_file = checkpoint_file + str(epoch) + checkpoint_file_ending
    saver.restore(sess, save_file)
else:
    sess.run(init)
epoch += 1
print()


best_recall5 = -1
best_recall20 = -1

num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
while epoch <= MAX_EPOCHS:
    print("Starting epoch #"+str(epoch))
    epoch_loss = 0
    
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, _ = datahandler.get_next_train_batch()

    while len(xinput) > int(BATCHSIZE/2):
        _batch_number += 1
        batch_start_time = time.time()

        feed_dict = {X: xinput, Y_: targetvalues, X_lt: session_reps, 
                seq_len_lt: sr_sl, lr: learning_rate, pkeep: dropout_pkeep, 
                batchsize: len(xinput), seq_len: sl}

        _, bl, final_H = sess.run([train_step, batchloss, H], feed_dict=feed_dict)

        datahandler.store_user_session_representations(final_H, user_list)
    
        # save training data for Tensorboard
        #summary_writer.add_summary(smm, _batch_number)

        batch_runtime = time.time() - batch_start_time
        epoch_loss += bl
        if _batch_number%100==0:
            print("Batch number:", str(_batch_number), "/", str(num_training_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
            print(" | Batch loss:", bl, end='')
            eta = (batch_runtime*(num_training_batches-_batch_number))/60
            eta = "%.2f" % eta
            print(" | ETA:", eta, "minutes.")
        
        xinput, targetvalues, sl, session_reps, sr_sl, user_list, _ = datahandler.get_next_train_batch()

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)

    ##
    ##  TESTING
    ##
    print("Starting testing")
    recall, mrr = 0.0, 0.0
    evaluation_count = 0
    tester = Tester()
    datahandler.reset_user_batch_data()
    _batch_number = 0
    xinput, targetvalues, sl, session_reps, sr_sl, user_list, _ = datahandler.get_next_test_batch()
    while len(xinput) > int(BATCHSIZE/2):
        batch_start_time = time.time()
        _batch_number += 1

        feed_dict = {X: xinput, pkeep: 1.0, batchsize: len(xinput), seq_len: sl,
                X_lt: session_reps, seq_len_lt: sr_sl}

        batch_predictions, final_H = sess.run([Y_prediction, H], feed_dict=feed_dict)
        datahandler.store_user_session_representations(final_H, user_list)
        
        # Evaluate predictions
        tester.evaluate_batch(batch_predictions, targetvalues, sl)

        # Print some stats during testing
        batch_runtime = time.time() - batch_start_time
        if _batch_number%100==0:
            print("Batch number:", str(_batch_number), "/", str(num_test_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds")
            eta = (batch_runtime*(num_test_batches-_batch_number))/60
            eta = "%.2f" % eta
            print("ETA:", eta, "minutes.")
            current_results = tester.get_stats()
            print("Current evaluation:")
            print(current_results)

        xinput, targetvalues, sl, session_reps, sr_sl, user_list, _ = datahandler.get_next_test_batch()

    # Print final test stats for epoch
    test_stats, current_recall5, current_recall20 = tester.get_stats_and_reset()
    print(test_stats)
    
    if current_recall5 > best_recall5 or current_recall20 > best_recall20:
        # Save the model
        print("Saving model.")
        save_file = checkpoint_file + str(epoch) + checkpoint_file_ending
        save_path = saver.save(sess, save_file)
        print("|- Model saved in file:", save_path)

        datahandler.store_current_epoch(epoch, epoch_file)

        best_recall5 = current_recall5
        best_recall20 = current_recall20
    

    datahandler.log_test_stats(epoch, epoch_loss, test_stats)

    epoch += 1
