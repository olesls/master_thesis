# Code to train a intra-/inter-session RNN for prediction on the lastfm dataset

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # will probably be moved to code in TF 1.1
import datetime
import os
import time
import math
import numpy as np
from lastfm_utils_II_RNN import II_RNNDataHandler

dataset_path = os.path.expanduser('~') + '/datasets/lastfm-dataset-1K/lastfm_4_train_test_split.pickle'
epoch_file = './epoch_file.pickle'
checkpoint_file = './checkpoints/plain-rnn-'
checkpoint_file_ending = '.ckpt'
date_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
log_file = './testlog/'+str(date_now)+'-testing'


# This might not work, might have to set the seed inside the training loop or something
# TODO: Check if this works
tf.set_random_seed(0)

N_ITEMS      = -1       # number of items (size of 1-hot vector) (number of artists or songs in lastfm case)
BATCHSIZE    = 50       #
SHORTTERM_INTERNALSIZE = 1000     # size of internal vectors/states in the rnn
LONGTERM_INTERNALSIZE = SHORTTERM_INTERNALSIZE  # need to be the same unless we add a layer inbetween.
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 20-1     # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions. This is important for training, but when running, we can have as long sequences as we want! Just need to keep the hidden state and compute the next action)
EMBEDDING_SIZE = 1000
TOP_K = 20
MAX_EPOCHS = 10

learning_rate = 0.001   # fixed learning rate
dropout_pkeep = 1.0     # no dropout

# Load training data
datahandler = II_RNNDataHandler(dataset_path, BATCHSIZE, log_file)
N_ITEMS = datahandler.get_num_items()

print("------------------------------------------------------------------------")
print("CONFIG: N_ITEMS=", N_ITEMS, "BATCHSIZE=", BATCHSIZE, "INTERNALSIZE=", INTERNALSIZE,
        "N_LAYERS=", N_LAYERS, "SEQLEN=", SEQLEN, "EMBEDDING_SIZE=", EMBEDDING_SIZE)
print("------------------------------------------------------------------------")

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
    
    X_lt = tf.placeholder(tf.float32, [None, LONGTERM_INTERNALSIZE], name='') # [ BATCHSIZE, LT_INTERNALSIZE]

    # Longterm RNN
    lt_cell = rnn.GRUCell(LONGTERM_INTERNALSIZE)
    lt_rnn_outputs, lt_rnn_states = tf.nn.dynamic_rnn(lt_cell, X_lt, 
            sequence_length=seq_len_lt, dtype=tf.float32)

    # Get the correct outputs (depends on session_lengths)
    last_lt_rnn_output = tf.gather_nd(lt_rnn_outputs, tf.pack([tf.range(batchsize), seq_len_lt-1], axis=1))

    # Shortterm RNN
    onecell = rnn.GRUCell(SHORTTERM_INTERNALSIZE)
    dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = rnn.MultiRNNCell([dropcell]*N_LAYERS, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    Yr, H = tf.nn.dynamic_rnn(multicell, X_embed, 
            sequence_length=seq_len, dtype=tf.float32, initial_state=last_lt_rnn_output)

    H = tf.identity(H, name='H') # just to give it a name

    # Apply softmax to the output
    # Flatten the RNN output first, to share weights across the unrolled time steps
    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])         # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
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
#istate = np.zeros([BATCHSIZE, INTERNALSIZE*N_LAYERS])    # initial zero input state
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

num_training_batches = datahandler.get_num_training_batches()
num_test_batches = datahandler.get_num_test_batches()
while epoch <= MAX_EPOCHS:
    print("Starting epoch #"+str(epoch))
    epoch_loss = 0
    
    for _batch_number in range(num_training_batches):
        batch_start_time = time.time()

        xinput, targetvalues, sl = datahandler.get_next_train_batch()
        
        feed_dict = {X: xinput, Y_: targetvalues, lr: learning_rate, pkeep: dropout_pkeep, batchsize: len(xinput), 
                seq_len: sl}

        _, bl = sess.run([train_step, batchloss], feed_dict=feed_dict)
    
        # save training data for Tensorboard
        #summary_writer.add_summary(smm, _batch_number)

        batch_runtime = time.time() - batch_start_time
        epoch_loss += bl
        if _batch_number%100==0:
            print("Batch number:", str(_batch_number+1), "/", str(num_training_batches), "| Batch time:", "%.2f" % batch_runtime, " seconds", end='')
            print(" | Batch loss:", bl, end='')
            eta = (batch_runtime*(num_training_batches-_batch_number))/60
            eta = "%.2f" % eta
            print(" | ETR:", eta, "minutes.")

    print("Epoch", epoch, "finished")
    print("|- Epoch loss:", epoch_loss)

    # Save the model
    print("Saving model.")
    save_file = checkpoint_file + str(epoch) + checkpoint_file_ending
    save_path = saver.save(sess, save_file)
    print("|- Model saved in file:", save_path)

    datahandler.store_current_epoch(epoch, epoch_file)
    

    ##
    ##  TESTING
    ##
    print("Starting testing")
    recall, mrr = 0.0, 0.0
    evaluation_count = 0
    for _ in range(num_test_batches):
        print(_, "/", num_test_batches)
        xinput, targetvalues, sl = datahandler.get_next_test_batch()

        feed_dict = {X: xinput, pkeep: 1.0, batchsize: len(xinput), seq_len: sl}
        batch_predictions = sess.run([Y_prediction], feed_dict=feed_dict)
        batch_predictions = batch_predictions[0]

        for batch_index in range(len(batch_predictions)):
            try:
                predicted_sequence = batch_predictions[batch_index]
                target_sequence = targetvalues[batch_index]
            except Exception:
                print("len(batch_predictions)", len(batch_predictions))
                print("batch_index", batch_index)

            for i in range(sl[batch_index]):
                target_item = target_sequence[i]
                k_predictions = predicted_sequence[i]

                if target_item in k_predictions:
                    recall += 1
                    rank = np.nonzero(k_predictions == target_item)[0][0]+1
                    mrr += 1.0/rank

                evaluation_count += 1
    recall = recall/evaluation_count
    mrr = mrr/evaluation_count

    print("  Recall@"+str(TOP_K)+":", recall)
    print("  MRR@"+str(TOP_K)+":", mrr)
    print()

    datahandler.log_test_stats(epoch, epoch_loss, recall, mrr, TOP_K)

    datahandler.reset_batches()
    epoch += 1
