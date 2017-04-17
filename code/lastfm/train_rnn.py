# Code to train a plain RNN for prediction on the lastfm dataset

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # will probably be moved to code in TF 1.1. Keep it imported as rnn to make the rest of the code independent of this.
import os
import time
import math
import numpy as np
from lastfm_utils import PlainRNNDataHandler

dataset_path = os.path.expanduser('~') + '/datasets/lastfm-dataset-1K/lastfm_4_train_test_split.pickle'

# This might not work, might have to set the seed inside the training loop or something
# TODO: Check if this works, google it if it does not work.
tf.set_random_seed(0)

N_ITEMS      = -1       # number of items (size of 1-hot vector) (number of artists or songs in lastfm case)
BATCHSIZE    = 100      #
INTERNALSIZE = 1000     # size of internal vectors/states in the rnn
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 20-1     # maximum number of actions in a session (or more precisely, how far into the future an action affects future actions. This is important for training, but when running, we can have as long sequences as we want! Just need to keep the hidden state and compute the next action)
EMBEDDING_SIZE = 1000
TOP_K = 20

learning_rate = 0.001   # fixed learning rate
dropout_pkeep = 1.0     # no dropout

# Load training data
datahandler = PlainRNNDataHandler(dataset_path, BATCHSIZE)
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
    X = tf.placeholder(tf.int32, [BATCHSIZE, None], name='X')    # [ BATCHSIZE, SEQLEN ]
    Y_ = tf.placeholder(tf.int32, [BATCHSIZE, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]

    W_embed = tf.Variable(tf.random_uniform([N_ITEMS, EMBEDDING_SIZE], -1.0, 1.0), name='embeddings')
    X_embed = tf.nn.embedding_lookup(W_embed, X)

with tf.device(gpu[0]):
    seq_len = tf.placeholder(tf.int32, [BATCHSIZE], name='seqlen')
    batchsize = tf.placeholder(tf.int32, name='batchsize')

    lr = tf.placeholder(tf.float32, name='lr')              # learning rate
    pkeep = tf.placeholder(tf.float32, name='pkeep')        # dropout parameter

    # Inputs

    # RNN TODO: Ensure that multicell works even though we only have one layer
    onecell = rnn.GRUCell(INTERNALSIZE)
    dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
    multicell = rnn.MultiRNNCell([dropcell]*N_LAYERS, state_is_tuple=False)
    multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
    Yr, H = tf.nn.dynamic_rnn(multicell, X_embed, sequence_length=seq_len, dtype=tf.float32)

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

    # Unflatten loss
    loss = tf.reshape(loss, [batchsize, -1])            # [ BATCHSIZE, SEQLEN ]

    # Get the index of the highest scoring prediction through Y
    Yout_softmax = tf.nn.softmax(Ylogits, name='Yout')  # [BATCHSIZE x SEQLEN, EMBEDDING_SIZE ] TODO: Do we really need to softmax here?
    Y = tf.argmax(Yout_softmax, 1)   # [ BATCHSIZE x SEQLEN ]
    Y = tf.reshape(Y, [batchsize, -1], name='Y')        # [ BATCHSIZE, SEQLEN ]
    
    # Get prediction
    top_k_values, top_k_predictions = tf.nn.top_k(Yout_softmax, k=TOP_K)        # [BATCHSIZE x SEQLEN, TOP_K]
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
# istate = np.zeros([BATCHSIZE, INTERNALSIZE*N_LAYERS])    # initial zero input state
init = tf.global_variables_initializer()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True      # be nice and don't use more memory than necessary
sess = tf.Session(config=config)
sess.run(init)

# Tensorboard stuff
# Saves Tensorboard information into a different folder at each run
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training", sess.graph)
#validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

##
##  TRAINING
##

print("Starting training.")
num_batches = datahandler.get_num_training_batches()
for _batch_number in range(10):
#for _batch_number in range(num_batches):
    epoch_loss = 0
    batch_start_time = time.time()

    xinput, targetvalues, sl = datahandler.get_next_train_batch()
    feed_dict = {X: xinput, Y_: targetvalues, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE, 
            seq_len: sl}

    #_, y, smm, bl = sess.run([train_step, Y, summaries, batchloss], feed_dict=feed_dict)
    _, bl = sess.run([train_step, batchloss], feed_dict=feed_dict)
    
    # save training data for Tensorboard
    #summary_writer.add_summary(smm, _batch_number)

    batch_runtime = time.time() - batch_start_time
    epoch_loss += bl
    print("Batch number:", str(_batch_number), "/", str(num_batches), "| Batch time:", batch_runtime, end='')
    print(" | Batch loss:", bl)



##
##  TESTING
##

print("\n\n#############################################\n\nStarting testing.")
num_batches = datahandler.get_num_test_batches()
for _ in range(100):
    pass
