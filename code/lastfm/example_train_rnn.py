# Code stolen from https://github.com/martin-gorner/tensorflow-rnn-shakespeare/blob/master/
# Only used to learn myself RNN in tensorflow-v1.0

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
tf.set_random_seed(0)

N_ITEMS      = 10000    # number of items (size of 1-hot vector) (artists in lastfm case)
BATCHSIZE    = 100
INTERNALSIZE = 512      # size of internal vectors/states in the rnn
N_LAYERS     = 1        # number of layers in the rnn
SEQLEN       = 30       # maximum number of actions in a session

learning_rate = 0.001   # fixed learning rate
dropout_pkeep = 1.0     # no dropout

# load data ....


#
# the model (see FAQ in README.md)
#
lr = tf.placeholder(tf.float32, name='lr')  # learning rate
pkeep = tf.placeholder(tf.float32, name='pkeep')  # dropout parameter
batchsize = tf.placeholder(tf.int32, name='batchsize')

# inputs
X = tf.placeholder(tf.uint8, [None, None], name='X')    # [ BATCHSIZE, SEQLEN ]
Xo = tf.one_hot(X, N_ITEMS, 1.0, 0.0)                   # [ BATCHSIZE, SEQLEN, N_ITEMS ], on-value=1.0, off=0.0
# expected outputs = same sequence shifted by 1 since we are trying to predict the next character
Y_ = tf.placeholder(tf.uint8, [None, None], name='Y_')  # [ BATCHSIZE, SEQLEN ]
Yo_ = tf.one_hot(Y_, N_ITEMS, 1.0, 0.0)                 # [ BATCHSIZE, SEQLEN, N_ITEMS ]
# input state
Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*N_LAYERS], name='Hin')  # [ BATCHSIZE, INTERNALSIZE * N_LAYERS]

# using a N_LAYERS=3 layers of GRU cells, unrolled SEQLEN=30 times
# dynamic_rnn infers SEQLEN from the size of the inputs Xo

onecell = rnn.GRUCell(INTERNALSIZE)
dropcell = rnn.DropoutWrapper(onecell, input_keep_prob=pkeep)
multicell = rnn.MultiRNNCell([dropcell]*N_LAYERS, state_is_tuple=False)
multicell = rnn.DropoutWrapper(multicell, output_keep_prob=pkeep)
Yr, H = tf.nn.dynamic_rnn(multicell, Xo, dtype=tf.float32, initial_state=Hin)
# Yr: [ BATCHSIZE, SEQLEN, INTERNALSIZE ]
# H:  [ BATCHSIZE, INTERNALSIZE*N_LAYERS ] # this is the last state in the sequence

H = tf.identity(H, name='H')  # just to give it a name

# Softmax layer implementation:
# Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, N_ITEMS ] => [ BATCHSIZE x SEQLEN, N_ITEMS ]
# then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
# From the readout point of view, a value coming from a cell or a minibatch is the same thing

# flatten rnn output
Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])    # [ BATCHSIZE x SEQLEN, INTERNALSIZE ]
# go from internal size (from rnn) to n_items size
Ylogits = layers.linear(Yflat, N_ITEMS)     # [ BATCHSIZE x SEQLEN, N_ITEMS ]
# flatten expected outputs
Yflat_ = tf.reshape(Yo_, [-1, N_ITEMS])     # [ BATCHSIZE x SEQLEN, N_ITEMS ]
# calculate loss of calculations
loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)  # [ BATCHSIZE x SEQLEN ]
# unflatten loss
loss = tf.reshape(loss, [batchsize, -1])      # [ BATCHSIZE, SEQLEN ]
# get the index of the highest scoring prediction through Y
Yo = tf.nn.softmax(Ylogits, name='Yo')        # [ BATCHSIZE x SEQLEN, N_ITEMS ]
Y = tf.argmax(Yo, 1)                          # [ BATCHSIZE x SEQLEN ]
Y = tf.reshape(Y, [batchsize, -1], name="Y")  # [ BATCHSIZE, SEQLEN ]
# training
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# stats for display
# avg loss across sequences in the batch
seqloss = tf.reduce_mean(loss, 1)
# avg loss across the batch
batchloss = tf.reduce_mean(seqloss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
loss_summary = tf.summary.scalar("batch_loss", batchloss)
acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
summaries = tf.summary.merge([loss_summary, acc_summary])

# Init Tensorboard stuff. This will save Tensorboard information into a different
# folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
# you can compare training and validation curves visually in Tensorboard.
timestamp = str(math.trunc(time.time()))
summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")

# Init for saving models. They will be saved into a directory named 'checkpoints'.
# Only the last checkpoint is kept.
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
saver = tf.train.Saver(max_to_keep=1)

# init
istate = np.zeros([BATCHSIZE, INTERNALSIZE*N_LAYERS])  # initial zero input state
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 0

# training loop
for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=1000):

    # train on one minibatch
    feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, pkeep: dropout_pkeep, batchsize: BATCHSIZE}
    _, y, ostate, smm = sess.run([train_step, Y, H, summaries], feed_dict=feed_dict)

    # save training data for Tensorboard
    summary_writer.add_summary(smm, step)

    # display a visual validation of progress (every 50 batches)
    if step % _50_BATCHES == 0:
        feed_dict = {X: x, Y_: y_, Hin: istate, pkeep: 1.0, batchsize: BATCHSIZE}  # no dropout for validation
        y, l, bl, acc = sess.run([Y, seqloss, batchloss, accuracy], feed_dict=feed_dict)
        txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)

    # run a validation step every 50 batches
    # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
    # so we cut it up and batch the pieces (slightly inaccurate)
    # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
    if step % _50_BATCHES == 0 and len(valitext) > 0:
        VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
        bsize = len(valitext) // VALI_SEQLEN
        txt.print_validation_header(len(codetext), bookranges)
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
        vali_nullstate = np.zeros([bsize, INTERNALSIZE*N_LAYERS])
        feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, pkeep: 1.0,  # no dropout for validation
                     batchsize: bsize}
        ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
        txt.print_validation_stats(ls, acc)
        # save validation data for Tensorboard
        validation_writer.add_summary(smm, step)

    # display a short text generated with the current weights and biases (every 150 batches)
    if step // 3 % _50_BATCHES == 0:
        txt.print_text_generation_header()
        ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
        rh = np.zeros([1, INTERNALSIZE * N_LAYERS])
        for k in range(1000):
            ryo, rh = sess.run([Yo, H], feed_dict={X: ry, pkeep: 1.0, Hin: rh, batchsize: 1})
            rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
            print(chr(txt.convert_to_alphabet(rc)), end="")
            ry = np.array([[rc]])
        txt.print_text_generation_footer()

    # save a checkpoint (every 500 batches)
    if step // 10 % _50_BATCHES == 0:
        saver.save(sess, 'checkpoints/rnn_train_' + timestamp, global_step=step)

    # display progress bar
    progress.step(reset=step % _50_BATCHES == 0)

    # loop state around
    istate = ostate
    step += BATCHSIZE * SEQLEN

