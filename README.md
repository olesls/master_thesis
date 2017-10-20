# master_thesis
Code for my master thesis: Inter-/Intra-session Recurrent Neural Network for Session-based Recommender Systems

# Requirements
Python 3  
Tensorflow for Python 3 (preferrably with GPU support)  
Numpy  
Pickle  

# Usage

## Datasets
You need to download one of the datasets (Sub-reddit, Last.fm, or Instacart).  
They can be found here:  
  
[Sub-reddit](https://www.kaggle.com/colemaclean/subreddit-interactions)  
[Last.fm](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html)  
[Instacart](https://www.instacart.com/datasets/grocery-shopping-2017)  
  
The code assumes that the datasets are located in:  
~/datasets/subreddit/  
~/datasets/lastfm/  
~/datasets/instacart/  
But you can of course store them somewhere else and change the path in the code.  
  
## Preprocessing
The sub-reddit and Last.fm dataset have a very similar structure, so the preprocessing of both these is done with `preprocess.py`. In this file you must specify which dataset you want to run preprocessing on (`dataset = reddit` or `dataset = lastfm`), and also make sure that the path to the dataset is correct.  
  
Preprocessing is done in multiple stages, and the data is stored after each stage. This is to make it possible to change the preprocessing wihout having to do all the stages each time. This is especially useful for the Last.fm dataset, where it takes some time to convert the original timestamps.  
  
The first stage is to convert timestamps and ensure that the data in sub-reddit and Last.fm is in the same format. Each dataset has its own method to do this. After this, both datasets use the same methods.  
The second stage is to map items/labels to integers.  
The third stage split the data into sessions.  
The fourth stage splits the data into a training set and a test set.  
The fifth stage creates a hold-one-out version of the training/test split, for testing with BPR-MF.  
  
The data in the Instacart dataset is on a very different format than the other two, therefore it has its own preprocessing code, `preprocess_instacart.py`. The stages are similar, but Instacart does not have timestamps, and is sorted into sessions as is.  
  
In `preprocess.py`, you can specify the time delta between user interactions that is the limit deciding whether two actions belong to the same session or two different ones. This is done by specifying `SESSION_TIMEDELTA` in seconds. Additionally, you can specify the maximum number of user interactions for a session with `MAX_SESISON_LENGTH`. Longer sessions are splitted, to accomodate this limit.  
`MAX_SESSION_LENGTH_PRE_SPLIT` defines the longest session length that will be accepted. I.e. sessions longer than this limit are discarded as outliers, and sessions within this limit are kept and potentially splitted if needed.  
`PAD_VALUE` specifies the integer used when padding short sessions to the maximum length. Tensorflow requires all sessions to have the same length, which is why shorter sessions must be padded. There should be no reason to change this value, and doing so will probably cause a lot of problems.  
  
After running preprocessing on a dataset, the resulting training and testing set are stored in a pickle file, `4_train_test_split.pickle`, in the same directory as the dataset.   
  
If you want to use a different dataset, you can either do all the preprocessing yourself, or try to configure `preprocess.py` if your dataset is on a format similar to sub-reddit and Last.fm. In any case, I suggest looking at `preprocess.py` to see how the data needs to be formatted after the preprocessing.  
  
# Running the RNN models
`train_rnn.py` and `train_ii_rnn.py` are the files for training and testing the standalone intra-session RNN and the II-RNN respectively. They are pretty similar, but `train_ii_rnn.py` has some additional parameters, so we explain that one.   
  
You need to specify the `dataset` variable, in the code, to the dataset you want to run the model on. Preprocessing need to have been performed on that dataset. If you don't have the dataset stored in `~/datasets/...`, you must set the `dataset_path` variable accordingly instead.  
  
The `do_training` and `save_best` variables can be toggled. `do_training` decides whether to run training. If this is off, only testing is performed. `save_best` is used to decide whether the model should be saved after an epoch of training if the training resulted in better performance on the test set. Recall@5 is used to compare models between epochs.  
  
Testresults are stored in `testlog/`.  
  
### Parameters
`use_last_hidden_state` defines whether to use last hidden state or average of embeddings as session representation.  
`BATCHSIZE` defines the number of sessions in each mini-batch.  
`ST_INTERNALSIZE` defines the number of nodes in the intra-session RNN layer (ST = Short Term)  
`LT_INTERNALSIZE` defines the number of nodes in the inter-session RNN layer. These two depends on each other and needs to be the same size as each other and as the embedding size. If you want to use different sizes, you probably need to change the model as well, or at least how session representations are created.  
`learning_rate` is what you think it is.  
`dropout_pkeep` is the propability to keep a random node. So setting this value to 1.0 is equivalent to not using dropout.  
`MAX_SESSION_REPRESENTATIONS` defines the maximum number of recent session representations to consider.  
`MAX_EPOCHS` defines the maximum number of training epochs before the program terminates. It is no problem to manually terminate the program while training/testing the model and continue later if you have `save_best = True`. But notice that when you start the program again, it will load and continue training the last saved (best) model. Thus, if you are currently on epoch #40, and the last best model was achieved at epoch #32, then the training will start from epoch #32 again when restarting.  
`N_LAYERS` defines the number of GRU-layers used in the intra-session RNN layer.  
`SEQLEN` should be set to the `MAX_SESSION_LENGTH - 1` (from preprocessing). This is the length of sessions (with padding), (minus one since the last user interactions is only used as a label for the previous one).  
`TOP_K` defines the number of items the model produces in each recommendation.  
  
### VRAM limitation
Because of `with tf.device(cpu[0]):` at the start of the model setup, embeddings are stored in (CPU) RAM. This is done because the embeddings require a decent amount of memory. If you have more than ~10 GB of (GPU) VRAM, you can try to change this to `with tf.device(gpu[0]):` to put everything in VRAM. This has not been tested but should result in a reduced runtime. The amount of memory required depends on the dataset. You can use `watch -n 1 nvidia-smi` to monitor memory usage of the GPU.  

# Other files
`baselines.py` contains implementation of baseline models. In the bottom of the file you can uncomment the models that you want to run.  
The `utils*.py` files are used by the `train*rnn.py` files to manage the datasets. The utils files takes control of retrieving mini-batches, logging results, storing session representations and some other stuff.

`test_util.py` is used by all models to score recommendations. It creates a first-n score. So remember that the first-n score for the highest value of n is equal to the overall score.  

# Other questions
If anything is unclear or there are any problems, just send me a message :)
