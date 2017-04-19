import dateutil.parser
import pickle
import os
import time

# There should be a total of ~19 150 868 songlistenings

runtime = time.time()

home = os.path.expanduser('~')
DATASET_DIR = home + '/datasets/lastfm-dataset-1K'
DATASET_FILE = DATASET_DIR + '/userid-timestamp-artid-artname-traid-traname.tsv'
DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/lastfm_1_converted_timestamps.pickle'
DATASET_USER_ARTIST_MAPPED = DATASET_DIR + '/lastfm_2_user_artist_mapped.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/lastfm_3_user_sessions.pickle'
DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + '/lastfm_4_train_test_split.pickle'

# The maximum amount of time between two consequtive events before they are
# considered belonging to different sessions. Remember to adjust for time 
# to listen to a song. 30 minutes should be reasonable.
SESSION_TIMEDELTA = 1200    # seconds. 60*20=1200 (20 minutes)
MAX_SESSION_LENGTH = 20     # maximum number of events in a session
MINIMUM_REQUIRED_SESSIONS = 3 # The dual-RNN should have minimum 2 two train + 1 to test
PAD_VALUE = 0   # Doesn't really matter, since these values are ignored, but must be same as a valid label


def file_exists(filename):
    return os.path.isfile(filename)

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

def convert_timestamps():
    dataset_list = []
    with open(DATASET_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
        for line in dataset:
            line = line.split('\t')
            user_id     = line[0]
            timestamp   = (dateutil.parser.parse(line[1])).timestamp()
            artist_id   = line[2]
            # We will not use the rest of the information for now
            #artist_name = line[3]
            #track_id    = line[4]
            #track_name  = line[5]
            dataset_list.append( [user_id, timestamp, artist_id] )

    save_pickle(dataset_list, DATASET_W_CONVERTED_TIMESTAMPS)

def map_user_and_artist_id_to_labels():
    dataset_list = load_pickle(DATASET_W_CONVERTED_TIMESTAMPS)
    artist_map = {}
    user_map = {}
    artist_id = ''
    user_id = ''
    for i in range(len(dataset_list)):
        user_id = dataset_list[i][0]
        artist_id = dataset_list[i][2]
        
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        if artist_id not in artist_map:
            artist_map[artist_id] = len(artist_map)
        
        dataset_list[i][0] = user_map[user_id]
        dataset_list[i][2] = artist_map[artist_id]
    
    # All artist_id (hash strings) have been repalced with numeric labels.
    # Save to pickle file
    save_pickle(dataset_list, DATASET_USER_ARTIST_MAPPED)

def split_single_session(session):
    splitted = [session[i:i+MAX_SESSION_LENGTH] for i in range(0, len(session), MAX_SESSION_LENGTH)]
    if len(splitted[-1]) < 2:
        del splitted[-1]

    return splitted

def perform_session_splits(sessions):
    splitted_sessions = []
    for session in sessions:
        splitted_sessions += split_single_session(session)

    return splitted_sessions

def split_long_sessions(user_sessions):
    for k, v in user_sessions.items():
        user_sessions[k] = perform_session_splits(v)


''' Splits sessions according to inactivity (time between two consecutive 
    actions) and assign sessions to their user. Sessions should be sorted, 
    both eventwise internally and compared to other sessions, but this should 
    be automatically handled since the dataset is presorted
'''
def sort_and_split_usersessions():
    dataset_list = load_pickle(DATASET_USER_ARTIST_MAPPED)
    user_sessions = {}
    current_session = []
    for event in dataset_list:
        user_id = event[0]
        timestamp = event[1]
        artist = event[2]
        
        # if new user -> new session
        if user_id not in user_sessions:
            user_sessions[user_id] = []
            current_session = []
            user_sessions[user_id].append(current_session)
            current_session.append(event)
            continue

        # it is an existing user: is it a new session?
        # we also know that the current session contains at least one event
        # NB: Dataset is presorted from newest to oldest events
        last_event = current_session[-1]
        last_timestamp = last_event[1]
        timedelta = last_timestamp - timestamp

        new_event = [timestamp, artist]

        if timedelta < SESSION_TIMEDELTA:
            # new event belongs to current session
            current_session.append(new_event)
        else:
            # new event belongs to new session
            current_session = [new_event]
            user_sessions[user_id].append(current_session)

    # Remove sessions that only contain one event
    # Bad to remove stuff from the lists we are iterating through, so create 
    # a new datastructure and copy over what we want to keep
    new_user_sessions = {}
    for k in user_sessions.keys():
        if k not in new_user_sessions:
            new_user_sessions[k] = []

        us = user_sessions[k]
        for session in us:
            if len(session) > 1:
                new_user_sessions[k].append(session)

    # Split too long sessions, before removing user with too few sessions
    #  because splitting can result in more sessions.
    split_long_sessions(new_user_sessions)

    # Remove users with less than 3 session
    # Find users with less than 3 sessions first
    to_be_removed = []
    for k, v in new_user_sessions.items():
        if len(v) < MINIMUM_REQUIRED_SESSIONS:
            to_be_removed.append(k)
    # Remove the users we found
    for user in to_be_removed:
        new_user_sessions.pop(user)


    # Do a remapping to account for removed data
    print("remapping to account for removed data...")

    # remap users
    nus = {}
    for k, v in new_user_sessions.items():
        nus[len(nus)] = new_user_sessions[k]
    
    # remap artistIDs
    art = {}
    for k, v in nus.items():
        for session in v:
            for i in range(len(session)):
                a = session[i][1]
                if a not in art:
                    art[a] = len(art)+1
                session[i][1] = art[a]

    save_pickle(nus, DATASET_USER_SESSIONS)


def get_session_lengths(dataset):
    session_lengths = {}
    for k, v in dataset.items():
        session_lengths[k] = []
        for session in v:
            session_lengths[k].append(len(session)-1)

    return session_lengths

def create_padded_sequence(session):
    if len(session) == MAX_SESSION_LENGTH:
        return session

    dummy_timestamp = 0
    dummy_label = 0
    length_to_pad = MAX_SESSION_LENGTH - len(session)
    padding = [[dummy_timestamp, dummy_label]] * length_to_pad
    session += padding
    return session

def pad_sequences(dataset):
    for k, v in dataset.items():
        for session_index in range(len(v)):
            dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])

# Splits the dataset into a training and a testing set, by extracting the last
# sessions of each user into the test set
def split_to_training_and_testing():
    dataset = load_pickle(DATASET_USER_SESSIONS)
    trainset = {}
    testset = {}

    for k, v in dataset.items():
        n_sessions = len(v)
        split_point = int(0.8*n_sessions)
        
        # runtime check to ensure that we have enough sessions for training and testing
        if split_point < 2:
            raise ValueError('User '+str(k)+' with '+str(n_sessions)+""" sessions, 
                resulted in split_point: '+str(split_point)+' which gives too 
                few training sessions. Please check that data and preprocessing 
                is correct.""")
        
        trainset[k] = v[:split_point]
        testset[k] = v[split_point:]

    # Also need to know session lengths for train- and testset
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)

    # Finally, pad all sequences before storing everything
    pad_sequences(trainset)
    pad_sequences(testset)

    # Put everything we want to store in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset
    pickle_dict['testset'] = testset
    pickle_dict['train_session_lengths'] = train_session_lengths
    pickle_dict['test_session_lengths'] = test_session_lengths
    
    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT)


if not file_exists(DATASET_W_CONVERTED_TIMESTAMPS):
    print("Converting timestamps.")
    convert_timestamps()

if not file_exists(DATASET_USER_ARTIST_MAPPED):
    print("Mapping user and artist IDs to labels.")
    map_user_and_artist_id_to_labels()

if not file_exists(DATASET_USER_SESSIONS):
    print("Sorting sessions to users.")
    sort_and_split_usersessions()

if not file_exists(DATASET_TRAIN_TEST_SPLIT):
    print("Splitting dataset into training and testing sets.")
    split_to_training_and_testing()


print("Runtime:", str(time.time()-runtime))
