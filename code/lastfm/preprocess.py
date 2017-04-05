import dateutil.parser
import pickle
import os
import time

# There should be a total of 19 150 868 songlistenings

runtime = time.time()

home = os.path.expanduser('~')
DATASET_DIR = home + '/datasets/lastfm-dataset-1K'
DATASET_FILE = DATASET_DIR + '/userid-timestamp-artid-artname-traid-traname.tsv'
DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/lastfm_converted_timestamps.pickle'
DATASET_USER_ARTIST_MAPPED = DATASET_DIR + '/lastfm_user_artist_mapped.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/lastfm_user_sessions.pickle'
DATASET_PLAIN_RNN = DATASET_DIR + '/lastfm_as_list_for_plain_rnn.pickle'

# The maximum amount of time between two consequtive events before they are
# considered belonging to different sessions. Remember to adjust for time 
# to listen to a song. 30 minutes should be reasonable.
SESSION_TIMEDELTA = 1200 # seconds. 60*20=1200 (20 minutes)

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

        if timedelta < SESSION_TIMEDELTA:
            # new event belongs to current session
            current_session.append(event)
        else:
            # new event belongs to new session
            current_session = [event]
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

    # Remove users with only 1 session
    # Find users with only 1 session first
    to_be_removed = []
    for k, v in new_user_sessions.items():
        if len(v) == 1:
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
                a = session[i][2]
                if a not in art:
                    art[a] = len(art)
                session[i][2] = art[a]
    

    save_pickle(nus, DATASET_USER_SESSIONS)

''' Take sessions from mapping (user->sessions) and store a sessions in a flat 
    list of sessions. Used for the plain RNN which has no concern for the 
    actual user.
'''
def create_flat_list_of_user_sessions():
    user_sessions = load_pickle(DATASET_USER_SESSIONS)
    flat_data = []
    
    for _, us in user_sessions.items():
        for session in us:
            s = []
            for event in session:
                s.append(event[2])

            flat_data.append(s)

    save_pickle(flat_data, DATASET_PLAIN_RNN)


# It takes a lot of time to convert the ISO timestamps in the dataset
# to unix timestamps (easier to work with). So first step is to convert these
# without modifying the dataset, and storing, so we can save time in the
# future.
if not file_exists(DATASET_W_CONVERTED_TIMESTAMPS):
    print("converting timestamps... ")
    convert_timestamps()

# Next we map the artist_ids to numbers (starting from 0), since this can be 
# converted to 1-HOT encodings.
if not file_exists(DATASET_USER_ARTIST_MAPPED):
    print("mapping user and artist IDs to labels...")
    map_user_and_artist_id_to_labels()

# Collect events into sessions, and assign the sessions to their user. Also
# remove sessions with only one event.
if not file_exists(DATASET_USER_SESSIONS):
    print("sorting sessions to users...")
    sort_and_split_usersessions()

# Put all sessions in a flat list, which is more appropriate for the plain
# RNN.
if not file_exists(DATASET_PLAIN_RNN):
    print("creating flat version for plain RNN...")
    create_flat_list_of_user_sessions()

print("Runtime:", str(time.time()-runtime))
