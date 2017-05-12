import pickle
import os
import time

runtime = time.time()

home = os.path.expanduser('~')
if home == '/root':
    home = '/notebooks'

DATASET_DIR = home + '/datasets/instacart'
ORDERS_FILE = DATASET_DIR + '/orders.csv'
ORDER_PRODUCTS_PRIOR = DATASET_DIR + '/order_products__prior.csv'
DATASET_USER_ORDER_PRODUCTS_COMBINED = DATASET_DIR + '/2_user_sessions_combined.pickle'
DATASET_USER_SESSIONS = DATASET_DIR + '/3_user_sessions.pickle'
DATASET_TRAIN_TEST_SPLIT = DATASET_DIR + '/4_train_test_split.pickle'

MAX_SESSION_LENGTH = 20
MAX_SESSION_LENGTH_PRE_SPLIT = MAX_SESSION_LENGTH * 2 # Within this limit we can split a session into two
MINIMUM_REQUIRED_SESSIONS = 3
PAD_VALUE = 0

def file_exists(filename):
    return os.path.isfile(filename)

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb'))

def save_pickle(data_object, data_file):
    pickle.dump(data_object, open(data_file, 'wb'))

if not file_exists(DATASET_USER_ORDER_PRODUCTS_COMBINED):
    user_orders = {}
    order_info = {}

    print("Reading user orders.")
    # read user orders
    with open(ORDERS_FILE, 'rt', buffering=10000, encoding='utf8') as orders:
        next(orders) # skip header
        # order_id  user_id  eval_set  order_number  order_dow  order_hod  days_since_prior_order
        for line in orders:
            line = line.rstrip()
            line = line.split(',')
            order_id     = int(line[0])
            user_id      = int(line[1])
            eval_set     = line[2]
            order_number = int(line[3])
            order_dow    = int(line[4])
            order_hod    = int(line[5])
    
            # Only care about the prior dataset for now
            if eval_set != 'prior':
                continue
    
            if len(line) == 7:
                days_since_prior_order = line[6]
            else:
                days_since_prior_order = 35 # We don't know, but more than a month at least
    
            if user_id not in user_orders:
                user_orders[user_id] = []
    
            user_orders[user_id].append([order_number, order_id])
            order_info[order_id] = [order_dow, order_hod, days_since_prior_order]

            if len(user_orders.keys()) > 20000:
                break
    
    print("Sorting user orders.")
    # ensure that orders are sorted for each user
    for user_id in user_orders.keys():
        orders = user_orders[user_id]
        user_orders[user_id] = sorted(orders, key=lambda x: x[0])
    
    print("Reading order products.")
    order_products = {}
    # read order products
    with open(ORDER_PRODUCTS_PRIOR, 'rt', buffering=10000, encoding='utf8') as op:
        next(op) # skip header
        # order_id  product_id  add_to_cart_order  reordered
        for line in op:
            line = line.rstrip()
            line = line.split(',')
            order_id          = int(line[0])
            product_id        = int(line[1])
            add_to_cart_order = int(line[2])
            
            if order_id not in order_products:
                order_products[order_id] = []
    
            order_products[order_id].append([add_to_cart_order, product_id])
    
    print("Sorting order products.")
    # ensure that products are sorted in the correct order for each user
    for order_id in order_products.keys():
        products = order_products[order_id]
        order_products[order_id] = sorted(products, key=lambda x: x[0])
    
    print("Connecting orders and users.")
    # connect product orders and users
    for user_id in user_orders.keys():
        # get a list of order_ids (sorted) for this user
        orders = [x[1] for x in user_orders[user_id]]
        # replace order_ids with the actual sequence of products (aka the session)
        for i in range(len(orders)):
            order_id = orders[i]
            orders[i] = order_products[order_id]
        user_orders[user_id] = orders

    # get some nais statistics
    print("Calculating some statistics.")
    session_lengths = [0]*155
    products = {}
    n_sessions = 0
    longest = 0
    shortest = 999999
    for user, orders in user_orders.items():
        n_sessions += len(orders)
        for order in orders:
            if len(order) > longest:
                longest = len(order)
            if len(order) < shortest:
                shortest = len(order)
            session_lengths[len(order)] += 1
            for x in order:
                product = x[1]
                if product not in products:
                    products[product] = True
    print("num products (labels):", len(products.keys()))
    print("num users:", len(user_orders.keys()))
    print("num sessions:", n_sessions)
    print("shortest session:", shortest)
    print("longest session:", longest)
    print()
    print("SESSION LENGTHS:")
    for i in range(len(session_lengths)):
        print(i, session_lengths[i])

    save_pickle(user_orders, DATASET_USER_ORDER_PRODUCTS_COMBINED)

if not file_exists(DATASET_USER_SESSIONS):
    user_sessions = load_pickle(DATASET_USER_ORDER_PRODUCTS_COMBINED)

    # Split sessions
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

    for k in user_sessions.keys():
        sessions = user_sessions[k]
        user_sessions[k] = perform_session_splits(sessions)
    
    # Remove too short sessions
    for k in user_sessions.keys():
        sessions = user_sessions[k]
        user_sessions[k] = [s for s in sessions if len(s)>1]

    # Remove users with too few sessions
    users_to_remove = []
    for u, sessions in user_sessions.items():
        if len(sessions) < MINIMUM_REQUIRED_SESSIONS:
            users_to_remove.append(u)

    for u in users_to_remove:
        del(user_sessions[u])

    # Remap user_ids
    if len(users_to_remove) > 0:
        nus = {}
        for k, v in user_sessions.items():
            nus[len(nus)] = user_sessions[k]

    # Remap labels
    lab = {}
    for k, v in nus.items():
        for session in v:
            for i in range(len(session)):
                l = session[i][1]
                if l not in lab:
                    lab[l] = len(lab)+1
                session[i][1] = lab[l]
    
    save_pickle(nus, DATASET_USER_SESSIONS)

if not file_exists(DATASET_TRAIN_TEST_SPLIT):
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

        length_to_pad = MAX_SESSION_LENGTH - len(session)
        padding = [[PAD_VALUE, PAD_VALUE]] * length_to_pad
        session += padding
        return session

    def pad_sequences(dataset):
        for k, v in dataset.items():
            for session_index in range(len(v)):
                dataset[k][session_index] = create_padded_sequence(dataset[k][session_index])

    dataset = load_pickle(DATASET_USER_SESSIONS)
    trainset = {}
    testset  = {}

    for k, v in dataset.items():
        n_sessions = len(v)
        split_point = int(0.8 * n_sessions)

        if split_point < 2:
            raise ValueError('WTF? so few sessions?')

        trainset[k] = v[:split_point]
        testset[k] = v[split_point:]

    # Also need to know the session lengths
    train_session_lengths = get_session_lengths(trainset)
    test_session_lengths = get_session_lengths(testset)

    # Finally, pad all sequences before storing everything
    pad_sequences(trainset)
    pad_sequences(testset)

    # Put everything in a dict, and just store the dict with pickle
    pickle_dict = {}
    pickle_dict['trainset'] = trainset
    pickle_dict['testset'] = testset
    pickle_dict['train_session_lengths'] = train_session_lengths
    pickle_dict['test_session_lengths'] = test_session_lengths

    save_pickle(pickle_dict , DATASET_TRAIN_TEST_SPLIT)

print("runtime:", str(time.time() - runtime))
