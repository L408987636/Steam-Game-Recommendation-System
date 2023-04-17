#!/usr/bin/env python
# coding: utf-8




import random
import pandas as pd
import numpy as np

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import implicit # The Cython library
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import coo_matrix





data = pd.read_csv('D:\HKU-Subject Material\SEM2\COMP7103\project1\data\steam_users_purchase_play.csv')
# Drop rows with missing values
data = data.dropna()



data



# Convert game names into numerical IDs
data['user_id'] = data['user'].astype("category").cat.codes
data['game_id'] = data['game_name'].astype("category").cat.codes

# Store the game names so we can refer to it later
game_names = data[['game_id', 'game_name']].drop_duplicates()
game_names['game_id'] = game_names.game_id.astype('double')
users = data[['user_id', 'user']].drop_duplicates()
users['user_id'] = users.user_id.astype('double')

# Drop any rows that have 0 plays
data = data.loc[data.hours != 0]



# Remove game with less than 30 users
game_freq = pd.DataFrame(data.groupby('game_id').size(),columns=['count'])
game_freq.head()
threshold_play_freq = 30
popular_game_id = list(set(game_freq.query('count>=@threshold_play_freq').index))
data_popular_game = data[data.game_id.isin(popular_game_id)]



# Remove users with less than 10 games
user_cnt = pd.DataFrame(data.groupby('user_id').size(),columns=['count'])
user_cnt.head()
threshold_val = 10
active_user = list(set(user_cnt.query('count>=@threshold_val').index))

#upadte data_popular_game
data_popular_game_with_active_user = data_popular_game[data_popular_game.user_id.isin(active_user)]


uni_users = data_popular_game_with_active_user[['game_id']].drop_duplicates()
uni_users


# Participating data into train and test set
data_train, data_test = train_test_split(data_popular_game_with_active_user, test_size = 0.2,random_state=42)


# Create lists of all users, games and hour plays
users_train = list(np.sort(data_train.user_id.unique()))
games_train = list(np.sort(data_train.game_id.unique()))
plays_train = list(data_train.hours)


'''
CF with ALS:
We have our original matrix R of size u x i with our users, items and some type of feedback data. 
We then want to find a way to turn that into one matrix with users and hidden features of size u x f and one with items and hidden features of size f x i. 
In U and V we have weights for how each user/item relates to each feature. 
What we do is we calculate U and V so that their product approximates R as closely as possible: R ≈ U x V.

'''


# Get the rows and columns for our new matrix R
rows = data_train.user_id.astype(int)
cols = data_train.game_id.astype(int)
print(rows)


# Contruct a sparse matrix for our users and games containing number of hours plays
user_game_sparse = sparse.csr_matrix((plays_train, (rows, cols)))
# weight the matrix, both to reduce impact of users that have played the same game tfor very long time
# and to reduce the weight given to popular items
user_game_sparse = bm25_weight(user_game_sparse, K1=100, B=0.8).tocsr()
user_game_sparse


# Calculate sparsity
sparsity = np.count_nonzero(user_game_sparse.todense()) / user_game_sparse.todense().size
sparsity


# ALS algorithm built by ourselves
def implicit_als(sparse_data, alpha_val=40, iterations=10, lambda_val=0.1, features=10):
    
    """ Implementation of Alternating Least Squares with implicit data. We iteratively
    compute the user (x_u) and item (y_i) vectors using the following formulas:
 
    x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (Y.T * Cu * p(u))
    y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (X.T * Ci * p(i))
 
    Args:
        sparse_data (csr_matrix): Our sparse user-by-item matrix
 
        alpha_val (int): The rate in which we'll increase our confidence
        in a preference with more interactions.
 
        iterations (int): How many times we alternate between fixing and 
        updating our user and item vectors
 
        lambda_val (float): Regularization value
 
        features (int): How many latent features we want to compute.
    
    Returns:     
        X (csr_matrix): user vectors of size users-by-features
        
        Y (csr_matrix): item vectors of size items-by-features
     """
    # Calculate the confidence for each value in our data
    confidence = sparse_data * alpha_val
    
    # Get the size of user rows and item columns
    user_size, item_size = sparse_data.shape
    
    # We create the user vectors X of size users-by-features,
    # the item vectors Y of size items-by-features and randomly assign the values.
    X = sparse.csr_matrix(np.random.normal(size = (user_size, features)))
    Y = sparse.csr_matrix(np.random.normal(size = (item_size, features)))
    
    #Precompute I and lambda * I
    X_I = sparse.eye(user_size)
    Y_I = sparse.eye(item_size)
    
    I = sparse.eye(features)
    lI = lambda_val * I
    
    # Alternatively computing X and Y for n=iterations times. For each iteration we first compute X and then Y
    for i in xrange(iterations):
        #print( 'iteration %d of %d' % (i+1, iterations))
        
        # Precompute Y-transpose-Y and X-transpose-X
        yTy = Y.T.dot(Y)
        xTx = X.T.dot(X)

        # Loop through all users
        for u in xrange(user_size):

            # Get the user row.
            u_row = confidence[u,:].toarray() 

            # Calculate the binary preference p(u)
            p_u = u_row.copy()
            p_u[p_u != 0] = 1.0

            # Calculate Cu and Cu - I
            CuI = sparse.diags(u_row, [0])
            Cu = CuI + Y_I   # 1+ alpha*r_ui

            # Put it all together and compute the final formula
            yT_CuI_y = Y.T.dot(CuI).dot(Y)
            yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
            X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

    
        for i in xrange(item_size):

            # Get the item column and transpose it.
            i_row = confidence[:,i].T.toarray()

            # Calculate the binary preference p(i)
            p_i = i_row.copy()
            p_i[p_i != 0] = 1.0

            # Calculate Ci and Ci - I
            CiI = sparse.diags(i_row, [0])
            Ci = CiI + X_I

            # Put it all together and compute the final formula
            xT_CiI_x = X.T.dot(CiI).dot(X)
            xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
            Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)

    return X, Y


# The above method takes too long to find the results, so we use the implicit library to reduce running time
# Initialize the als model and fit it using the sparse user-game matrix
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, alpha = 15, iterations = 30, 
                                             calculate_training_loss = True, random_state = 42)
# Fit the model
model.fit(user_game_sparse.astype('double'))


item_matrix = model.item_factors
user_matrix = model.user_factors

pred = np.dot(user_matrix, item_matrix.T)

true = user_game_sparse.toarray()


rmse = np.sqrt(sum(sum((pred-true)**2)/true.size))
mae = sum(sum(np.abs(pred-true))/true.size)
print('RMSE: ',rmse)
print('MAE: ', mae) 


# Function for recommendation built by ourselves
def recommend(user_id, sparse_user_item, user_vecs, item_vecs, num_items=20):
    '''
    user_id: user_id after conversion
    sparse_user_item: the user-item sparse matrix
    user_vecs: the user-factor sparse matrix
    item_vecs: the item-factor sparse matrix
    num_items: Top N item to recommand
    '''

    user_interactions = sparse_user_item[user_id,:].toarray()

    user_interactions = user_interactions.reshape(-1) + 1
    user_interactions[user_interactions > 1] = 0

    rec_vector = user_vecs[user_id,:].dot(item_vecs.T).toarray()

    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions * rec_vector_scaled

    item_idx = np.argsort(recommend_vector)[::-1][:num_items]

    games = []
    scores = []

    for idx in item_idx:
        games.append(game_names.game_name.loc[game_names.game_id == idx].iloc[0])
        scores.append(recommend_vector[idx])

    recommendations = pd.DataFrame({'games': games, 'score': scores})

    return recommendations


# Make recommendation for a user
user_vecs = sparse.csr_matrix(model.user_factors)
item_vecs = sparse.csr_matrix(model.item_factors)
user_id = 1
recommendations = recommend(1, user_game_sparse, user_vecs, item_vecs, 10)
recommendations


# Use the recommondation function in Implicit lib
game_id,score = model.recommend(1, user_game_sparse[1], N=10)

games = []
scores = []

# Get game names from ids
for i in range(0,len(game_id)):
    row_idx = game_names[game_names.game_id == game_id[i]].index.tolist()[0]
    games.append(game_names['game_name'][row_idx])
    scores.append(score[i])

# Create a dataframe of game names and scores
recommendations = pd.DataFrame({'game': games, 'score': scores})
recommendations


def cal_precision(data_test_user,recommendations):
    cnt = 0
    for game_test in data_test_user.game_name:
        for game_pred in recommendations.game:
            if game_test == game_pred:
                cnt += 1
    precision = cnt/min(len(data_test_user),len(recommendations))
    return precision


# calculate precision
user_list = set(data_train.user_id)&set(data_test.user_id)
precision = 0

for uid in user_list:
    game_id,score = model.recommend(uid, user_game_sparse[uid], N=10)
    games = []
    scores = []
    precision_i = 0

    for i in range(0,len(game_id)):
        row_idx = game_names[game_names.game_id == game_id[i]].index.tolist()[0]
        games.append(game_names['game_name'][row_idx])
        scores.append(score[i])

    recommendations = pd.DataFrame({'game': games, 'score': scores})
    data_test_user = data_test[data_test.user_id==uid]
    precision_i = cal_precision(data_test_user,recommendations)
    precision += precision_i

avg_precision = precision/(len(user_list))


recall = 0
for uid in user_list:
    game_id,score = model.recommend(uid, user_game_sparse[uid], N=5)
    games = []
    scores = []
    recall_i = 0
    TP = 0
    FN = 0

    for i in range(0,len(game_id)):
        row_idx = game_names[game_names.game_id == game_id[i]].index.tolist()[0]
        games.append(game_names['game_name'][row_idx])
        scores.append(score[i])
    
    data_test_user = data_test[data_test.user_id==uid].game_name.tolist()
    for i in range(0,len(games)):
        if games[i] in data_test_user:
            TP += 1
    for i in range(0,len(data_test_user)):
        if data_test_user[i] not in games:
            FN += 1
    recall_i = TP/(TP+FN)
    recall += recall_i
avg_recall = recall/(len(user_list))


F1_score = 2*avg_precision*avg_recall/(avg_precision+avg_recall)


print('Precision: ', avg_precision)
print('Recall: ', avg_recall)
print('F1_score: ', F1_score)





