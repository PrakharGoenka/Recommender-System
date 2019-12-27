import pandas as pd
from time import time

# import all classes from recommenders directory
from recommenders.collaborative import Collaborative
from recommenders.collab_with_baseline import CollaborativeWB
from recommenders.data import *

def main():

    '''
        Initialize objects for all classes (or maybe make static methods);
        and call functions to
        (1) get recommendations
        (2) calculate RMSE
        (3) calculate MAE
        (4) execution time
        for each of the recommender types
    '''
    # get movie details' table
    movie_ind = movie_index()

    # input user id
    user_id = int(input('Enter user id \n'))

    # recommendations using collaborative filtering
    cb = Collaborative()
    # Start execution time
    start = time()
    movie_list = cb.get_recommendation(user_id)
    movie_list = pd.merge(movie_list, movie_ind, on = 'movieId', how = 'inner')
    # End execution time
    end = time()

    print('Recommendations using collaborative filtering for the user are \n', movie_list)
    # print('RMSE is\n', (cb.get_rmse()))
    # print('MAE is\n', (cb.get_mae()))
    print('Time Taken is {}'.format(end - start))

    # recommendations using collaborative filtering with baseline
    cwb = CollaborativeWB()
    # Start execution time
    start = time()
    movie_list = cwb.get_recommendation(user_id)
    movie_list = pd.merge(movie_list, movie_ind, on = 'movieId', how = 'inner')
    # End execution time
    end = time()
    print('Recommendations using collaborative filtering with baseline for the user are \n', movie_list)
    # print('RMSE is\n', cwb.get_rmse())
    # print('MAE is\n', cwb.get_mae())
    print('Time Taken is {}'.format(end - start))


if __name__ == '__main__':
    main()
    