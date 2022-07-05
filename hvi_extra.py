import pandas as pd
import datetime as dt
import numpy as np
import pickle
from functools import partial
import psutil
from multiprocessing import Pool
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '3'}

# load xgb model and training data for sale price prediction
xgb_file_name = "xgb_reg.pkl"
X_file_name = "X.pkl"
xgb_model_loaded = pickle.load(open(xgb_file_name, "rb"))
X = pickle.load(open(X_file_name, "rb"))

# generate the timetable
# start is the earliest time
# end is the latest time
# The timetable contains continuing time for each day including the month and year information for future average.
#start, end = X.Timestamp0.min(), X.Timestamp0.max()
#start, end = pd.to_datetime(start, unit='s', origin='unix'), pd.to_datetime(end, unit='s', origin='unix')

start, end = X['SALEDATE'].min(), X['SALEDATE'].max()
df_time = pd.DataFrame(columns=['SALEDATE'],
                       data=[start.to_pydatetime() + dt.timedelta(days=i)
                             for i in range((end.to_pydatetime() - start.to_pydatetime()).days)])
#df_time['Timestamp0'] = df_time['SALEDATE'].values.astype(np.int64) // 10 ** 9
df_time['year'] = df_time.SALEDATE.dt.year
df_time['month'] = df_time.SALEDATE.dt.month
df_time['Timestamp0'] = df_time['SALEDATE'].apply(lambda x: x.timestamp())
time_list = df_time.index.to_list()
#time_list =[0,1,2]
X = X.drop(columns=['SALEDATE'])

# Predict the housing prices in certain timestamp
# output the sum for all houses with year-built less than the current year
def get_price_sum(i):
    print(i, dt.datetime.now(), "calculations started.")
    # copy the whole dataset
    X_test = X.copy()
    # only change the sale date feature with certain timestamp
    X_test['Timestamp0'] = df_time['Timestamp0'].iloc[i]
    # keep the houses with year-built less or equal to the year associated with the timestamp
    X_test = X_test[X_test['YEARBLT'] <= df_time['year'].iloc[i]]
    y_test = xgb_model_loaded.predict(X_test)
    print(dt.datetime.now(), "calculations ended.")

    # return the sum of the housing price predictions
    return np.exp(y_test).sum()


# Using multiple CPU cores to parallelize the housing price predictions for different timestamps
def main():
    st = time.time()

    # set up the partial function for multiple CPUs calculations
    func = partial(get_price_sum)

    # check the logical cpu core number
    cpu_cores = psutil.cpu_count(logical=True)
    print(dt.datetime.now(), cpu_cores, "cpu cores.")

    # start to build the multiple
    # create a multiprocessing.pool.Pool object
    pool = Pool(processes=cpu_cores - 10, maxtasksperchild=20)

    print(dt.datetime.now(), "calculations started.")

    # call the Pool.map with the partial function and the iterable(the time range from the start to the end)
    results = pool.map(func, time_list, 5)

    pool.close()
    pool.join()

    # dump the results to a pkl file for next use
    pickle.dump(results, open('results_sum.pkl', "wb"))
    print(dt.datetime.now(), "Time cost:", time.time() - st, "secs.")


if __name__ == "__main__":
    main()
