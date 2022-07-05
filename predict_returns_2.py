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

xgb_model_loaded = pickle.load(open("data/xgb_returns.pkl", "rb"))
X_recent = pickle.load(open("data/X_recent.pkl", "rb"))
timestamp_recent = pickle.load(open("data/timestamp_recent.pkl", "rb"))
timestamp_old = pickle.load(open("data/timestamp_old.pkl", "rb"))
lst = list(range(len(X_recent)))
#lst = [0,1,2,3,4]

# Predict the housing prices in certain timestamp
# output the sum for all houses with year-built less than the current year
def get_price_sum(i):
    print(i, dt.datetime.now(), "calculations started.")
    # copy the whole dataset
    test_time = [x for x in timestamp_recent if x <= X_recent.iloc[i, :]["Timestamp0"]]
    X_train = pd.concat([pd.DataFrame(X_recent.iloc[i, :]).T] * (len(timestamp_old) + len(test_time)),
                        ignore_index=True)
    X_train["Timestamp0"] = list(timestamp_old) + test_time
    y_train_pred = xgb_model_loaded.predict(X_train)
    print(i, dt.datetime.now(), "calculations ended.")
    return y_train_pred


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
    results = pool.map(func, lst, 5)

    pool.close()
    pool.join()

    # dump the results to a pkl file for next use
    pickle.dump(results, open('results.pkl', "wb"))
    print(dt.datetime.now(), "Time cost:", time.time() - st, "secs.")


if __name__ == "__main__":
    main()
