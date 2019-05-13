import pickle
import pandas as pd
import numpy as np
from helper import *

def concat_dicts(dict1,dict2, csv_filepath):
    dict1["Id"] += dict2["Id"]
    dict1["Predicted"] += dict2["Predicted"]
    submission = pd.DataFrame(dict1)
    submission.to_csv(csv_filepath, index=False)


results_a = load_pickle("angles.pickle")
results_d = load_pickle("distance.pickle")
concat_dicts(results_a, results_d, "mysubmission.csv")
