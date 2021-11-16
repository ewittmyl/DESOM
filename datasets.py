"""
Dataset loading functions

@author Florent Forest
@version 1.0
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(dataset_name, validation_size):
    # Load data set
    data = pd.read_csv(dataset_name)
    
    x = data.values
    if validation_size == 0: # no validation
        return (x, None), (None, None)
    elif (validation_size > 0) and (validation_size < 1): # validation fraction
        x_train, x_test = train_test_split(x)
        return (x_train, None), (x_test, None)
    else:
        raise ValueError("Argument 'validation_size' should be in [0,1).") # invalid fraction