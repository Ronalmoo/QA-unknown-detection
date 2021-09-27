import json
import numpy as np
import pandas as pd
import string
import re
from sklearn import metrics

from pathlib import Path

def temperature_scaling(z, T):
    z = np.array(z)
    z = z / T
    max_z = np.max(z)
    exp_z = np.exp(z - max_z)
    sum_exp_z = np.sum(exp_z)
    y = exp_z / sum_exp_z
    return y


if __name__ == "__main__":
    print(temperature_scaling([1, 2, 3], 2) )
    # [0.23023722 0.32132192 0.44844086]