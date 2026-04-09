import pandas as pd
from scipy.stats import ks_2samp
import subprocess

old = pd.read_csv("data/train_data.csv")
new = pd.read_csv("data/churn.csv").drop('Churn', axis=1)

def check_drift():
    for col in old.columns:
        stat, p = ks_2samp(old[col], new[col])
        if p < 0.05:
            return True
    return False

if check_drift():
    print("DRIFT DETECTED")
    print('RETRAINING MODEL')
    subprocess.run(["python", "train.py"])
else:
    print('NO DRIFT DETECTED')