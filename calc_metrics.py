import yaml
import numpy as np
import sys
import os

from utils.calc_forgetting import calc_forgetting

def load_yaml_to_y(fname):
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)

    history = np.array(data['history'])
    no_trials = history.shape[-1]
    T = len(history)
    y = np.zeros((no_trials, T, T))

    def to_float(x):
        while isinstance(x, list):
            if len(x) == 0:
                return 0.0
            x = x[0]
        try:
            return float(x)
        except Exception:
            return 0.0

    for trial in range(no_trials):
        for t in range(T):  # after training task t
            for i in range(t + 1):  # tasks seen so far
                val = to_float(history[i, t, trial])
                y[trial, t, i] = val
    
    # y = y.transpose((0,2,1))
    return y

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calc.py <path_to_yaml>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

    y = load_yaml_to_y(path)

    mean_fgt, std_fgt = calc_forgetting(y)
    print(f"Incremental Forgetting = {mean_fgt:.4f} ± {std_fgt:.4f}")                                                                    