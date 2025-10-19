import yaml
import numpy as np
import sys
import os

from utils.calc_forgetting import calc_forgetting

def load_yaml_to_y(fname):
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)

    history = data['history']
    T = len(history)               # number of training steps / tasks
    R = len(history[0])            # number of runs
    y = np.zeros((R, T, T))        # allocate full array

    def to_float(x):
        while isinstance(x, list):
            if len(x) == 0:
                return 0.0
            x = x[0]
        try:
            return float(x)
        except Exception:
            return 0.0

    for t in range(T):
        for r in range(R):
            for i in range(min(t + 1, T)): 
                val = to_float(history[t][r])
                y[r, t, i] = val

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