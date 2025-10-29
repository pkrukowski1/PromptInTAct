import numpy as np

# incremental forgetting metric
# ASSUMES EQUAL TASK SIZE (e.g., classes per task)
# 
# y is num_trials x num_tasks_time x num_tasks_eval
#
# so, y[0,1,0] would be the eval performance of the first task (third index) after
# training the second task (second index) in random seed trial 1 (first index). y[0,1,3] would be the eval performance of the
# fourth task after training the second task in random seed trial 1, and thus should not exist (will be ignored)
#
# note that for 1 random trial, input should be 1 x T x T size, where T is number of tasks

def calc_forgetting(y):
    y = np.asarray(y)
    if len(y.shape) != 3:
        raise ValueError("Input y must be a 3D array: trials x tasks_time x tasks_eval")
    num_trials, num_tasks_time, num_tasks_eval = y.shape
    if num_tasks_time != num_tasks_eval:
        raise ValueError("tasks_time and tasks_eval must be equal (square matrix per trial)")

    T = num_tasks_time
    fgt_all = []

    for r in range(num_trials):
        normalized_sum = 0.0
        for k in range(T - 1):
            initial = y[r, k, k]
            final = y[r, T - 1, k]
            f = initial - final
            subsequent = T - k - 1
            norm_f = f / subsequent if subsequent > 0 else 0.0
            normalized_sum += norm_f
        avg_normalized = normalized_sum / (T - 1) if T > 1 else 0.0
        fgt_all.append(avg_normalized)

    fgt_all = np.asarray(fgt_all)
    return np.mean(fgt_all), np.std(fgt_all)