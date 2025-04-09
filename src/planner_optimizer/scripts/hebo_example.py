import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from hebo.design_space.design_space import DesignSpace

import warnings
warnings.filterwarnings("ignore")


def branin_objective(x: pd.DataFrame) -> np.ndarray:
    """
    Compute the Branin function for each input (x0, x1) provided as a DataFrame.
    """
    X = x[['x0', 'x1']].values
    num_x = X.shape[0]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * np.pi)
    results = np.zeros((num_x, 1))
    for i in range(num_x):
        x0, x1 = X[i]
        results[i, 0] = a * (x1 - b*x0**2 + c*x0 - r)**2 + s*(1-t)*np.cos(x0) + s
    return results

# Define the HEBO design space
space = DesignSpace().parse([
    {'name': 'x0', 'type': 'num', 'lb': -5, 'ub': 10},
    {'name': 'x1', 'type': 'num', 'lb': 0,  'ub': 15}
])

# BO loop
bo = BO(space, model_name='gp')
start_time = time.time()
for i in range(64):
    rec_x = bo.suggest()
    precomputed = branin_objective(rec_x)
    bo.observe(rec_x, precomputed)
    if i % 4 == 0:
        print('Iter %d, best_y = %.2f' % (i, bo.y.min()))
print('BO time: %.2f' % (time.time() - start_time))

# HEBO seq loop
start_time = time.time()
hebo_seq = HEBO(space, model_name = 'gp', rand_sample = 4)   
for i in range(64):
    rec_x = hebo_seq.suggest(n_suggestions=1)
    precomputed = branin_objective(rec_x)
    hebo_seq.observe(rec_x, precomputed)
    if i % 4 == 0:
        print('Iter %d, best_y = %.2f' % (i, hebo_seq.y.min()))
print("Seq HEBO time: %.2f" % (time.time() - start_time))


# HEBO batch loop
hebo_batch = HEBO(space, model_name='gp', rand_sample=4)
start_time = time.time()
for i in range(16):
    rec_x = hebo_batch.suggest(n_suggestions=8) 
    # Compute the Branin objective sequentially for each suggestion
    precomputed_list = []
    for j in range(len(rec_x)):
        single_x = rec_x.iloc[[j]]  
        precomputed = branin_objective(single_x)  
        precomputed_list.append(precomputed)
    precomputed_array = np.vstack(precomputed_list)
    hebo_batch.observe(rec_x, precomputed_array)
    print('Iter %d, best_y = %.2f' % (i, hebo_batch.y.min()))

print('HEBO batch time: %.2f' % (time.time() - start_time))


# Compute the cumulative minimum value over iterations
conv_hebo_batch = np.minimum.accumulate(hebo_batch.y)
conv_bo_seq     = np.minimum.accumulate(bo.y)
conv_hebo_seq   = np.minimum.accumulate(hebo_seq.y)

plt.figure(figsize=(8, 6))
plt.semilogy(conv_hebo_batch[::8] - np.min(hebo_batch.y), 'x-', label='HEBO, Parallel, Batch = 8')
plt.semilogy(conv_hebo_seq - np.min(hebo_batch.y), 'x-',label = 'HEBO, Sequential')
plt.semilogy(conv_bo_seq - np.min(hebo_batch.y), 'x-',label = 'BO, LCB')
plt.xlabel('Iterations')
plt.ylabel('Regret')
plt.legend()
plt.show()
