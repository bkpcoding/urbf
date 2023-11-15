from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np

# load the data from two experimental files and put them 
# in a dictinary according to their experiment number

def load_dict():
    # scores is a dictionary with experiment number as key and
    # a matrix of scores with size (num_runs, 1) as value
    scores = {}
    algorithm = ['mlp', 'urbf']
    data_arr = []
    for j in range(10):
        file = '../experiments/experiment_111013/repetition_00000'+ str(j) + '/data/rollout_ep_rew_mean.npy'
        data = np.load(file, allow_pickle=True)
        data = (data + 200) / (200 - 11)
        data_arr.append(data)
    scores['mlp'] = np.array(data_arr)
    scores = scores['mlp'].reshape(10, 1)
    data_arr = []
    for j in range(10):
        file = '../experiments/experiment_111013/repetition_00000'+ str(j) + '/data/rollout_ep_rew_mean.npy'
        data = np.load(file, allow_pickle=True)
        data = (data + 200) / (200 - 11)
        data_arr.append(data)
    scores['urbf'] = np.array(data_arr)
    return scores

