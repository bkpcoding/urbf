import numpy as np
import exputils as eu
import matplotlib.pyplot as plt


def plot_loss(experiment_id, repetition_ids, phase, action=None, n_positions_per_dim=100, orientations=None,
                              cmap=plt.cm.coolwarm, vmin=None, vmax=None, plot_colorbar=True):


    ##############################################################
    # collect data


    # load environment from first repetition
    rep_config_module = eu.data.loading.load_experiment_python_module(
        'repetition_config.py',
        experiment_id=experiment_id,
        repetition_id=0
    )
    #take the data from the first repetition
    error = eu.data.loading.load_experiment_data(
        experiments_directory = "../experiments",
        allowed_experiments_id_list=[experiment_id],
    )
    print(error)

# load the mean total reward of each experiment
def load_total_reward_per_experiment(min_number_of_repetition):

    # load experiment descriptions one after the other
    experiment_descriptions = eu.data.load_experiment_descriptions(experiments_directory='../experiments')

    total_reward_per_experiment = dict()

    for exp_descr in experiment_descriptions.values():
        single_experiment_descriptions = eu.AttrDict()
        single_experiment_descriptions[exp_descr.id] = exp_descr

        single_experiment_data, _ = eu.data.load_experiment_data(single_experiment_descriptions,
                                                                 pre_allowed_data_filter=['Error'])

        total_reward, _ = eu.data.select_experiment_data(
            single_experiment_data,
            datasources='Error')
        total_reward = total_reward[0][0]  # first plot, first experiment

        # do not include experiments where not all repetitions are computed
        if total_reward is not None and len(total_reward) >= min_number_of_repetition and not np.any(np.isnan(total_reward)):
            total_reward_per_experiment[exp_descr.id] = np.mean(total_reward)

    return total_reward_per_experiment

if __name__ == '__main__':

    experiment_id = 200001
    repetition_ids = [0, 1]
    phase = 0

    #fig = plot_loss(experiment_id, repetition_ids, phase, n_positions_per_dim=100, vmin=0.0, vmax=10.0)
    error = load_total_reward_per_experiment(2)
    print(error)
    plt.show()