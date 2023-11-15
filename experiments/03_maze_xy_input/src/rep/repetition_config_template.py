import exputils as eu
import torch


##############################
# general config

config = eu.AttrDict(
        size = <size>,
        
        seed = 42 + <repetition_id>,
        
        difficulty = <difficulty>,
        
        timesteps = <timesteps>,
        
        net_arch = <net_arch>,
        
        lr = <lr>,
        
        gamma = <gamma>,
        
        batch_size = <batch_size>,
        
        buffer_size = <buffer_size>,
        
        exploration_initial_eps = <exploration_initial_eps>,
        
        exploration_fraction = <exploration_fraction>,
        
        exploration_final_eps = <exploration_final_eps>,
        
        rbf_mlp = <rbf_mlp>,
        
        n_neurons_per_input = <n_neurons_per_input>,
        
        ranges = <ranges>,
        
        latent_dim = <latent_dim>,

        sutton_maze = <sutton_maze>,
        
        mrbf_on = <mrbf_on>,
        
        mrbf_units = <mrbf_units>,

 )
