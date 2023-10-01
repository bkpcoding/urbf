# Univariate Radial Basis Function Layers: Brain-inspired Deep Neural layers for Low-Dimensional Inputs

Source code for the paper "Univariate Radial Basis Function Layers: Brain-inspired Deep Neural Layers for Low-Dimensional Inputs".

#### Abstract

Deep Neural Networks (DNNs) became the standard tool for function approximation with most of the introduced architectures being developed for high-dimensional input data. However, many real-world problems have low-dimensional inputs. By default, standard Multi-Layer Perceptrons (MLPs) are used in these cases and an investigation into specialized architectures is missing. We propose a new DNN layer called Univariate Radial Basis Function (U-RBF) layer as an alternative for low-dimensional continuous input data. Similar to sensory neurons in the brain, the U-RBF layer processes each individual input dimension with a population of neurons whose activations depend on different preferred input values. We verify its effectiveness compared to MLPs in a function regression and a reinforcement learning task. We show that the U-RBF is especially advantageous for lower dimensional inputs and when the target function becomes more complex and difficult to approximate. We hope our research will spark interest in applying the U-RBF layer to appropriate problems and to further explore architectures for low-dimensional input data.

#### Installation

Requirements:

- Python 3.6 or higher
- Linux (developed under Ubuntu 22.04)



Create the conda environment,

```bash
conda create --new urbf python==3.8
conda activate urbf
```

Install the required packages using installation script,

```bash
chmod +x install.sh
./install.sh
```



#### Overview

The source code has 3 components,

- src: This package contains the source code for running the experiments,
  - regression: This folder contains source code for forming the dataset as well as the training code for the supervised learning task.
  - rl_maze: This folder contains the source code for the environment and experimental procedure for the reinforcement learning tasks.
- experiments: Contains the parameters settings and start scripts to run the experiments. All the experiments data are stored in this folder, further notebooks in analyze folder can be used to visualize the data. 
- utils: Contains the code for the U-RBF, M-RBF network as well as the agents (the folder contains a fork of Stable-Baselines3 library, which is modified internally to build the agents using our networks).

Experiments:

- 01_Gaussians_regression :  Experiment for Gaussians function of section 4.1 of the paper.
- 02_Discountinuous_regression: Experiment for Discountinuous function of section 4.1 of the paper.
- 03_maze_xy_input: Experiment for Maze task using Reinforcement Learning using coordinate input from section 4.2 of the paper.
- 04_maze_matrix_input: Experiment for Maze task using Reinforcement Learning using matrix input from section 4.2 of the paper.
- 05_maze_image_input: Experiment for Maze task using Reinforcement Learning using image input from section 4.2 of the paper.



#### Usage

Each experiment has several parameters. Each parameter defines which algorithm to use and its hyperparameters. Several repetitions (for the paper n=10) for each parameter are executed which have a different random seed.
 The exputils package is used to generate for each parameter and repetition the code from a ODS file and code templates. 

Each experiment folder contains an *experiment_configurations.ods*. This file contains the different parameters, one per row. Each parameter has an ID (called *Experiment-ID*) which must be unique. Please note, rows in the *experiment_configurations.ods* that have an empty *Experiment-ID* column are ignored

The exputils package uses code templates under the *src* directory to create for each parameter and repetition (number of repetitions are defined in the *repetitions* column of the ODS) the required code. These are stored in an *experiments* directory which will be automatically generated.

To generate and run the experiments, run the *run_experiments.sh* script:

```bash
./run_experiments.sh
```

The command takes as input argument how many experiments (or better repetitions) should run in parallel. This saves time if you have several CPU's or cores. Example: `./run_experiments.sh 6`

Please note, you can add new parameters or a higher number of repetitions after your first run in the *experiment_configurations.ods* file. If you then run the *run_experiments.sh* script, it will only run the repetitions for the new parameters or where more repetitions are required.

#### Analyzing the Results

After the experiments have been finished, the Jupyter notebooks in the *analyze* directory can be used to look at the results. Each repetition has saved its result in a *data* directory under the *experiments* directory. The notebooks will load this data and visualize it.

To use the notebooks, first run Jupyter notebook using the following command:

```bash
jupyter notebook
```

The *plot_figures.ipynb* replicates the plots that can be found in the paper.

The *overview.ipynb* notebook is used to visualize the data. The table-widget that is displayed after running the its first cell, allows to load data from specific parameters. It also allows to set names for each parameter which are then used in the figures. Please note, loading data from too many parameters might result in an out-of-memory situation! The other cells can be used to visualize different aspects from the experiments.