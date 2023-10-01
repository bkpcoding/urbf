# set number of parallel processes
NUM_PARALLEL=1
if [ $# -eq 1 ]
then
  NUM_PARALLEL=$1
fi

# use exputils package to generate experiments and run them
python -c "import exputils

print('Generate experiments ...')
exputils.manage.generate_experiment_files('experiment_configurations.ods', directory='./experiments/')

print('Run experiments ...')
exputils.manage.start_experiments(
  start_scripts='run_repetition.py',
  parallel=$NUM_PARALLEL,
  is_chdir=True,
  verbose=False)

print('Finished')"
