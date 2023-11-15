#!/bin/bash

# registers the current repetition as a job to the OAR scheduler

##########################
# Parameters

NUM_PROCESSES=$EU_PRJ_GRICAD_DEFAULT_NUM_PROCESSES
WALLTIME=$EU_PRJ_GRICAD_DEFAULT_WALLTIME
PROJECT=$EU_PRJ_GRICAD_DEFAULT_PROJECT
RUN_REPETITION_SCRIPT=$EU_PRJ_DEFAULT_RUN_REPETITION_SCRIPT

##########################
# Execution

while getopts ":n:w:r" arg; do
  case $arg in
    n) NUM_PROCESSES=$OPTARG;;
    w) WALLTIME=$OPTARG;;
  esac
done

# identify if the repetition needs to run to only create a job if necessary
NUM_REPETITIONS_TO_EXECUTE=`eu_gricad_python -c "
import exputils

n_scripts = exputils.manage.get_number_of_scripts_to_execute(
  directory='.',
  start_scripts='$RUN_REPETITION_SCRIPT')

print(n_scripts)"`

if [ $NUM_REPETITIONS_TO_EXECUTE -gt 0 ]; then

  # Create a script that runs the job. It has two features:
  # 1) identify if the repetition still has to run
  # 2) activate the conda environment
  echo "NUM_REPETITIONS_TO_EXECUTE=\`eu_gricad_python -c \"
import exputils

n_scripts = exputils.manage.get_number_of_scripts_to_execute(
  directory='.',
  start_scripts='$RUN_REPETITION_SCRIPT')

print(n_scripts)\"\`

if [ \$NUM_REPETITIONS_TO_EXECUTE -gt 0 ]; then

  CONDA_ENV=\$EU_PRJ_GRICAD_DEFAULT_CONDA_ENV

  if ! command -v conda &> /dev/null
  then
    source /applis/environments/conda.sh
  fi

  if [ \"\$CONDA_DEFAULT_ENV\" != \"\$CONDA_ENV\" ]; then
    conda activate \$CONDA_ENV
  fi

  ./\$EU_PRJ_DEFAULT_RUN_REPETITION_SCRIPT
fi" > oar_job_run_repetition.sh
  chmod +x oar_job_run_repetition.sh

  echo "Register current repetition as a job ..."
  oarsub -l /nodes=1/core=$NUM_PROCESSES,walltime=$WALLTIME --project $PROJECT --stdout gricad_run_repetition.out.log --stderr gricad_run_repetition.err.log ./oar_job_run_repetition.sh
  echo "Finished."
fi