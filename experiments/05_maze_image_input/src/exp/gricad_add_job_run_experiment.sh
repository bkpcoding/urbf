#!/bin/bash

# registers the current experiment as a job to the OAR scheduler

##########################
# Parameters

NUM_PROCESSES=$EU_PRJ_GRICAD_DEFAULT_NUM_PROCESSES
WALLTIME=$EU_PRJ_GRICAD_DEFAULT_WALLTIME
PROJECT=$EU_PRJ_GRICAD_DEFAULT_PROJECT
RUN_REPETITION_SCRIPT=$EU_PRJ_DEFAULT_RUN_REPETITION_SCRIPT
ARGUMENTS=$EU_PRJ_GRICAD_ADD_JOB_RUN_EXPERIMENT_ARGUMENTS

##########################
# Execution

while getopts ":n:w:r" arg; do
  case $arg in
    n) NUM_PROCESSES=$OPTARG;;
    w) WALLTIME=$OPTARG;;
    r) ARGUMENTS="$ARGUMENTS -r";;
  esac
done

# identify how many repetitions need to run to only create a job if necessary and for the needed resources
# this depends if all processes need to rerun (-r) or if not
if [[ "$ARGUMENTS" == *"-r"* ]]; then
  NUM_REPETITIONS_TO_EXECUTE=`eu_gricad_python -c "
import exputils

n_scripts = exputils.manage.get_number_of_scripts(
  directory='.',
  start_scripts='$RUN_REPETITION_SCRIPT')

print(n_scripts)"`
else
  NUM_REPETITIONS_TO_EXECUTE=`eu_gricad_python -c "
import exputils

n_scripts = exputils.manage.get_number_of_scripts_to_execute(
  directory='.',
  start_scripts='$RUN_REPETITION_SCRIPT')

print(n_scripts)"`
fi

if [ $NUM_REPETITIONS_TO_EXECUTE -gt 0 ]; then

  # only register requiered for resources
  if [ $NUM_REPETITIONS_TO_EXECUTE -lt $NUM_PROCESSES ]; then
    NUM_PROCESSES=$NUM_REPETITIONS_TO_EXECUTE
  fi

  # the oarsub command does not allow to give arguments to the script
  # therefore create a script here with the arguments which is then given to the oarsub
  echo "eu_gricad_run_experiment -n $NUM_PROCESSES $ARGUMENTS" > oar_job_run_experiment.sh
  chmod +x oar_job_run_experiment.sh

  echo "Register current experiment as a job ..."
  oarsub -l /nodes=1/core=$NUM_PROCESSES,walltime=$WALLTIME --project $PROJECT --stdout gricad_run_experiment.out.log --stderr gricad_run_experiment.err.log ./oar_job_run_experiment.sh
  echo "Finished."
fi