#!/bin/bash

# registers the current repetition as a job to the OAR scheduler

##########################
# Parameters
WALLTIME=$EU_PRJ_INRIA_CLUSTER_DEFAULT_WALLTIME
CLUSTER=$EU_CLUSTER
RUN_REPETITION_SCRIPT=$EU_PRJ_DEFAULT_RUN_REPETITION_SCRIPT
ACCESSMACHINE=$EU_PRJ_INRIA_CLUSTER_ACCESS_MACHINE
MACHINE=$EU_MACHINE
##########################
# Execution

while getopts ":m:w:c" arg; do
  case $arg in
    m) MACHINE=$OPTARG;;
    w) WALLTIME=$OPTARG;;
    c) CLUSTER=$OPTARG;;
  esac
done

# if machine is given, then create string 'AND <machine>' to add to the command
ADDMACHINE=
if [ -n "$MACHINE" ]; then
    ADDMACHINE="AND host='$MACHINE'"
fi

# path to experiments on the local host
LOCAL_EXPERIMENTS_DIR=$EU_PRJ_LOCAL_EXPERIMENTS_PATH
# path to the experiments directory on the target host
EXTERNAL_EXPERIMENTS_DIR=$EU_PRJ_INRIA_EXPERIMENTS_PATH

# replace the path to the experiments on local host with the one at the target host
LOCAL_SCRIPT_PATH=`pwd`
EXTERNAL_PATH=`echo $LOCAL_SCRIPT_PATH | sed "s+${LOCAL_EXPERIMENTS_DIR}+${EXTERNAL_EXPERIMENTS_DIR}+g"`

echo "Register current repetition as a job ..."
ssh $ACCESSMACHINE "source eu_activate $EU_ACTIVE_PRJ;
  cd ${EXTERNAL_PATH};
  echo \"eu_inria_singularity ./\$EU_PRJ_DEFAULT_RUN_REPETITION_SCRIPT\" > oar_job_run_repetition.sh;
  chmod +x oar_job_run_repetition.sh;
  oarsub -l /host=1/gpudevice=1,walltime=$WALLTIME -p \"cluster='$CLUSTER' $ADDMACHINE\" -t besteffort --stdout inria_cluster_run_repetition.out.log --stderr inria_cluster_run_repetition.err.log ./oar_job_run_repetition.sh"
echo "Finished."
