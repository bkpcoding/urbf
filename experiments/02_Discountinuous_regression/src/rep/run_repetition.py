import exputils as eu
from repetition_config import config
import sys
sys.path.append("/scratch/bootes/bbasavas/code/")
sys.path.append("/home/sagar/inria/code/non_convex_testing/")
from non_convex_testing.core.train1 import run_task
log = run_task(config=config)

# save the log
#log.save()
