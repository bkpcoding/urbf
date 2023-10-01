import exputils as eu
from repetition_config import config
from rl_maze.trainers.random_trainer_image import run

log = run(config=config)

# save the log
log.save()
