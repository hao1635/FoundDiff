import os
from accelerate.utils import write_basic_config
write_basic_config() # Write a config file
os._exit(0) # Restart the notebook to reload info from the latest config file 
