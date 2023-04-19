import os
import socket
from pathlib import Path

if socket.gethostname() == 'pit-lab':
    # DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/main_pita_clean")
    # DATASET_DIR = Path("/home/pita/Documents/Projects/Aspanformer/merged_3")
    DATASET_DIR = Path("/home/pita/Documents/Projects/HDome/sim2real/agi_aka_aspanformer/assets/datasets/test_1")
elif socket.gethostname() == 'p-pit17':
    DATASET_DIR = Path("/home/pita/Documents/PhD/Aspanformer/clean_dataset")
else:
    DATASET_DIR = Path("/rds/general/user/pv2017/projects/head-cam-dome/live/datasets/full_blender_dataset")