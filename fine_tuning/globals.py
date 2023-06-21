import os
import socket
from pathlib import Path

if socket.gethostname() == 'pit-lab':
     ### Multi Object
    # DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/main_pita_clean")
    # DATASET_DIR = Path("/home/pita/Documents/Projects/Aspanformer/merged_3")
    ### Single Object
    # DATASET_DIR = Path("/home/pita/Documents/Projects/HDome/sim2real/agi_aka_aspanformer/assets/datasets/test_1")
    # DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/full_blender_dataset_so")
    ### Single Object Single Axis Rotation
    DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/full_sar")
    REALWORLD_DATA_DIR = Path("/home/pita/HDD/sdb/datasets/val_sar")
elif socket.gethostname() == 'p-pit17':
    DATASET_DIR = Path("/home/pita/Documents/PhD/Aspanformer/clean_dataset")
else:
    # DATASET_DIR = Path("/rds/general/user/pv2017/projects/head-cam-dome/live/datasets/full_blender_dataset_so")
    DATASET_DIR = Path("/rds/general/user/pv2017/projects/head-cam-dome/live/datasets/full_sar")
    REALWORLD_DATA_DIR = Path("/rds/general/user/pv2017/home/Datasets/HDome/val_sar")