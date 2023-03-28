import os
import socket
from pathlib import Path

if socket.gethostname() == 'pit-lab':
    # DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/main_pita_clean")
    DATASET_DIR = Path("/home/pita/Documents/Projects/Aspanformer/merged_3")
elif socket.gethostname() == 'p-pit17':
    DATASET_DIR = Path("/home/pita/Documents/PhD/Aspanformer/clean_dataset")
else:
    DATASET_DIR = Path("/home/pita/HDD/sdb/datasets/main_pita")