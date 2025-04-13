
import json
import os


data = {

    "adj_matrix": [],
    "metrics": [],
    "reward": [],

}
DATA_FILE = "process_data.json"


def save_data():
    """Save data dictionary to file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)


def load_data():
    """Load data dictionary from file"""
    global data
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
    return data
