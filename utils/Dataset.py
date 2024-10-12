import json
import numpy as np
import pandas as pd


with open("data_attribute_name.json", "r") as file_handle:
    data_attribute = json.load(file_handle)

feature_name_list = data_attribute['feature_name']
target_name = data_attribute['target_name']


def load_logging_data(file_path_list: list):
    logging_data = []

    for file_path in file_path_list:
        logging_data.append(pd.read_csv(file_path))

    logging_data = pd.concat(logging_data)

    return logging_data


def create_dataset(logging_data: pd.DataFrame):
    feature = logging_data[feature_name_list]
    target = logging_data[target_name]

    return feature, target