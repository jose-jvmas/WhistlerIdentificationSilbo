import os
from functools import partial

import torch
import pandas as pd
import librosa
import numpy as np
from datasets import load_dataset, Dataset as HF_Dataset
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor


################ RESULTS LOGGING:


def write_results(dst_results_file: str, res_dict: dict):
    """Write results on external file"""
    try:
        if os.path.isfile(dst_results_file):
            out_file = pd.read_csv(dst_results_file)
            out_file.loc[len(out_file)] = res_dict
        else:
            out_file = pd.DataFrame([res_dict])
        out_file.to_csv(dst_results_file, index=False)
        print(f"Results written on {dst_results_file}")

    except Exception as e:
        print(f"Error writing results on {dst_results_file}")
        print(e)


################ DATASET LOADING:


def update_features(in_data):
    cleaned = in_data['features'].replace('[', '').replace(']', '').replace(',', '')
    in_data['features'] = np.fromstring(cleaned, sep=' ').tolist()

    return in_data


def load_CSV_dataset(src_folder: str, encoder: str, param: str) -> HF_Dataset:
    """ Loading features from dataset directly from CSV file """

    # Path to file:
    path_file = os.path.join(src_folder, "{}_{}.csv".format(encoder, param))

    # Load file as HF dataset:
    encoded_silbo = load_dataset("csv", data_files = path_file)['train']
    encoded_silbo = encoded_silbo.map(update_features)

    return encoded_silbo