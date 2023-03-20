import os
import multiprocessing as mp
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

LANDMARK_FILES_DIR = "data/train_landmark_files"
TRAIN_FILE = "data/train.csv"
label_map = json.load(open("data/sign_to_prediction_index_map.json", "r"))

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        face_x = x[:,:468,:].contiguous().view(-1, 468*3)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*3)
        pose_x = x[:,489:522,:].contiguous().view(-1, 33*3)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*3)
        
        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        x1m = torch.mean(face_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)
        
        x1s = torch.std(face_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)
        
        xfeat = torch.cat([x1m,x2m,x3m,x4m, x1s,x2s,x3s,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        
        return xfeat
    
feature_converter = FeatureGen()

ROWS_PER_FRAME = 543
def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def convert_row(row):
    x = load_relevant_data_subset(os.path.join("data", row[1].path))
    x = feature_converter(torch.tensor(x)).cpu().numpy()
    return x, row[1].label

def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)
    npdata = np.zeros((df.shape[0], 3258))
    nplabels = np.zeros(df.shape[0])
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata[i,:] = x
            nplabels[i] = y
    
    np.save("data/feature_data.npy", npdata)
    np.save("data/feature_labels.npy", nplabels)
        
convert_and_save_data()