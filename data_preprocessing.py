from mimetypes import MimeTypes
from matplotlib.pyplot import axes
import torch
import numpy as np
import pandas as pd
import pickle
import os.path
import random
from torchvision.transforms.functional import to_pil_image

from util import *
from config import *
from tqdm import tqdm

class FootballDataset(torch.utils.data.Dataset):
    def __init__(self, start_ratio, end_ratio, missing_ratio, train=True,  list_save=True, missing=True, overlap=True):

        self.overlap = overlap
        self.missing = missing
        self.missing_ratio = missing_ratio
        self.player_num = 22

        self.max_missing_num_per_frame = self.player_num // self.missing_ratio + 1

        df = pd.read_csv('./raw_data/new_main_add_episode_use.csv')
        
        self.start_idx = int((len(df) // self.player_num) * start_ratio) * self.player_num
        self.end_idx = int((len(df) // self.player_num) * end_ratio) * self.player_num

        self.data = df.iloc[self.start_idx : self.end_idx]  # split data
        self.data = self.data.reset_index().drop(['index'], axis=1)

        frame_drop_list = np.array(self.data[self.data.episode_use == 1].index) // 22
        
        episode_list, use_idx = find_frame_drop_idx(frame_drop_list)

        self.available_idx_list = array_of_available_indices(episode_list, use_idx, overlap=overlap)

        self.data = self.data.to_numpy()

        self.training_data_file_name = f'data/train_{train}_training_data.pkl'

        if os.path.isfile(self.training_data_file_name) and not list_save:
            print("Load training data...")
            print(self.training_data_file_name)
            with open(self.training_data_file_name, 'rb') as f:
                self.x_data, self.x_img_data, self.y_data, self.masking, self.time_lag = pickle.load(f)
        else:
            print("Create training data...")
            self.create_training_data()
    
    def create_training_data(self):
        batch_list_x = []
        batch_list_x_img = []
        batch_list_gt = []
        batch_list_x_mask = []

        for frame_idx in tqdm(self.available_idx_list):
            seq_list_x, seq_list_x_img, seq_list_gt, seq_list_mask = [], [], [], []

            for _ in range(0, SEQ_SAMPLING-SAMPLE_UNIT, SAMPLE_UNIT):  # make sequence data

                current_frames = self.data[frame_idx * 22:frame_idx * 22 + 22]
                gt_frames = self.data[(frame_idx + SAMPLE_UNIT) * 22: (frame_idx + SAMPLE_UNIT * 22) + 22]

                missing_num = random.randint(0, self.max_missing_num_per_frame) if self.missing else 0
                missing_player_list = random.sample(range(22), missing_num)

                current_frames[missing_player_list, 2:4] = MISSING_VALUE  # missing x,y coordinate

                x_list = current_frames[:,2:4].astype(float).reshape(-10)  # 44
                x_mask_list = np.ones(44) - np.ma.masked_where(x_list == MISSING_VALUE, x_list)

                seq_list_x.append(torch.tensor(x_list))
                seq_list_gt.append(make_player_images(gt_frames))
                seq_list_mask.append(torch.tensor(x_mask_list))
                seq_list_x_img.append(make_player_images(current_frames))
                
                frame_idx = frame_idx + SAMPLE_UNIT

            batch_list_x.append(torch.stack(seq_list_x).float())
            batch_list_x_img.append(torch.stack(seq_list_x_img).float())
            batch_list_gt.append(torch.stack(seq_list_gt).float())
            batch_list_x_mask.append(torch.stack(seq_list_mask).int())
        
        self.x_data = torch.stack(batch_list_x)  # batch_size x seq_len x 44

        self.x_img_data = torch.stack(batch_list_x_img)  #  batch_size x h x w
        self.x_img_data  = self.x_img_data / 255 # scaling

        self.y_data = torch.stack(batch_list_gt)  #  batch_size x h x w
        self.y_data = self.y_data / 255 # scaling

        self.masking = torch.stack(batch_list_x_mask)  # batch_size x seq_len x 44

        self.time_lag = time_interval(self.masking) # batch_size x seq_len x 44

        with open(self.training_data_file_name, 'wb') as f:
            pickle.dump([self.x_data, self.x_img_data, self.y_data, self.masking, self.time_lag], f)

    def __len__(self):
        return len(self.available_idx_list)

    def __getitem__(self, idx):
        x_data = self.x_data[idx]  # seq_len x 44
        x_img_data = self.x_img_data[idx]  #  seq_len x w x h
        y_data = self.y_data[idx]  #  seq_len x 22
        x_mask = self.masking[idx]  #  seq_len x 44
        time_lag = self.time_lag[idx] # seq_len

        return x_data, x_img_data, y_data, x_mask, time_lag
