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
                self.x_data, self.gt_data, self.x_img_data, self.gt_img_data, self.coor_masking, self.img_masking, self.time_lag = pickle.load(f)
        else:
            print("Create training data...")
            self.create_training_data()
    
    def create_training_data(self):
        batch_list_x = []
        batch_list_gt = []
        batch_list_x_img = []
        batch_list_gt_img = []
        batch_list_coor_mask = []
        batch_list_img_mask = []
        
        for frame_idx in tqdm(self.available_idx_list):
            seq_list_x, seq_list_gt, seq_list_x_img, seq_list_gt_img, seq_list_coor_mask, seq_list_img_mask = [], [], [], [], [], []

            missing_frame_list = random.sample(range(int((SEQ_SAMPLING - SAMPLE_UNIT) / SAMPLE_UNIT)), int(((SEQ_SAMPLING-SAMPLE_UNIT) / SAMPLE_UNIT) * (self.missing_ratio / 100)))
            missing_frame_idx = 0

            for _ in range(0, SEQ_SAMPLING-SAMPLE_UNIT, SAMPLE_UNIT):  # make sequence data
            
                current_frames = self.data[frame_idx * 22 : frame_idx * 22 + 22]
                current_frames_cp = current_frames.copy()
                gt_frames = self.data[(frame_idx + SAMPLE_UNIT) * 22 : (frame_idx + SAMPLE_UNIT) * 22 + 22]

                if(missing_frame_idx in missing_frame_list):
                    current_frames_cp[:, 2:4] = MISSING_VALUE

                x_list = current_frames_cp[:, 2:4].astype(float).reshape(-10)  # 44
                y_list = gt_frames[:, 2:4].astype(float).reshape(-10)  # 44
                coor_mask_list = np.ones(44) - np.ma.masked_where(x_list == MISSING_VALUE, x_list).mask

                img_mask_list = np.ones((HEIGHT + (RADIUS * 2), WIDTH + (RADIUS * 2)))
                if(coor_mask_list[0] == 0):
                    img_mask_list = np.zeros((HEIGHT + (RADIUS * 2), WIDTH + (RADIUS * 2)))

                seq_list_x.append(torch.tensor(x_list))  # LSTM for coordinate
                seq_list_gt.append(torch.tensor(y_list))  # LSTM for gt coordinate
                seq_list_x_img.append(torch.tensor(make_coor_to_heatmap(current_frames_cp)))  # ConvLSTM for image
                seq_list_gt_img.append(torch.tensor(make_coor_to_heatmap(gt_frames)))  # ConvLSTM for gt image
                seq_list_coor_mask.append(torch.tensor(coor_mask_list))  # mask for coordinate
                seq_list_img_mask.append(torch.tensor(img_mask_list))  # mask for image

                frame_idx = frame_idx + SAMPLE_UNIT
                missing_frame_idx += 1

            batch_list_x.append(torch.stack(seq_list_x).float())  # LSTM for coordinate
            batch_list_gt.append(torch.stack(seq_list_gt).float())  # LSTM for gt coordinate
            batch_list_x_img.append(torch.stack(seq_list_x_img).float())  # LSTM for image
            batch_list_gt_img.append(torch.stack(seq_list_gt_img).float())  # LSTM for gt image
            batch_list_coor_mask.append(torch.stack(seq_list_coor_mask).int())  # mask for coordinate
            batch_list_img_mask.append(torch.stack(seq_list_img_mask).int())  # mask for image
        
        self.x_data = torch.stack(batch_list_x)  # batch_size x seq_len x 44 (LSTM for coordinate)

        self.gt_data = torch.stack(batch_list_gt)  # batch_size x seq_len x 44 (LSTM for gt coordinate)

        self.x_img_data = torch.stack(batch_list_x_img)  #  batch_size x h x w (ConvLSTM for images)

        self.gt_img_data = torch.stack(batch_list_gt_img)  #  batch_size x h x w (ConvLSTM for gt images)

        self.coor_masking = torch.stack(batch_list_coor_mask)  # batch_size x seq_len x 44 (mask for coordinate)

        self.img_masking = torch.stack(batch_list_img_mask)  # batch_size x seq_len x h x w  (mask for image)

        self.time_lag = time_interval(self.coor_masking) # batch_size x seq_len x 44

        with open(self.training_data_file_name, 'wb') as f:
            pickle.dump([self.x_data, self.gt_data, self.x_img_data, self.gt_img_data, self.coor_masking, self.img_masking, self.time_lag], f)

    def __len__(self):
        return len(self.available_idx_list)

    def __getitem__(self, idx):
        x_data = self.x_data[idx]  # seq_len x 44
        gt_data = self.gt_data[idx]  # seq_len x 44
        x_img_data = self.x_img_data[idx]  # seq_len x h x w
        gt_img_data = self.gt_img_data[idx]  # seq_len x 22
        coor_mask = self.coor_masking[idx]  # seq_len x 44
        img_mask = self.img_masking[idx]  # seq_len x h x w
        time_lag = self.time_lag[idx]  # seq_len x 44

        return x_data, gt_data, x_img_data, gt_img_data, coor_mask, img_mask, time_lag
