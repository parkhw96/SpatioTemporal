from audioop import bias
from tkinter import HIDDEN
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import *
from config import *
from torch.utils.data import DataLoader
from model_ConvLSTM import *
# from model_BRITS import *
# from model_LSTM import *

from torchsummary import summary as summary_
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

train_set = FootballDataset(start_ratio = 0.0, end_ratio = 0.9, missing_ratio = 10, train = True, list_save = False, missing = True, overlap = False)
valid_set = FootballDataset(start_ratio = 0.9, end_ratio = 1.0, missing_ratio = 10, train = False, list_save = False, missing = True, overlap = False)

data_loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
data_loader_valid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

gpu = 0
device = torch.device(f'cuda:{str(gpu)}' if torch.cuda.is_available() else 'cpu')

# model_convlstm = ConvLSTM(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, kernel_size = KERNEL_SIZE, num_layers = NUM_LAYERS).to(device)
model_lstm = LSTM_Model(feature_dim=FEATURE_DIM, rnn_dim=RNN_DIM, hidden_dim=HIDDEN_DIM, input_dim=INPUT_DIM, kernel_size=KERNEL_SIZE, bias=True).to(device)
# model_brits = BRITS(feature_dim=FEATURE_DIM, rnn_dim=RNN_DIM)
# model_lstm = nn.LSTM(input_size = FEATURE_DIM, hidden_size = HIDDEN_DIM, num_layers = NUM_LAYERS, batch_first = True).to(device)
# model_lstm = RNN(input_size=FEATURE_DIM, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, feature_dim = FEATURE_DIM).to(device)

print(count_parameters(model_lstm))

criterion = nn.MSELoss()

# optimizer_convlstm = optim.AdamW(model_convlstm.parameters(), lr=LR, weight_decay=W_DECAY)
optimizer_lstm = optim.AdamW(model_lstm.parameters(), lr=LR, weight_decay=W_DECAY)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", min_lr=1e-8, verbose=True, patience=10)

# print(len(train_set), len(valid_set))
# print(len(data_loader_train), len(data_loader_valid))

# tensorboard settings
eventid = './runs/radius:{}_layer:{}_h_d:{}_distance-threshold:{}_time:{}_local-threshold:{}_weight_deacy:{}'.format(RADIUS, NUM_LAYERS, HIDDEN_DIM, DISTANCE_THRESHOLD, datetime.now(), LOCAL_MAXIMA_THRESHOLD, W_DECAY)
writer = SummaryWriter(eventid)


def compute_loss(model, optimizer, loader, epoch, mode):
    accum_loss = 0
    accum_total_gt = 0
    accum_total_recall_count = 0
    accum_total_pred = 0
    accum_total_precision_count = 0
    
    for batch_idx, samples in tqdm(enumerate(loader), total=len(loader)):
        loss = 0
        total_gt = 0
        total_recall_count = 0
        total_pred = 0
        total_precision_count = 0

        if mode == "train":
            model.train()
        else:
            model.eval()
        
        # x_coor_batch = B x S x 44 (coordinate for LSTM)
        # y_coor_batch = B x S x 44 (GT coordinate for LSTM)
        # x_img_batch = B x S x h x w (image for ConvLSTM)
        # y_img_batch = B x S x h x w (GT image for ConvLSTM)
        # coor_mask_batch = B x S x 44 (mask for LSTM)
        # img_mask_batch = B x S x h x w (mask for ConvLSTM)
        # coor_mask_batch_y = B x S x 44 (mask for LSTM(GT))
        # img_mask_batch_y = B x S x h x w (mask for ConvLSTM(GT))
        x_coor_batch, y_coor_batch, x_img_batch, y_img_batch, coor_mask_batch, img_mask_batch, coor_mask_batch_y, img_mask_batch_y = samples
        
        ### ConvLSTM ###
        # x_img_batch = x_img_batch.unsqueeze(2)  # B x S x h x w -> B x S x 1 x h x w
        # pred_img_batch, _ = model(x_img_batch)  # B x S x h x w 
        
        if torch.cuda.is_available():
            x_coor_batch, coor_mask_batch, x_img_batch = x_coor_batch.to(device), coor_mask_batch.to(device), x_img_batch.to(device)

        # print(summary(model_lstm.to(device), input_size = [(8, 24, 44), (8, 24, 44), (8, 24, 78, 114)]))

        ### LSTM ###
        pred_coor_batch = model_lstm(x_coor_batch, coor_mask_batch, x_img_batch, embed_mode = True)
        pred_coor_batch = pred_coor_batch.permute(1, 0, 2)

        if epoch == 2999 and mode == 'valid':
            torch.save(x_img_batch, './x_img_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(y_img_batch, './y_img_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(pred_coor_batch, './pred_coor_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(y_coor_batch, './y_coor_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(x_coor_batch, './x_coor_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(coor_mask_batch, './coor_mask_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(img_mask_batch, './img_mask_batch_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(coor_mask_batch_y, './coor_mask_batch_y_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            torch.save(img_mask_batch_y, './img_mask_batch_y_radius{}_epoch{}_d_t{}_lstm{}.pt'.format(RADIUS, epoch, DISTANCE_THRESHOLD, mode))
            

        # img_mask_complement = (1 - img_mask_batch).to(device)
        x_mask_complement_y = (1 - coor_mask_batch_y).to(device)

        # loss = criterion(y_img_batch.to(device) * img_mask_complement, pred_img_batch * img_mask_complement)  #ConvLSTM
        loss = criterion(y_coor_batch.to(device) * x_mask_complement_y, pred_coor_batch.to(device) * x_mask_complement_y)  #LSTM
        
        # total_gt -> 전체 정답의 개수
        # total_recall_count -> prediction과 매칭된 개수
        # total_pred -> 전체 예측된 개수
        # total_precision_count -> gt와 매치된 개수
        # total_gt, total_recall_count, total_pred, total_precision_count = make_heatmap_to_coor_ConvLSTM(y_coor_batch.to(device) * x_mask_complement, pred_img_batch * img_mask_complement)  #ConvLSTM
        total_gt, total_recall_count, total_pred, total_precision_count = make_heatmap_to_coor_LSTM(y_coor_batch.to(device) * x_mask_complement_y, pred_coor_batch.to(device) * x_mask_complement_y)  #LSTM
        
        if epoch == 3000:
            print(total_gt)
            print(total_recall_count)
            print(total_pred)
            print(total_precision_count)
            total_gt, total_recall_count, total_pred, total_precision_count = make_heatmap_to_coor_LSTM(y_coor_batch.to(device) * x_mask_complement_y, pred_coor_batch.to(device) * x_mask_complement_y)  #LSTM

        # print("total_gt : {}".format(total_gt))
        # print("total_recall_count : {}".format(total_recall_count))
        # print("total_pred : {}".format(total_pred))
        # print("total_precision_count : {}".format(total_precision_count))

        accum_total_gt += total_gt
        accum_total_recall_count += total_recall_count
        accum_total_pred += total_pred
        accum_total_precision_count += total_precision_count
        accum_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        

    return accum_loss, accum_total_gt, accum_total_recall_count, accum_total_pred, accum_total_precision_count



for epoch in range(N_EPOCHS):

    ### print LSTM
    train_loss, total_gt_train, total_recall_count_train, total_pred_train, total_precision_count_train = compute_loss(model_lstm, optimizer_lstm, data_loader_train, epoch, mode = "train")
    valid_loss, total_gt_valid, total_recall_count_valid, total_pred_valid, total_precision_count_valid = compute_loss(model_lstm, optimizer_lstm, data_loader_valid, epoch, mode = "valid")

    if total_pred_train == 0:
        final_precision_train = 0
    else:
        final_precision_train = total_precision_count_train / total_pred_train

    if total_pred_valid == 0:
        final_precision_valid = 0
    else:
        final_precision_valid = total_precision_count_valid / total_pred_valid

    print("epoch : {}, train_loss : {}, valid_loss : {}".format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid)))
    print('recall_train : {}, recall_valid : {}'.format(total_recall_count_train / total_gt_train, total_recall_count_valid / total_gt_valid))
    print('precision_train : {}, precision_valid : {}'.format(final_precision_train, final_precision_valid))

    writer.add_scalar(f"Loss/train", train_loss / len(data_loader_train), epoch)
    writer.add_scalar(f"Loss/valid", valid_loss / len(data_loader_valid), epoch)

    writer.add_scalar(f"recall/train", total_recall_count_train / total_gt_train, epoch)
    writer.add_scalar(f"precision/train", final_precision_train, epoch)

    writer.add_scalar(f"recall/valid", total_recall_count_valid / total_gt_valid, epoch)
    writer.add_scalar(f"precision/valid", final_precision_valid, epoch)

    writer.add_scalar(f"total_pred/train", total_pred_train, epoch)
    writer.add_scalar(f"total_pred/valid", total_pred_valid, epoch)


    ### print BRITS
    # train_loss = compute_loss(model_lstm, optimizer_lstm, data_loader_train, epoch, mode = "train")
    # valid_loss = compute_loss(model_brits, optimizer_brits, data_loader_valid, epoch, mode = "valid")

    # print("epoch : {}, train_loss : {}, valid_loss : {}".format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid)))
    # print('iou_count_train : {}, iou_count_valid : {}'.format(total_iou_sum_train, total_iou_sum_valid))

    # writer.add_scalar(f"Loss/train", train_loss / len(data_loader_train), epoch)
    # writer.add_scalar(f"Loss/valid", valid_loss / len(data_loader_valid), epoch)
    
