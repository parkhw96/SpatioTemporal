from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import *
from config import *
from torch.utils.data import DataLoader
from model_ConvLSTM import *
# from model_BRITS import *
# from model_LSTM import *

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

model_convlstm = ConvLSTM(input_dim = INPUT_DIM, hidden_dim = HIDDEN_DIM, kernel_size = KERNEL_SIZE, num_layers = NUM_LAYERS).to(device)
# model_brits = BRITS(feature_dim=FEATURE_DIM, rnn_dim=RNN_DIM)
# model_lstm = nn.LSTM(input_size = FEATURE_DIM, hidden_size = HIDDEN_DIM, num_layers = NUM_LAYERS, batch_first = True).to(device)
# model_lstm = RNN(input_size=FEATURE_DIM, hidden_size=HIDDEN_DIM, num_layers=NUM_LAYERS, feature_dim = FEATURE_DIM).to(device)

criterion = nn.MSELoss()

optimizer_convlstm = optim.AdamW(model_convlstm.parameters(), lr=LR, weight_decay=W_DECAY)
# optimizer_lstm = optim.AdamW(model_lstm.parameters(), lr=LR, weight_decay=W_DECAY)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", min_lr=1e-8, verbose=True, patience=10)

# print(len(train_set), len(valid_set))
# print(len(data_loader_train), len(data_loader_valid))

# tensorboard settings
eventid = './runs/radius{}_layer{}_iou-threshold{}_time:{}'.format(RADIUS, NUM_LAYERS, IOU_THRESHOLD, datetime.now())
writer = SummaryWriter(eventid)


def compute_loss(model, optimizer, loader, epoch, mode):
    accum_loss = 0

    global acc_train
    global acc_valid

    for batch_idx, samples in tqdm(enumerate(loader), total=len(loader)):
        loss = 0
        # x_coor_batch = B x S x 44 (coordinate for LSTM)
        # y_coor_batch = B x S x 44 (GT coordinate for LSTM)
        # x_img_batch = B x S x h x w (image for ConvLSTM)
        # y_img_batch = B x S x h x w (GT image for ConvLSTM)
        # coor_mask_batch = B x S x 44 (mask for BRITS)
        # img_mask_batch = B x S x h x w (mask for ConvLSTM)
        # time_lag_batch = B x S x 44 (time_lag for BRITS)
        x_coor_batch, y_coor_batch, x_img_batch, y_img_batch, coor_mask_batch, img_mask_batch, time_lag_batch = samples
        
        ### ConvLSTM ###
        x_img_batch = x_img_batch.unsqueeze(2)  # B x S x h x w -> B x S x 1 x h x w
        pred_img_batch, _ = model(x_img_batch)  # B x S x h x w 
        
        # if(epoch == 200):
        #     torch.save(x_img_batch, './x_img_batch_radius{}_epoch{}_layer{}.pt'.format(RADIUS, epoch, NUM_LAYERS))
        #     torch.save(y_img_batch, './y_img_batch_radius{}_epoch{}_layer{}.pt'.format(RADIUS, epoch, NUM_LAYERS))
        #     torch.save(pred_img_batch, './pred_img_batch_radius{}_epoch{}_layer{}.pt'.format(RADIUS, epoch, NUM_LAYERS))
        #     torch.save(y_coor_batch, './y_coor_batch_radius{}_epoch{}_layer{}.pt'.format(RADIUS, epoch, NUM_LAYERS))
        #     asd

        pred_coor_batch = make_heatmap_to_coor(pred_img_batch)  # B x S x 44

        img_mask_complement = (1 - img_mask_batch).to(device)
        loss = criterion(y_img_batch.to(device) * img_mask_complement, pred_img_batch * img_mask_complement)
        
        ### LSTM ###
        # pred_coor_batch = model(x_coor_batch)
        # print("Asd")


        ### BRITS ###
        # forward_output, _, _, backward_output, _, _ = model(x_batch, x_mask_batch, time_lag_batch)
        # average_output = (forward_output + backward_output) / 2
        # pred_img = make_player_images2(average_output.to(device))

        # x_mask_complement = (1 - x_mask_batch).to(device)

        # loss = criterion(y_gt_batch.to(device) * x_mask_complement, average_output.to(device) * x_mask_complement)
        
        accum_loss += loss.item()

        x_mask_complement = (1 - coor_mask_batch).to(device)

        if(epoch % 5 == 0):
            if mode == 'train': 
                total_imputation_count = (coor_mask_batch == 0).sum() / 2
                iou_count_Brits = (IoU(y_coor_batch.to(device) * x_mask_complement, pred_coor_batch.to(device) * x_mask_complement))
                
                # print("ioud_count_train : {}".format(iou_count_Brits))
                acc_train = (iou_count_Brits / total_imputation_count) * 100
                
            elif mode == 'valid':
                total_imputation_count = (coor_mask_batch == 0).sum() / 2
                iou_count_Brits = (IoU(y_coor_batch.to(device) * x_mask_complement, pred_coor_batch.to(device) * x_mask_complement))

                # print("ioud_count_valid : {}".format(iou_count_Brits))
                acc_valid = (iou_count_Brits / total_imputation_count) * 100

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return accum_loss



for epoch in range(N_EPOCHS):

    acc_train = 0
    acc_valid = 0

    ### print ConvLSTM
    train_loss = compute_loss(model_convlstm, optimizer_convlstm, data_loader_train, epoch, mode = "train")
    valid_loss = compute_loss(model_convlstm, optimizer_convlstm, data_loader_valid, epoch, mode = "valid")

    print("epoch : {}, train_loss : {}, valid_loss : {}".format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid)))
    print('iou_train : {}%, iou_valid : {}%'.format(acc_train, acc_valid))

    writer.add_scalar(f"Loss/train", train_loss / len(data_loader_train), epoch)
    writer.add_scalar(f"Loss/valid", valid_loss / len(data_loader_valid), epoch)
    
    if(epoch % 5 == 0):
        writer.add_scalar(f"ACC/train", acc_train, epoch)
        writer.add_scalar(f"ACC/valid", acc_valid, epoch)



    ### print BRITS
    # train_loss = compute_loss(model_lstm, optimizer_lstm, data_loader_train, epoch, mode = "train")
    # valid_loss = compute_loss(model_brits, optimizer_brits, data_loader_valid, epoch, mode = "valid")

    # print("epoch : {}, train_loss : {}, valid_loss : {}".format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid)))
    # print('iou_count_train : {}, iou_count_valid : {}'.format(total_iou_sum_train, total_iou_sum_valid))

    # writer.add_scalar(f"Loss/train", train_loss / len(data_loader_train), epoch)
    # writer.add_scalar(f"Loss/valid", valid_loss / len(data_loader_valid), epoch)
    
