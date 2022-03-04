# from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import *
from config import *
from torch.utils.data import DataLoader
from models import *
from model_BRITS import *

import torch
import torch.optim as optim

train_set = FootballDataset(start_ratio=0.0, end_ratio=0.9, missing_ratio=10, train=True, list_save=False, missing=True, overlap=False)
valid_set = FootballDataset(start_ratio=0.9, end_ratio=1.0, missing_ratio=10, train=False, list_save=False, missing=True, overlap=False)

data_loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
data_loader_valid = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)

gpu = 0
device = torch.device(f'cuda:{str(gpu)}' if torch.cuda.is_available() else 'cpu')

model = ConvLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS).to(device)
model_brits = BRITS(feature_dim=FEATURE_DIM, rnn_dim=RNN_DIM)

criterion = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=W_DECAY)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", min_lr=1e-8, verbose=True, patience=10)

# print(len(train_set), len(valid_set))
# print(len(data_loader_train), len(data_loader_valid))

# tensorboard settings
# writer = SummaryWriter()
# eventid = './runs/tmp'
# writer = SummaryWriter(eventid)

def compute_loss(model, optimizer, loader, epoch, mode):
    accum_loss = 0
    for batch_idx, samples in tqdm(enumerate(loader), total=len(loader)):
        loss = 0
        # x_data_batch = B x S x 44
        # x_img_data_batch = B x S x h x w
        # y_data_batch = B x S x h x w
        # x_mask_batch = B x S x 44

        x_batch, x_img_batch, y_batch, x_mask_batch, time_lag_batch = samples
        
        x_img_batch = x_img_batch.unsqueeze(2)

        # layer_output_list = 1 x B x S x hidden_dim x h x w
        # layer_output_list, _ = model(x_img_batch)
        
        a, b, c, d, e, f = model_brits(x_batch, x_mask_batch, time_lag_batch)

        average_output = (a + d) / 2

        print("Asd")
        # loss = criterion(y_batch.to(device), layer_output_list)

        accum_loss += loss.item()

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return accum_loss 

for epoch in range(N_EPOCHS):
    train_loss = compute_loss(model, optimizer, data_loader_train, epoch, mode = "train")
    valid_loss = compute_loss(model, optimizer, data_loader_valid, epoch, mode = "valid")

    print("epoch : {}, train_loss : {}, valid_loss : {}".format(epoch, train_loss / len(data_loader_train), valid_loss / len(data_loader_valid)))