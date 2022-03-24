import numpy as np
import matplotlib.pyplot as plt
import torch
from kernel import QuarticKernel
from config import *

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

gpu = 0
device = torch.device(f'cuda:{str(gpu)}' if torch.cuda.is_available() else 'cpu')

def find_frame_drop_idx(episode_list_idx):
    frame_drop_idx_np = np.array(episode_list_idx)
    tmp = np.where((frame_drop_idx_np[1:] - frame_drop_idx_np[:-1]) >= SEQ_SAMPLING)

    return frame_drop_idx_np, tmp[0]

def array_of_available_indices(episode_list, use_idx, overlap=False):
    available_idx_list = []

    for episode_list_idx in range(len(episode_list)):
        if episode_list_idx in use_idx:
            if overlap:
                shift_list = range(0, episode_list[episode_list_idx + 1] - episode_list[episode_list_idx] - SEQ_SAMPLING + 1, OVERLAP_GAP)
            else:
                shift_list = [0]

            for shift in shift_list:
                frame_idx = episode_list[episode_list_idx] + shift
                available_idx_list.append(frame_idx)

    return np.array(available_idx_list)
 
import numpy
 
def fig2data_gray(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
     
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    
    buf = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
    
    buf.shape = (h, w, 3)
     
    R, G, B = buf[:, :, 0], buf[:, :, 1], buf[:, :, 2] 
    buf = R
    
    return buf

def xy_conversion_center_to_lowerleft(center_x, center_y):

    upper_left_x = (WIDTH / 2) + center_x
    upper_left_y = (HEIGHT / 2) - center_y
    
    return upper_left_x, upper_left_y


def make_coor_to_heatmap(current_frames):

    x_list = current_frames[:, 2]
    y_list = current_frames[:, 3]
    
    quartic_kernel = QuarticKernel(RADIUS)
    heatmap = Heatmap(1, RADIUS, WIDTH, HEIGHT, quartic_kernel)

    x_min, x_max = 0, WIDTH
    y_min, y_max = 0, HEIGHT

    x_grid = np.arange(x_min - RADIUS, x_max + RADIUS, 1)
    y_grid = np.arange(y_min - RADIUS, y_max + RADIUS, 1)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

    new_x, new_y = xy_conversion_center_to_lowerleft(x_list, y_list)

    new_x = new_x.astype(float)
    new_y = new_y.astype(float)

    new_x = torch.tensor(new_x)
    new_y = torch.tensor(new_y)

    return heatmap.compute_intensity(new_x, new_y)


class Heatmap:
    def __init__(self, grid_size, radius, width, height, kernel):
        self.grid_size = grid_size
        self.radius = radius
        self.width = width
        self.height = height
        self.kernel = kernel
        self.xc, self.yc = self.get_grid_center()

    def get_grid_center(self):
        # GETTING X,Y MIN AND MAX
        x_min, x_max = 0, self.width
        y_min, y_max = 0, self.height

        # CONSTRUCT GRID
        x_grid = np.arange(x_min - self.radius, x_max +
                           self.radius, self.grid_size)
        y_grid = np.arange(y_min - self.radius, y_max +
                           self.radius, self.grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        # GRID CENTER POINT
        xc = x_mesh + (self.grid_size / 2)
        yc = y_mesh + (self.grid_size / 2)

        return xc, yc

    def compute_intensity(self, x, y, penalty=False):
        Y = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        num_Y = len(Y)

        # PROCESSING
        intensity_list = []
        for j in range(len(self.xc)):
            intensity_row = []
            X = np.hstack((self.xc[j].reshape(-1, 1),
                          self.yc[j].reshape(-1, 1)))
            num_X = len(X)
            if penalty:
                if num_Y < 8:
                    p = np.ones((num_X, 1))
                else:
                    p = self.kernel(X, Y)
            else:
                p = self.kernel(X, Y)
            intensity_row = np.sum(p, axis=1)
            intensity_list.append(intensity_row)

        return np.array(intensity_list)

def make_heatmap_to_coor(current_img):
    
    neighborhood_size = 5
    threshold = 0.2

    data = current_img
    data = data.detach().cpu().numpy()

    batch_list_coor = []
    for b in range(current_img.shape[0]):
        seq_list_coor = []
        for s in range(current_img.shape[1]):
            data_max = filters.maximum_filter(data[b, s, :, :], neighborhood_size)
            maxima = (data[b, s, :, :] == data_max)

            data_min = filters.minimum_filter(data[b, s, :, :], neighborhood_size)
            diff = ((data_max - data_min) > threshold)
            maxima[diff == 0] = 0

            labeled, num_objects = ndimage.label(maxima)
            slices = ndimage.find_objects(labeled)
            x, y = [], []

            for dy,dx in slices:
                x_center = (dx.start + dx.stop - 1) / 2
                x.append(x_center)
                y_center = (dy.start + dy.stop - 1) / 2    
                y.append(y_center)

            if(len(x)) != 22:
                for _ in range(22 - len(x)):
                    x.append(1000)
                    y.append(1000)

            seq_list_coor.append(torch.stack([torch.tensor(x), torch.tensor(y)], dim = -1).flatten())
        
        batch_list_coor.append(torch.stack(seq_list_coor).float())
    
    conversion_coor = xy_conversion_upperleft_to_center(torch.stack(batch_list_coor))

    return conversion_coor

# center coordinates -> upper left coordinates
# together with 'make_player_coor_to_img' function
def xy_conversion_center_to_upperleft(center_x, center_y):

    upper_left_x = center_x + (WIDTH / 2)
    upper_left_y = center_y + (HEIGHT / 2)
    
    return upper_left_y, upper_left_x


# def make_player_coor_to_img(current_frames):

#     img = torch.zeros(HEIGHT, WIDTH)

#     x_list = current_frames[:, 2]
#     y_list = current_frames[:, 3]

#     x, y = xy_conversion_center_to_upperleft(x_list, y_list)

#     # MiSSING VALUE(Inf) exclude
#     new_x = x[np.where(x != MISSING_VALUE)]
#     new_y = y[np.where(y != MISSING_VALUE)]

#     # 소수점은 찍을 수 없기에 int형으로 변환
#     img[new_x.astype(int), new_y.astype(int)] = 1
    
#     # 중복으로 인해 제거되는 좌표 count
#     # miss = 22 - (torch.stack([torch.tensor(new_x.astype(int)), torch.tensor(new_y.astype(int))], dim = 1).unique(dim=0).shape[0])

#     return img  # h x w


# upper left coordinates -> center coordinates
# together with 'make_player_img_to_coor' function
def xy_conversion_upperleft_to_center(upper_left_coor):
    
    center_coor = upper_left_coor.clone().detach()

    center_coor[:, :, 0::2] = upper_left_coor[:, :, 0::2] - ((WIDTH + RADIUS * 2) / 2)
    center_coor[:, :, 1::2] = -(upper_left_coor[:, :, 1::2] - ((HEIGHT + RADIUS * 2) / 2))
    
    return center_coor


#  image -> center coordinate
def make_player_img_to_coor(img):
    
    x_list, y_list = torch.argwhere(img[:, :, :, :].cpu() == 1)

    center_y, center_x = xy_conversion_upperleft_to_center(x_list, y_list)

    final_coor = torch.stack([center_y, center_x], dim = 1).flatten()

    return final_coor  # B x S x 44


# def make_player_coor_to_images(current_frames):
    
#     x_list = current_frames[:, 2]
#     y_list = current_frames[:, 3]

#     fig, ax = plt.subplots(figsize=(3.2, 2.4))  # control figure size(default : (6.4, 4.8))

#     fig.patch.set_facecolor('black')

#     ax.set_facecolor("black")

#     ax.set_xlim(-160, 160) # x축 범위
#     ax.set_ylim(120, -120) # y축 범위

#     ax.axes.xaxis.set_visible(False) # x축 레이블 안보이게 설정
#     ax.axes.yaxis.set_visible(False) # y축 레이블 안보이게 설정

#     ax.spines['left'].set_position('center') # 왼쪽 축을 가운데 위치로 이동
#     ax.spines['left'].set_visible(False)

#     ax.spines['bottom'].set_position('center') # 아래 축을 가운데 위치로 이동
#     ax.spines['bottom'].set_visible(False)

#     ax.spines['top'].set_visible(False) # 윗 축을 안보이게 설정

#     ax.spines['right'].set_visible(False) # 오른쪽 축을 안보이게 설정

#     plt.scatter(x_list, y_list, s=3, marker='o', c="white", linewidths=0, edgecolors='none')
#     gray_array = fig2data_gray(fig) # w x h
#     plt.close()

#     return torch.tensor(gray_array)

def make_player_images_to_coor(input_tensor):
    batch_list = []
    input_tensor = input_tensor.to(device)

    fig, ax = plt.subplots(figsize=(3.2, 2.4))  # control figure size(default : (6.4, 4.8))

    fig.patch.set_facecolor('black')

    ax.set_facecolor("black")

    ax.set_xlim(-120, 120) # x축 범위
    ax.set_ylim(-120, 120) # y축 범위
    
    ax.axes.xaxis.set_visible(False) # x축 레이블 안보이게 설정
    ax.axes.yaxis.set_visible(False) # y축 레이블 안보이게 설정

    ax.spines['left'].set_position('center') # 왼쪽 축을 가운데 위치로 이동
    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_position('center') # 아래 축을 가운데 위치로 이동
    ax.spines['bottom'].set_visible(False)

    ax.spines['top'].set_visible(False) # 윗 축을 안보이게 설정

    ax.spines['right'].set_visible(False) # 오른쪽 축을 안보이게 설정

    for b in range(input_tensor.size(0)):
        seq_list = []
        for s in range(input_tensor.size(1)):
            for i in range(22):
                plt.scatter(input_tensor[b,s,2 * i].cpu().detach().numpy(), input_tensor[b,s,(2 * i)+1].cpu().detach().numpy(), s=1, marker=',', c="white", linewidths=0, edgecolors='none')
            gray_array = fig2data_gray(fig) # w x h
            seq_list.append(torch.tensor(gray_array))
            plt.close()
        batch_list.append(torch.stack(seq_list).float())
        
    pred_img = torch.stack(batch_list)

    return pred_img / 255

def IoU(box1_list, box2_list, threshold=IOU_THRESHOLD):
    # box1_list = gt(B x S x 44)
    # box2_list = pred(B x S x 44)

    box1_list = box1_list.to(device).detach().cpu().numpy()
    box2_list = box2_list.to(device).detach().cpu().numpy()

    box1_len, box2_len = 25, 25
    box1_area, box2_area = (box1_len * 2) ** 2, (box2_len * 2) ** 2

    # linear_sum_assignment안의 cdist에 2D array가 들어가야 하기 때문에 이중 for문 사용
    # 예측된 이미지에서 선수들의 좌표를 골라내면 GT 좌표 순서와 다르기 때문에 linear_sum_assignment 사용
    for b in range(box1_list.shape[0]):
        for s in range(box1_list.shape[1]):
            row_idx, col_idx = linear_sum_assignment(cdist(box1_list[b, s, :].reshape(22, 2), box2_list[b, s, :].reshape(22, 2)))
            box2_list[b, s, :] = box2_list[b, s, :].reshape(22, 2)[col_idx].flatten()
    
    box1_list = torch.tensor(box1_list).to(device)
    box2_list = torch.tensor(box2_list).to(device)
    
    iou_batch_list = []
    for b in range(box1_list.shape[0]):
        iou_seq_list = []
        for s in range(box1_list.shape[1]):
            iou_player_list = []
            for i in range(22):
                x1 = torch.max(box1_list[b, s, 2 * i] - box1_len, box2_list[b, s, 2 * i] - box2_len)
                y1 = torch.max(box1_list[b, s, (2 * i) + 1] - box1_len, box2_list[b, s, (2 * i) + 1] - box2_len)
                x2 = torch.min(box1_list[b, s, 2 * i] + box1_len, box2_list[b, s, 2 * i] + box2_len)
                y2 = torch.min(box1_list[b, s, (2 * i) + 1] + box1_len, box2_list[b, s, (2 * i) + 1] + box2_len)

                # compute the width and height of the intersection
                w = torch.max(torch.zeros(x1.shape).to(device), x2 - x1)
                h = torch.max(torch.zeros(y1.shape).to(device), y2 - y1)

                inter = w * h
                iou = inter / (box1_area + box2_area - inter)

                iou_player_list.append(iou)
        
            iou_seq_list.append(torch.stack(iou_player_list).float())
        
        iou_batch_list.append(torch.stack(iou_seq_list).float())

    iou_batch_list = torch.stack(iou_batch_list)
    # result = iou_list.permute(1, 2, 0)  # B x S x 22
    
    result_count = (iou_batch_list >= threshold).sum() - (iou_batch_list  == 1).sum()
    # result_count = (result >= threshold).sum()

    return result_count

def time_interval(x_mask):
    # x_mask = b x s x 44
    seq_len = x_mask.shape[1]
    
    x_mask = x_mask[:,:, 0::2]  # b x s x 22
    
    interval_matrix = torch.zeros(x_mask.shape)  # b x s x 22

    # if previous mask value is 1, modify interval_matrix values to (s_t - s_t-1)
    # if previous mask value is 0, modify interval_matrix values to (s_t  - s_t-1 + d_t-1)
    for timestep in range(1, seq_len):
        
        prev_one = x_mask[:, timestep-1].nonzero()  
        prev_zero = (x_mask[:, timestep-1] ==0).nonzero()

        interval_matrix[prev_one[:,0], timestep, prev_one[:,1]] = 1
        interval_matrix[prev_zero[:,0], timestep, prev_zero[:,1]] = 1 + interval_matrix[prev_zero[:,0], timestep-1, prev_zero[:,1]]
        
    return torch.repeat_interleave(interval_matrix, 2, dim=2)  # b x s x 22
