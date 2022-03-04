import numpy as np
import matplotlib.pyplot as plt
import torch

from config import *

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

def fig2data_gray(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
     
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
#     print("width : {}, height : {}".format(w, h))
    
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    
    buf.shape = (h, w, 3)
#     print("RGB : {}".format(buf.shape))
     
    R, G, B = buf[:, :, 0], buf[:, :, 1], buf[:, :, 2] 
    buf = 0.299 * R + 0.587 * G + 0.114 * B
    
#     print("GRAY : {}".format(buf.shape))
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
#     buf = numpy.roll(buf, 3, axis = 2)
    return buf

def make_player_images(current_frames):

    x_list = current_frames[:, 2]
    y_list = current_frames[:, 3]

    fig, ax = plt.subplots()
                
    ax.set_xlim(-120, 120) # x축 범위
    ax.set_ylim(-120, 120) # y축 범위

    ax.axes.xaxis.set_visible(False) # x축 레이블 안보이게 설정
    ax.axes.yaxis.set_visible(False) # y축 레이블 안보이게 설정

    ax.spines['left'].set_position('center') # 왼쪽 축을 가운데 위치로 이동
    ax.spines['bottom'].set_position('center') # 아래 축을 가운데 위치로 이동
    ax.spines['top'].set_visible(False) # 윗 축을 안보이게 설정
    ax.spines['right'].set_visible(False) # 오른쪽 축을 안보이게 설정

    plt.scatter(x_list, y_list)
    gray_array = fig2data_gray(fig) # w(432) x h(288)
    plt.close()

    return torch.tensor(gray_array)

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