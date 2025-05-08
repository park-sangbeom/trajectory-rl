import torch 
import torch.nn as nn 
import numpy as np 
import math 
from datetime import datetime
import matplotlib.pyplot as plt 
from scipy import ndimage
from pathlib import Path
import matplotlib.patches as patches
from itertools import combinations
import numpy as np


def get_diversity(actions, env_name='place'):
    trajectories = []
    anchor_num = 10
    if env_name == 'place':
        for action in actions:
            xs = []; ys = []; zs = []
            x = 0.6; y = 0.; z = 1.0
            action = action.reshape(3,10)/30
            for i in range(anchor_num):
                x += action[0,i]; y += action[1,i]; z+=action[2,i]
                x = np.clip(x, 0.6, 0.83); y = np.clip(y, -0.33, 0.33); z = np.clip(z, 0.9, 1.1)
                xs.append(x); ys.append(y); zs.append(z)
            trajectories.append(np.array([xs, ys, zs]).T.reshape(anchor_num,3))
    
    elif env_name == 'sweep':
        for action in actions:
            xs = []; ys = []; zs = []
            x = 0.65; y = 0.
            action = action.reshape(2,10)/20
            for i in range(anchor_num):
                x += action[0,i]; y += action[1,i]; z+=action[2,i]
                x = np.clip(x, 0.65, 1.15); y = np.clip(y, -0.33, 0.33)
                xs.append(x); ys.append(y)
            trajectories.append(np.array([xs, ys]).T.reshape(anchor_num,2))
    diversity = average_pairwise_distance(trajectories)
    return diversity 

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def pairwise_trajectory_distance(trajectory1, trajectory2):
    assert trajectory1.shape == trajectory2.shape, "Trajectories must have the same shape"
    distances = [euclidean_distance(p1, p2) for p1, p2 in zip(trajectory1, trajectory2)]
    return np.mean(distances)

def average_pairwise_distance(trajectories):
    pairwise_distances = [pairwise_trajectory_distance(t1, t2) for t1, t2 in combinations(trajectories, 2)]
    return np.mean(pairwise_distances)

def get_linear_layer(hdim, hidden_actv, initializer='normal'):
    layers = []
    for hdim_idx in range(0,len(hdim)-1):
        layer = nn.Linear(hdim[hdim_idx],hdim[hdim_idx+1])
        if initializer=='normal':
            nn.init.normal_(layer.weight,mean=0.0, std=0.01)
            nn.init.zeros_(layer.bias)
        elif initializer=='kaiming':
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif initializer=='xavier':
            torch.nn.init.xavier_normal_(layer.weight)
        layers.append(layer)
        layers.append(hidden_actv())
    return layers

def torch2np(x_torch):
    if x_torch is None:
        x_np = None
    else:
        x_np = x_torch.detach().cpu().numpy()
    return x_np

def np2torch(x_np,device='cpu'):
    if x_np is None:
        x_torch = None
    else:
        x_torch = torch.tensor(x_np,dtype=torch.float32,device=device)
    return x_torch

def get_runname():
    now = datetime.now()
    format = "%m%d:%H%M"
    runname = now.strftime(format)
    return runname

def scale_up(anchor, scale_params=np.array([3.14,2.22])):
    return np.clip(anchor, a_min=-1, a_max=1)*scale_params

def scale_down(anchor, scale_parmas=np.array([3.14,2.22])):
    return anchor/scale_parmas

def scaling(anchor):
    scale_params = np.ones_like(anchor.shape[0])*0.21
    return anchor/scale_params

def scale_down_list(anchors, scale_parmas=np.array([3.14,2.22])):
    down_scaled_list = []
    for anchor in anchors: 
        down_scaled_list.append(anchor/scale_parmas)
    return np.array(down_scaled_list).flatten()

""" Make a range of reward as 0~1 """
def reward_nzr(x=np.random.randn(64,1)):
    if len(x.shape)==1:
        x_mean = np.mean(x, axis=None)
        x_std = np.std(x, axis=None)
    else:
        x_mean  = np.mean(x,axis=0)
        x_std   = np.std(x,axis=0)        
    return (x-x_mean)/x_std  

def reward_nzr_torch(x=torch.randn(64,1)):
    if len(x.shape)==1:
        x_mean = torch.mean(x)
        x_std = torch.std(x)
    else: 
        x_mean = torch.mean(x, axis=0)
        x_std = torch.std(x, axis=0)
    return (x-x_mean)/x_std 

def diversity_coverage(position_lst=[[0.1,0.1]], length=0.05):
    # grid_map = set({(0,0),(0,1),(0,2),(0,3),
    #                 (1,0),(1,1),(1,2),(1,3),
    #                 (2,0),(2,1),(2,2),(2,3),
    #                 (3,0),(3,1),(3,2),(3,3)})
    small_grid_map = set()
    for position in position_lst:
        if position[0]<0: 
            position[0]+=0.2 
        elif position[1]<0:
            position[1]+=0.2 
        x_dim = position[0]//length 
        y_dim = position[1]//length  
        small_grid_map.add((x_dim,y_dim))
    cover = len(small_grid_map)
    return cover 

def heatmap_viz(position_lst=[[0.1,0.1]], length=0.00625):
    dim_lst = []
    for position in position_lst:
        if position[1] >= 0.2: position[1]-=0.01 
        elif position[1] <= -0.2: position[1]+=0.01 
        if position[0] >=0.2: position[0]-=0.01 
        elif position[0] <= -0.2: position[0] +=0.01
        height_dim = abs(((position[1]+0.2)//length)-63)
        width_dim  = (position[0]+0.2)//length
        dim_lst.append([math.floor(height_dim),math.floor(width_dim)])
    background = np.zeros((int(0.4//length), int(0.4//length)))
    heatmaps = []
    for dim in dim_lst: 
        background[dim[0],dim[1]]+=0.2
    heat_map = ndimage.gaussian_filter(background, sigma=1)
    heatmaps.append(heat_map)
    heatmaps = np.array(heatmaps)
    fig, ax = plt.subplots()
    fig.set_size_inches(64, 64, forward=True)
    result = plt.imshow(heatmaps.sum(axis=0),  cmap=plt.get_cmap("viridis"))
    plt.savefig("heatmap_viz.png")

def coverage_viz(position_lst=[[0.1,0.1]], length=0.05):
    dim_lst = []
    for position in position_lst:
        if position[1] >= 0.2: position[1]-=0.01 
        elif position[1] <= -0.2: position[1]+=0.01 
        if position[0] >=0.2: position[0]-=0.01 
        elif position[0] <= -0.2: position[0] +=0.01
        height_dim = abs(((position[1]+0.2)//length)-7)
        width_dim  = (position[0]+0.2)//length
        dim_lst.append([math.floor(height_dim),math.floor(width_dim)])
    background = np.zeros((int(0.4//length), int(0.4//length)))
    heatmaps = []
    for dim in dim_lst: 
        background[dim[0],dim[1]]+=0.1
    heat_map = ndimage.gaussian_filter(background, sigma=0.01)
    heatmaps.append(heat_map)
    heatmaps = np.array(heatmaps)
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 8, forward=True)
    result = plt.imshow(heatmaps.sum(axis=0),  cmap=plt.get_cmap("viridis"))
    plt.savefig("coverage_viz.png")

def viz_diversity(position_lst, title='Coverage', accuracy=None,plot_name="test.png"):
    identified_center = get_center_pos(position_lst)
    identified_center = list(set(identified_center))
    diversity = len(identified_center)/256#296
    plt.figure(figsize=(10,10))

    plt.title(title, fontsize=20)
    plt.xlim(-0.22, 0.22)
    plt.ylim(-0.22, 0.22)

    a,b = make_circle(radius=0.09)
    plt.plot(a,b, color='k', alpha=0.5)
    a,b = make_circle(radius=0.21)
    plt.plot(a,b, color='k')

    plt.axhline(y=0, xmin=0.02, xmax=0.98-0.685, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0, xmin=0.02+0.685, xmax=0.98, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0, ymin=0.02+0.685, ymax=0.98, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0, ymin=0.02, ymax=0.98-0.685, color='k', alpha=0.4,linestyle='dashed')

    # Vertical Lines Right
    plt.axvline(x=0.21-0.021*9, ymin=0.0225+0.685-0.01, ymax=0.975, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*9, ymin=0.0225, ymax=0.975-0.685+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*8, ymin=0.02+0.685-0.025, ymax=0.98-0.0165, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*8, ymin=0.02+0.0165, ymax=0.98-0.685+0.025, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*7, ymin=0.02+0.0382-0.01+0.685-0.09, ymax=0.98-0.0382+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*7, ymin=0.02+0.0382-0.01, ymax=0.98-0.0382+0.01-0.685+0.09, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*6, ymin=0.02+0.0191*3-0.013+0.685-0.173, ymax=0.98-0.0191*3+0.013, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*6, ymin=0.02+0.0191*3-0.013, ymax=0.98-0.0191*3+0.013-0.685+0.173, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*5, ymin=0.02+0.0191*4-0.01, ymax=0.98-0.0191*4+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*4, ymin=0.02+0.0191*5, ymax=0.98-0.0191*5, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*3, ymin=0.02+0.0191*7+0.01, ymax=0.98-0.0191*7-0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*2, ymin=0.02+0.0191*8+0.0375, ymax=0.98-0.0191*8-0.0375, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.21-0.021*1, ymin=0.02+0.0191*13+0.02, ymax=0.98-0.0191*13-0.02, color='k', alpha=0.4,linestyle='dashed')

    # Vertical Lines Left
    plt.axvline(x=0.-0.021*1, ymin=0.0225+0.685-0.01, ymax=0.975, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*1, ymin=0.0225, ymax=0.975-0.685+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*2, ymin=0.02+0.685-0.025, ymax=0.98-0.0165, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*2, ymin=0.02+0.0165, ymax=0.98-0.685+0.025, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*3, ymin=0.02+0.0382-0.01+0.685-0.09, ymax=0.98-0.0382+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*3, ymin=0.02+0.0382-0.01, ymax=0.98-0.0382+0.01-0.685+0.09, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*4, ymin=0.02+0.0191*3-0.013+0.685-0.173, ymax=0.98-0.0191*3+0.013, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*4, ymin=0.02+0.0191*3-0.013, ymax=0.98-0.0191*3+0.013-0.685+0.173, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*5, ymin=0.02+0.0191*4-0.01, ymax=0.98-0.0191*4+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*6, ymin=0.02+0.0191*5, ymax=0.98-0.0191*5, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*7, ymin=0.02+0.0191*7+0.01, ymax=0.98-0.0191*7-0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*8, ymin=0.02+0.0191*8+0.0375, ymax=0.98-0.0191*8-0.0375, color='k', alpha=0.4,linestyle='dashed')
    plt.axvline(x=0.-0.021*9, ymin=0.02+0.0191*13+0.02, ymax=0.98-0.0191*13-0.02, color='k', alpha=0.4,linestyle='dashed')

    # Horizontal Lines Upper
    plt.axhline(y=0.21-0.021*9, xmin=0.0225+0.685-0.01, xmax=0.975, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*9, xmin=0.0225, xmax=0.975-0.685+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*8, xmin=0.02+0.685-0.025, xmax=0.98-0.0165, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*8, xmin=0.02+0.0165, xmax=0.98-0.685+0.025, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*7, xmin=0.02+0.0382-0.01+0.685-0.09, xmax=0.98-0.0382+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*7, xmin=0.02+0.0382-0.01, xmax=0.98-0.0382+0.01-0.685+0.09, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*6, xmin=0.02+0.0191*3-0.013+0.685-0.165, xmax=0.98-0.0191*3+0.013, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*6, xmin=0.02+0.0191*3-0.013, xmax=0.98-0.0191*3+0.013-0.685+0.165, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*5, xmin=0.02+0.0191*4-0.01, xmax=0.98-0.0191*4+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*4, xmin=0.02+0.0191*5, xmax=0.98-0.0191*5, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*3, xmin=0.02+0.0191*7+0.01, xmax=0.98-0.0191*7-0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*2, xmin=0.02+0.0191*8+0.0375, xmax=0.98-0.0191*8-0.0375, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.21-0.021*1, xmin=0.02+0.0191*13+0.025, xmax=0.98-0.0191*13-0.025, color='k', alpha=0.4,linestyle='dashed')

    # Horizontal Lines Bottome
    plt.axhline(y=0.-0.021*1, xmin=0.0225+0.685-0.01, xmax=0.975, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*1, xmin=0.0225, xmax=0.975-0.685+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*2, xmin=0.02+0.685-0.025, xmax=0.98-0.0165, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*2, xmin=0.02+0.0165, xmax=0.98-0.685+0.025, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*3, xmin=0.02+0.0382-0.01+0.685-0.09, xmax=0.98-0.0382+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*3, xmin=0.02+0.0382-0.01, xmax=0.98-0.0382+0.01-0.685+0.09, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*4, xmin=0.02+0.0191*3-0.013+0.685-0.165, xmax=0.98-0.0191*3+0.013, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*4, xmin=0.02+0.0191*3-0.013, xmax=0.98-0.0191*3+0.013-0.685+0.165, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*5, xmin=0.02+0.0191*4-0.01, xmax=0.98-0.0191*4+0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*6, xmin=0.02+0.0191*5, xmax=0.98-0.0191*5, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*7, xmin=0.02+0.0191*7+0.01, xmax=0.98-0.0191*7-0.01, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*8, xmin=0.02+0.0191*8+0.0375, xmax=0.98-0.0191*8-0.0375, color='k', alpha=0.4,linestyle='dashed')
    plt.axhline(y=0.-0.021*9, xmin=0.02+0.0191*13+0.025, xmax=0.98-0.0191*13-0.025, color='k', alpha=0.4,linestyle='dashed')

    plt.xlabel("X-axis", fontsize=14)
    plt.ylabel("Y-axis", fontsize=14)

    font1 = {'family': 'serif',
        'color':  'k',
        'weight': 'normal',
        'size': 16}
    quad1=True;quad2=True;quad3=True;quad4=True
    for position in identified_center:
        if position[0]>=0 and position[1]>=0:
            if quad1:
                shp=patches.Rectangle(position, 0.021,0.021, color='dodgerblue', alpha=1.0)
                quad1=False 
            else:
                shp=patches.Rectangle(position, 0.021,0.021, color='dodgerblue', alpha=1.0)
        if position[0]<0 and position[1]>=0:
            if quad2:
                shp=patches.Rectangle(position, 0.021,0.021, color='#e35f62', alpha=1.0)
                quad2=False 
            else:
                shp=patches.Rectangle(position, 0.021,0.021, color='#e35f62', alpha=1.0)
        if position[0]>=0 and position[1]<0:
            if quad3:
               shp=patches.Rectangle(position, 0.021,0.021, color='limegreen', alpha=1.0)
               quad3=False 
            else:
                shp=patches.Rectangle(position, 0.021,0.021, color='limegreen', alpha=1.0)
        if position[0]<0 and position[1]<0:
            if quad4:
               shp=patches.Rectangle(position, 0.021,0.021, color='violet', alpha=1.0)
               quad4=False 
            else:
                shp=patches.Rectangle(position, 0.021,0.021, color='violet', alpha=1.0)
        plt.gca().add_patch(shp)
    # plt.legend(['Quadrant.1', 'Quadrant.2', 'Quadrant.3','Quadrant.4'], loc='upper right',)
    # plt.scatter(position_lst[:,0], position_lst[:,1])


    plt.text(0.10,-0.212,"Coverage:{:.2f}%".format(diversity*100), fontdict = font1)
    plt.text(0.101,-0.196,"Accuracy:{:.2f}%".format(accuracy*100), fontdict = font1)
    plt.savefig(plot_name)

def get_center_pos(position_lst,offset=0.021, total_len=10):
    center_list = []
    for i in range(total_len):
        center_list.append(0.+i*(offset))
    identified_center = []
    for position in position_lst:
        x_idx = abs(position[0])//offset
        y_idx = abs(position[1])//offset 

        if position[0]>=0 and position[1]>=0:
            x_val = center_list[int(x_idx)]
            y_val = center_list[int(y_idx)]
        elif position[0]<0 and position[1]>=0:
            x_val = -(center_list[int(x_idx)]+0.021)
            y_val = center_list[int(y_idx)]
        elif position[0]>=0 and position[1]<0:
            x_val = center_list[int(x_idx)]
            y_val = -(center_list[int(y_idx)]+0.021)
        elif position[0]<0 and position[1]<0:
            x_val = -(center_list[int(x_idx)]+0.021)
            y_val = -(center_list[int(y_idx)]+0.021)
        identified_center.append((x_val,y_val))
    return identified_center

### Visualize ###
def make_circle(radius, center=np.zeros(2)):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius * np.cos(theta) + center[0]
    b = radius * np.sin(theta) + center[1]
    return [a,b]

