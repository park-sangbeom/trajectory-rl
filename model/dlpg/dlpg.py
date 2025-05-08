import sys
sys.path.append("..")
import numpy as np 
import random as rd
import math
import torch 
import torch.nn as nn
from scipy.spatial import distance
from model.dlpg.cvae import ConditionalVariationalAutoEncoderClass
from model.utils.gaussian_path_planner import GaussianRandomPathClass, kernel_levse, kernel_se,soft_squash
from model.utils.utils import np2torch, torch2np

class DeepLatentPolicyGradient():
    def __init__(self, xdim     = 2,
                       cdim     = 4,
                       zdim     = 2,
                       hdims    = [64,64],
                       actv_enc = nn.ReLU(),
                       actv_dec = None, #nn.LeakyReLU(), 
                       actv_out = nn.Tanh(), 
                       actv_q   = nn.Softplus(),
                       device   = 'cpu',
                       spectral_norm = False):
        super(DeepLatentPolicyGradient,self).__init__()
        self.cvae = ConditionalVariationalAutoEncoderClass(x_dim=xdim,
                                                           c_dim=cdim,
                                                           z_dim=zdim,
                                                           h_dims=hdims,
                                                           actv_enc=actv_enc,
                                                           actv_dec=actv_dec,
                                                           actv_out=actv_out,
                                                           actv_q =actv_q,
                                                           spectral_norm=spectral_norm,
                                                           device=device).to(device) 
        self.grp         = GaussianRandomPathClass(name='GRP',kernel=kernel_levse)
        self.xdim      = xdim 
        self.zdim      = zdim 
        self.cdim      = cdim

    """ Prior: Multi trajectories """
    def random_explore(self, n_sample=50):
        t_test    = np.linspace(start=0.65,stop=1.0,num=10).reshape((-1,1))
        t_anchor = np.array([0.65,1.0]);x_anchor = np.array([0,0])
        self.grp.set_data(t_anchor= t_anchor.reshape((-1,1)),
                    x_anchor = np.array([x_anchor]).T,
                    l_anchor = np.array([[1,0.1]]).T,
                    t_test   = t_test,
                    l_test   = np.ones((len(t_test),1)),
                    hyp_mean = {'g':0.1,'l':0.08,'w':1e-6},
                    hyp_var  = {'g':0.05,'l':0.1,'w':1e-6})
        ys, xs = self.grp.sample(n_sample=n_sample)
        return xs, ys 

    def random_explore_3d(self, n_sample=50, scenario=1):
        if scenario==1:
            rand_x = np.random.uniform(low=0.6, high=0.83)
            rand_y = np.random.uniform(low=-0.35, high=0.)
        elif scenario==2:
            rand_x = np.random.uniform(low=0.6, high=0.7)
            rand_y = np.random.uniform(low=-0.35, high=0.35)
        elif scenario==3:
            if rd.random()<0.5:
                rand_x = np.random.uniform(low=0.73, high=0.83)
                rand_y = np.random.uniform(low=-0.35, high=-0.1)
            else: 
                rand_x = np.random.uniform(low=0.6, high=0.7)
                rand_y = np.random.uniform(low=0.1, high=0.35)
        elif scenario==4:
            rand_x = np.random.uniform(low=0.6, high=0.83)
            rand_y = np.random.uniform(low=0, high=0.35)
        elif scenario==5:
            rand_x = np.random.uniform(low=0.73, high=0.83)
            rand_y = np.random.uniform(low=-0.35, high=0.35)
        elif scenario==6:
            if rd.random()<0.5:
                rand_x = np.random.uniform(low=0.6, high=0.7)
                rand_y = np.random.uniform(low=-0.35, high=-0.1)
            else: 
                rand_x = np.random.uniform(low=0.7, high=0.8)
                rand_y = np.random.uniform(low=0.1, high=0.35)
        rand_x = np.random.uniform(low=0.6, high=0.83)
        rand_y = np.random.uniform(low=-0.35, high=0.35)
        t_anchor = np.linspace(start=0.6,stop=rand_x,num=2).reshape((-1,1))
        t_test = np.linspace(start=0.6,stop=rand_x,num=10).reshape((-1,1))
        self.grp.set_data(t_anchor    = t_anchor,
           x_anchor    = np.array([[0., rand_y]]).T,
           t_test      = t_test,
           l_anchor = np.array([[1,0.6]]).T,
           l_test   = np.ones((10,1)),
           hyp_mean    = {'g':0.1,'l':0.1,'w':1e-6},
           hyp_var     = {'g':0.1,'l':0.2,'w':1e-6},
        #    hyp_mean = {'g':0.1,'l':0.1,'w':1e-6},
        #     hyp_var  = {'g':0.04,'l':0.06,'w':1e-6},
           APPLY_EPSRU = False)
        y_trajs, x_traj = self.grp.sample(n_sample=n_sample)
        y_trajs = soft_squash(y_trajs,x_min=-0.35,x_max=0.35, margin=0.05)
        z_traj = np.array([[1.1], [1.1], [1.07], [1.06], [1.03], [1.0], [0.99], [0.98], [0.96], [0.92]])
        x_trajs = [x_traj[:] for _ in range(n_sample)]
        z_trajs = [z_traj[:] for _ in range(n_sample)]

        # self.grp.set_data(t_anchor = np.linspace(start=0.0,stop=1,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[0.6, rand_x]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.5,'l':0.5,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.5,'w':1e-6},
        #    APPLY_EPSRU = False)
        # x_trajs, _ = self.grp.sample(n_sample=n_sample)
        # x_trajs = soft_squash(x_trajs,x_min=0.6,x_max=0.8, margin=0.05)

        # self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[0., 0.]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,0.1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.1,'l':0.1,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.2,'w':1e-6},
        #    APPLY_EPSRU = False)
        # y_trajs, _ = self.grp.sample(n_sample=n_sample)
        # y_trajs = soft_squash(y_trajs,x_min=-0.33,x_max=0.33, margin=0.05)

        # self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=2).reshape((-1,1)),
        #    x_anchor    = np.array([[1.2, 1.0]]).T,
        #    t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
        #    l_anchor = np.array([[1,1]]).T,
        #    l_test   = np.ones((10,1)),
        #    hyp_mean    = {'g':0.5,'l':0.5,'w':1e-6},
        #    hyp_var     = {'g':0.1,'l':0.5,'w':1e-6},
        #    APPLY_EPSRU = False)
        # z_trajs, _ = self.grp.sample(n_sample=n_sample)
        # z_trajs = soft_squash(z_trajs,x_min=0.9,x_max=1.2, margin=0.05)
        return x_trajs,y_trajs,z_trajs
    
    """ Posterior: Single trajectory """
    @torch.no_grad()
    def exploit_place(self, z=torch.randn(64,2),
                      c=torch.randn(64,4)):        
        # Get reconstructed anchors
        x_anchor_recon = self.cvae.zc_to_x_recon(z=z, c=c)        
        #x_anchor_recon = np.insert(torch2np(x_anchor_recon).reshape(-1), 0,0)
        x_anchor_recon = x_anchor_recon.reshape(-1)
        t_anchor = np.linspace(start=0.6,stop=0.8,num=10).reshape((-1,1))
        t_test = np.linspace(start=0.6,stop=0.8,num=10).reshape((-1,1))
        self.grp.set_data(t_anchor    = t_anchor, 
                          x_anchor    = np.array([x_anchor_recon.numpy()]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = t_test,
                          l_test      = np.ones((len(t_test),1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
                        #   hyp_mean    = {'g':0.2,'l':0.08,'w':1e-6},
                        #   hyp_var     = {'g':0.2,'l':0.08,'w':1e-6})
        ys, xs = self.grp.mean_traj(x_min=-0.33,x_max=0.33,margin=0.05, SQUASH=True)
        zs = np.array([[1.1], [1.1], [1.07], [1.06], [1.03], [1.0], [0.99], [0.98], [0.96], [0.92]])
        return xs, ys, zs

    def exploit(self, z=torch.randn(64,2),
                      c=torch.randn(64,4)):        
        # Get reconstructed anchors
        x_anchor_recon = self.cvae.zc_to_x_recon(z=z, c=c)        
        #x_anchor_recon = np.insert(torch2np(x_anchor_recon).reshape(-1), 0,0)
        x_anchor_recon = x_anchor_recon.reshape(-1).detach().numpy()  
        t_anchor = np.linspace(start=0.65,stop=1.0,num=10).reshape((-1,1))
        self.grp.set_data(t_anchor    = t_anchor, 
                          x_anchor    = np.array([x_anchor_recon]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = t_anchor,
                          l_test      = np.ones((len(t_anchor),1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
                        #   hyp_mean    = {'g':0.2,'l':0.08,'w':1e-6},
                        #   hyp_var     = {'g':0.2,'l':0.08,'w':1e-6})
        ys, xs = self.grp.mean_traj(x_min=-0.33,x_max=0.33,margin=0.05, SQUASH=True)
        return xs, ys, x_anchor_recon
 
    @torch.no_grad()
    def exploit_3d(self, z=torch.randn(64,2),
                      c=torch.randn(64,4)):        
        # Get reconstructed anchors
        x_anchor_recon = self.cvae.zc_to_x_recon(z=z, c=c)  
        x_anchor_recon = x_anchor_recon.reshape(10,-1).T.numpy()  
        self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
                          x_anchor    = np.array([x_anchor_recon[0,:]]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                          l_test      = np.ones((10,1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        xs, _ = self.grp.mean_traj(x_min=0.6,x_max=0.83, margin=0.01)

        self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
                          x_anchor    = np.array([x_anchor_recon[1,:]]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                          l_test      = np.ones((10,1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        ys, _ = self.grp.mean_traj(x_min=-0.33,x_max=0.33, margin=0.01)

        self.grp.set_data(t_anchor    = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)), 
                          x_anchor    = np.array([x_anchor_recon[2,:]]).T,
                          l_anchor    = np.array([[1,1,1,1,1,1,1,1,1,1]]).T,
                          t_test      = np.linspace(start=0.0,stop=1.0,num=10).reshape((-1,1)),
                          l_test      = np.ones((10,1)),
                          hyp_mean    = {'g':0.05,'l':0.2,'w':1e-6},
                          hyp_var     = {'g':0.05,'l':0.2,'w':1e-6})
        zs, _ = self.grp.mean_traj(x_min=0.9,x_max=1.2, margin=0.01)
        return xs,ys,zs,x_anchor_recon

    # def get_reward(self, obs_state_lst):
    #     reward_scale = 0.1875
    #     reward_lst=[];result_lst=[]
    #     for each_worker_obs_state in obs_state_lst:
    #         total_reward=0; min_position_y=100
    #         for obs_idx in range(len(each_worker_obs_state)):
    #             position = each_worker_obs_state[obs_idx]["position"]
    #             if -0.07<position[1] and position[1]<0.07:
    #                 result=0
    #                 total_reward=-5
    #                 break 
    #             else:
    #                 result=1
    #                 if abs(position[1])<min_position_y: 
    #                     min_position_y=abs(position[1])
    #                     min_position = position 
    #                 total_reward+= abs(min_position[1])*10*reward_scale
    #         if total_reward>=8*reward_scale: # Reward limit 
    #             total_reward=8*reward_scale
    #         reward_lst.append(total_reward)
    #         result_lst.append(result)
    #     return reward_lst, result_lst               
