import mujoco
import mujoco_viewer
import numpy as np 
import os
import cv2 
import torch 
import torch.nn.functional as F
import gym 
import wandb
import argparse
import torch.nn as nn 
import sys 
sys.path.append('..')
from model.dlpg.dlpg import DeepLatentPolicyGradient 
from model.dlpg.buffer import BufferClass
from model.utils.utils import np2torch, torch2np, get_runname, scaling, get_diversity
from env.sweep import SweepEnvrionment
from env.manipulator_agent import ManipulatorAgent

def main(args):
    # Set random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
     # Set logger 
    runname = get_runname() if args.runname=='None' else args.runname
    if args.WANDB:
        wandb.init(project = args.pname)
        wandb.run.name = runname   
    # Make a path directory of save weight folder 
    SAVE_WEIGHT_PATH = "./weights/dlpg/"+args.runname
    os.makedirs(SAVE_WEIGHT_PATH, exist_ok=True) 
    if args.RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
        
    xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
    agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
    agent.reset() # reset
    # Move tables and robot base
    agent.model.body('base_table').pos = np.array([0,0,0.395])
    agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
    agent.model.body('base').pos = np.array([0.18,0,0.79])
    init_pose= np.array([-0.73418, -1.08485, 2.7836, -1.699, 0.8366, 0])
    env = SweepEnvrionment(agent=agent, init_pose=init_pose, RENDER=args.RENDER, seed=args.random_seed)

    buffer    = BufferClass(xdim=args.xdim, cdim=args.cdim, buffer_limit=args.buffer_limit, device=args.device)
    dlpg      = DeepLatentPolicyGradient(xdim=args.xdim, cdim=args.cdim, zdim=args.zdim, hdims=args.hdims,
                                         actv_enc = nn.ReLU(),actv_dec = None, actv_out = nn.Tanh(), actv_q = nn.Softplus(), 
                                         spectral_norm = args.spectral_norm)
    optimizer = torch.optim.Adam(params=dlpg.cvae.parameters(), lr = args.lr, betas=(0.9, 0.99), eps=1e-4)
    traj_cnt  = 0 
    for epoch in range(args.total_epochs):
        obs    = env.reset()
        obs_randxys = obs[3:].reshape(6,-1)
        obs_randxs, obs_randys = obs_randxys[:,0],obs_randxys[:,1]
        c = env.one_hot.copy()
        # Epsgrdy
        one_to_zero = 1.0-(epoch/(args.total_epochs-1))
        exploration_rate = 0.8*one_to_zero 
        if (np.random.rand()>exploration_rate): # Exploitation
            z = torch.randn(size=(1, args.zdim)).to(args.device)
            # z = z/torch.norm(z)
            z = F.normalize(z, p=2, dim=1)
            xs, ys, _ = dlpg.exploit(c=np2torch(c, device=args.device).reshape(-1,args.cdim), z=z)
            # Execute the trajectory
            _, reward, _, _ = env.step_traj(xs=xs, ys=ys)
            _, anchor = dlpg.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
            buffer.store(x=anchor.reshape(-1), c=c, reward=reward)
            avg_reward = reward 
            max_reward = reward
            traj_cnt +=1 
        else: # Exploration
            total_reward = 0 
            max_reward = -10
            xs, ys_lst = dlpg.random_explore(n_sample=10)
            for ys in ys_lst:
                traj_cnt +=1 
                _, reward, _, _ = env.step_traj(xs=xs, ys=ys)
                if reward>max_reward:
                    max_reward = reward
                _, anchor = dlpg.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
                buffer.store(x=anchor.reshape(-1), c=c, reward=reward)
                total_reward+=reward
                env.manual_reset(obs_randxs, obs_randys)
            avg_reward = total_reward/args.n_sample
        print("[Epoch:{}][Avg Reward:{}][Max Reward:{}]".format(epoch+1, avg_reward, max_reward))  
        if args.WANDB:
            wandb.log({"train avg reward":avg_reward}, step=epoch+1)   

        # if traj_cnt>500:
        #     traj_cnt=0
        if (epoch+1)%50==0:
            print('updated.')
            unscaled_loss_recon_sum=0;loss_recon_sum=0;loss_kl_sum=0;n_batch_sum=0;train_unscaled_reward_sum=0
            # if ((epoch+1)>=1000): beta = args.beta
            # else: beta = 0.0
            for iter in range(args.MAXITER):
                if iter<args.MAXITER/4:
                    beta = 0.0
                else:
                    beta = args.beta
                batch = buffer.sample_batch(sample_method=args.sample_method,
                                            batch_size=args.batch_size)
                x_batch, c_batch, reward_batch = batch["x"], batch["c"], batch["reward"]
                total_loss_out,loss_info = dlpg.cvae.loss_total(x               = x_batch, 
                                                                c               = c_batch, 
                                                                q               = reward_batch, 
                                                                LOSS_TYPE       = 'L1+L2',
                                                                recon_loss_gain = args.recon_gain,
                                                                beta            = beta,
                                                                STOCHASTICITY   = True)
                optimizer.zero_grad()
                total_loss_out.backward()
                optimizer.step()
                n_batch        = x_batch.shape[0]
                loss_recon_sum = loss_recon_sum + n_batch*loss_info['loss_recon_out']
                unscaled_loss_recon_sum = unscaled_loss_recon_sum+n_batch*loss_info['unscaled_loss_recon_out']
                loss_kl_sum    = loss_kl_sum + n_batch*loss_info['loss_kl_out']
                n_batch_sum    = n_batch_sum + n_batch   
                unscaled_reward_batch_np = torch2np(batch["reward"])
                train_unscaled_reward_sum +=np.average(unscaled_reward_batch_np) 
            # Average loss during train
            loss_recon_avg, loss_kl_avg, unscaled_loss_recon_avg = (loss_recon_sum/n_batch_sum),(loss_kl_sum/n_batch_sum), (unscaled_loss_recon_sum/n_batch_sum)            
            print ("[%d/%d] DLPG updated. Total loss:[%.3f] (recon:[%.3f] kl:[%.3f])"%
            (epoch+1,args.total_epochs,loss_recon_avg+loss_kl_avg,loss_recon_avg,loss_kl_avg))
            # Evaluation 
            with torch.no_grad():
                eval_total_reward = 0
                eval_cnt = 0
                done = True 
                trajectories = []
                for i in range(10):
                    eval_xs=[]; eval_ys=[]; eval_zs=[]
                    obs = env.reset()
                    c =  env.one_hot.copy()
                    z = torch.randn(size=(1, args.zdim)).to(args.device)
                    # z = z/torch.norm(z)
                    z = F.normalize(z, p=2, dim=1)
                    xs, ys, _ = dlpg.exploit(c=np2torch(c, device=args.device).reshape(-1,args.cdim), z=z)
                    # Execute the trajectory
                    _, reward, done, _ = env.eval_step_traj(xs=xs, ys=ys)
                    for x,y in zip(xs,ys):
                        eval_xs.append(x[0]); eval_ys.append(y[0])
                    trajectories.append(np.array([eval_xs, eval_ys]).reshape(2,10))
                    eval_total_reward+=reward 
                    if done:
                        eval_cnt+=1
                diversity = get_diversity(trajectories)
                if args.WANDB:
                    wandb.log({"Total loss":loss_recon_avg+loss_kl_avg,
                    "recon_loss":loss_recon_avg,
                    "kl_loss":loss_kl_avg,
                    "unscaled_loss_recon":unscaled_loss_recon_avg,
                    "eval avg reward":eval_total_reward/10,
                    "SR":eval_cnt/10,
                    "diversity": diversity}, step=epoch+1)   
        if (epoch+1)%50==0 or (epoch+1)==(args.total_epochs-1):
            torch.save(dlpg.cvae.state_dict(),SAVE_WEIGHT_PATH+"/dlpg_{}.pth".format(epoch+1))
            print("WEIGHT SAVED.")

if __name__=="__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str, default='final')
    parser.add_argument('--WANDB', type=bool, default=True)
    parser.add_argument('--pname', type=str, default='place')
    parser.add_argument('--sample_method', type=str, default='random')
    parser.add_argument('--spectral_norm', type=bool, default=True)
    parser.add_argument('--xdim', type=int, default=20) 
    parser.add_argument('--cdim', type=int, default=3)
    parser.add_argument('--zdim', type=int, default=5)
    parser.add_argument("--hdims", nargs="+", default=[128, 128])    
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--total_epochs', type=int, default=10_000)
    parser.add_argument('--MAXITER', type=int, default=200)
    parser.add_argument('--recon_gain', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--RENDER', type=bool, default=True)
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--buffer_limit', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=device)
    args = parser.parse_args()

    main(args)