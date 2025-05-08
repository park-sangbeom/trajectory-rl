import mujoco 
import numpy as np 
import random as rd 
import matplotlib.pyplot as plt 
import json
import os 
import sys 
sys.path.append('../')
from env.manipulator_agent import ManipulatorAgent
from env.mujoco.mujoco_utils import * 

class SweepEnvrionment:
    def __init__(self, agent, init_pose, RENDER=False, seed=0):
        self.agent = agent
        self.init_pose = init_pose
        self.action_space = np.zeros(20,) # end-effector x,y * 10
        self.observation_space = np.zeros(15) # (obj position x,y *6 + scenarios = 12+3)
        self._max_episode_steps = 1 
        self.one_hot  = np.zeros((3)) # In case of old qsd, and sac, it should be 4 
        self.target_position = None
        self.max_x = 0
        self.RENDER = RENDER    
        self.seed = seed
        # Set seed 
        np.random.seed(self.seed)
        rd.seed(self.seed)
        self.obj_name_lst = ['obj_box_01','obj_box_02','obj_box_03',
                             'obj_box_04','obj_box_05','obj_box_06']

    def reset(self):
        self.agent.reset()
        self.initialize()
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def step(self, action):
        # Scaling 
        action = action.reshape(2,10)/20
        xs,ys = [], []
        x = 0.65; y = 0.0
        self.max_x = 0 
        # Relative position moving 
        for i in range(self.action_space.shape[0]//2):
            x += action[0,i]; y += action[1,i]
            x = np.clip(x, 0.65, 1.15); y = np.clip(y, -0.33, 0.33)
            if x>self.max_x:
                self.max_x = x
            xs.append([x]); ys.append([y])
        _, q_traj=self.agent.move_sweep_traj(ys=ys, xs=xs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y in zip(xs,ys):
                    self.agent.plot_sphere(p=[x[0],y[0], 0.85],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_reward()
        info = None 
        return obs, reward, done, info
    
    def render(self):
        self.agent.render()

    def get_obs(self):
        position_lst = self.get_position()
        obs = np.concatenate([self.one_hot, np.array(position_lst).flatten()])
        return obs
    
    def get_position(self):
        position_lst = []
        for body_name in self.obj_name_lst:
            body_position = self.agent.get_p_body(body_name)
            position_lst.append(body_position[:2])
        return position_lst

    # def get_reward(self):
    #     done=True
    #     obs_position_lst = self.get_position()
    #     obs_rpy_lst = self.get_rotation()
    #     total_reward = 0
    #     failing_cnt = 0
    #     for obs_position, obs_rpy in zip(obs_position_lst, obs_rpy_lst): 
    #         distance = self.minimum_distance(obs_position)
    #         if abs(obs_rpy[0]) >0.05:
    #             total_reward+=-2
    #             failing_cnt+=1
    #         # if distance<0.1:
    #         #     return (self.max_x-0.7)*5+failing_cnt*(-0.3), done
    #         # else:
    #         #     total_reward+=distance*10
    #         if distance<0.1:
    #             total_reward+=-5
    #             total_reward+=(self.max_x-0.7)*5
    #         if obs_position[1]<-0.45 or obs_position[1]>0.45:
    #             total_reward+=-10
    #         total_reward+=distance*10

    #     total_reward += failing_cnt*(-1)
    #     return total_reward, done

    def get_reward(self):
        done = True 
        obs_position_lst = self.get_position()
        obs_rpy_lst = self.get_rotation()
        total_reward = 0
        min_distance = 100
        for obs_position, obs_rpy in zip(obs_position_lst, obs_rpy_lst):
            distance = self.minimum_distance(obs_position)
            if abs(obs_rpy[0]) >0.05 or obs_position[1]<-0.45 or obs_position[1]>0.45:
                return 0, done 
            if min_distance>distance:
                min_distance = distance
        if min_distance<0.1:
            total_reward+=(self.max_x-0.65)/2
        else: 
            total_reward+=min_distance*3
            total_reward+=(self.max_x-0.65)/2
        return total_reward, done 


    def get_rotation(self):
        rpy_lst = []
        for body_name in self.obj_name_lst:
            body_rotation= self.agent.get_R_body(body_name)
            body_rpy = r2rpy(body_rotation)
            rpy_lst.append(body_rpy)
        return rpy_lst

    def minimum_distance(self, p):
        start_p = np.array([0.65, .0])  
        segment_length_squared = np.sum((self.target_position[:2] - start_p)**2)
        if segment_length_squared == 0:
            return np.sqrt(np.sum((p - start_p)**2))  # The segment is a point
        t = np.dot(p - start_p, self.target_position[:2] - start_p) / segment_length_squared
        t = max(0, min(1, t))
        projection = start_p + t * (self.target_position[:2] - start_p)
        return np.linalg.norm(p - projection)

    def manual_reset(self, obs_randxs, obs_randys):
        self.agent.reset()
        for obj_name, x, y in zip(self.agent.obj_name_lst, obs_randxs, obs_randys):
            jntadr = self.agent.model.body(obj_name).jntadr[0]
            self.agent.model.joint(jntadr).qpos0[:3] = np.array([x,y,0.74])
        jntadr = self.agent.model.body('target_box_01').jntadr[0]
        self.agent.model.joint(jntadr).qpos0[:3] = self.target_position
        self.agent.reset(q=np.array([-0.53, -1.43,  0.58,  1.96,  1.32, -0.47]))
        obs = self.get_obs()
        return np.array(obs).flatten()


    def initialize(self):
        scenario = 2 #rd.randint(1,3)  
        self.one_hot  = np.zeros((3))
        capture_pose= np.array([-0.53, -1.43,  0.58,  1.96,  1.32, -0.47])
        self.one_hot[scenario-1]=1 
        if scenario==1: # Left 
            obs_x1,obs_y1 = np.random.uniform(0.74, 0.84),np.random.uniform(-0.25,-0.22) 
            obs_x2,obs_y2 = np.random.uniform(0.85, 1),np.random.uniform(-0.2,-0.17) 
            obs_x3,obs_y3 = np.random.uniform(0.74, 0.84),np.random.uniform(-0.15,-0.12)
            obs_x4,obs_y4 = np.random.uniform(0.85, 1),np.random.uniform(-0.1,-0.07) 
            obs_x5,obs_y5 = np.random.uniform(0.74, 0.84),np.random.uniform(-0.02,0)
            obs_x6,obs_y6 = np.random.uniform(0.85, 1),np.random.uniform(0.05,0.08)
            self.target_position = np.array([1.25,-0.14,0.73])

        elif scenario==2: # Center 
            obs_x1,obs_y1 = np.random.uniform(0.74, 0.84),np.random.uniform(-0.3,-0.25) 
            obs_x2,obs_y2 = np.random.uniform(0.85, 1),np.random.uniform(-0.2,-0.15) 
            obs_x3,obs_y3 = np.random.uniform(0.74, 0.84),np.random.uniform(-0.1,-0.05)
            obs_x4,obs_y4 = np.random.uniform(0.85, 1),np.random.uniform(0.,0.05) 
            obs_x5,obs_y5 = np.random.uniform(0.74, 0.84),np.random.uniform(0.1,0.15)
            obs_x6,obs_y6 = np.random.uniform(0.85, 1),np.random.uniform(0.2,0.25)
            self.target_position = np.array([1.25,0.,0.73])

        elif scenario==3: # Right 
            obs_x1,obs_y1 = np.random.uniform(0.74, 0.84),np.random.uniform(0.22,0.25) 
            obs_x2,obs_y2 = np.random.uniform(0.85, 1),np.random.uniform(0.17,0.2) 
            obs_x3,obs_y3 = np.random.uniform(0.74, 0.84),np.random.uniform(0.12,0.15)
            obs_x4,obs_y4 = np.random.uniform(0.85, 1),np.random.uniform(0.07,0.1) 
            obs_x5,obs_y5 = np.random.uniform(0.74, 0.84),np.random.uniform(0.,0.02)
            obs_x6,obs_y6 = np.random.uniform(0.85, 1),np.random.uniform(-0.08,-0.05)
            self.target_position = np.array([1.25,0.14,0.73])

        obs_randxs = [obs_x1, obs_x2, obs_x3, obs_x4, obs_x5, obs_x6]
        obs_randys = [obs_y1, obs_y2, obs_y3, obs_y4, obs_y5, obs_y6]
        for obj_name, x, y in zip(self.agent.obj_name_lst, obs_randxs, obs_randys):
            jntadr = self.agent.model.body(obj_name).jntadr[0]
            self.agent.model.joint(jntadr).qpos0[:3] = np.array([x,y,0.74])
        jntadr = self.agent.model.body('target_box_01').jntadr[0]
        self.agent.model.joint(jntadr).qpos0[:3] = self.target_position
        self.agent.reset(q=capture_pose)

    def eval_step(self, action, save_path=None, file_name=None, epoch=0):
        # Scaling 
        action = action.reshape(2,10)/20
        xs,ys = [], []
        eval_xs, eval_ys = [], []
        self.max_x = 0 
        x = 0.65; y = 0.0
        # Relative position moving 
        for i in range(self.action_space.shape[0]//2):
            x += action[0,i]; y += action[1,i]
            x = np.clip(x, 0.65, 1.15); y = np.clip(y, -0.33, 0.33)
            if x>self.max_x:
                self.max_x = x
            xs.append([x]); ys.append([y])
            eval_xs.append(x); eval_ys.append(y)

        _, q_traj=self.agent.move_sweep_traj(ys=ys, xs=xs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y in zip(xs,ys):
                    self.agent.plot_sphere(p=[x[0],y[0], 0.85],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_eval_reward()
        info = None 
        if save_path is not None:
            # Save action 
            path = save_path+file_name
            with open(path, "a") as f:
                content = {"epoch":epoch,
                            "x":eval_xs,
                            "y":eval_ys,
                            'reward':reward,
                            'done':done}
                
                f.write(json.dumps(content)+'\n')

        return obs, reward, done, info
    
    def get_eval_reward(self):
        done = True 
        obs_position_lst = self.get_position()
        obs_rpy_lst = self.get_rotation()
        total_reward = 0
        min_distance = 100
        for obs_position, obs_rpy in zip(obs_position_lst, obs_rpy_lst):
            distance = self.minimum_distance(obs_position)
            if abs(obs_rpy[0]) >0.05 or obs_position[1]<-0.45 or obs_position[1]>0.45:
                done = False 
                return 0, done 
            if min_distance>distance:
                min_distance = distance
        if min_distance<0.1:
            total_reward+=(self.max_x-0.7)/2
            done = False 
        else: 
            total_reward+=min_distance*3
            total_reward+=(self.max_x-0.7)/2
        return total_reward, done 

    # def get_eval_reward(self):
    #     done=True
    #     obs_position_lst = self.get_position()
    #     obs_rpy_lst = self.get_rotation()
    #     total_reward = 0
    #     failing_cnt = 0
    #     for obs_position, obs_rpy in zip(obs_position_lst, obs_rpy_lst): 
    #         distance = self.minimum_distance(obs_position)
    #         if abs(obs_rpy[0]) >0.05:
    #             total_reward -=2
    #             failing_cnt+=1
    #         if distance<0.1:
    #             return (self.max_x-0.7)*5+failing_cnt*(-0.3), False
    #         else:
    #             total_reward+=distance*10
    #     total_reward += failing_cnt*(-1)
    #     if failing_cnt==0:
    #         done = True 
    #     else: done = False
    #     return total_reward, done

    def step_traj(self, xs, ys):
        _, q_traj=self.agent.move_sweep_traj(ys=ys, xs=xs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y in zip(xs,ys):
                    self.agent.plot_sphere(p=[x[0],y[0], 0.85],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_reward()
        info = None 
        return obs, reward, done, info

    def eval_step_traj(self, xs, ys, save_path=None, file_name=None, epoch=0):
        _, q_traj=self.agent.move_sweep_traj(ys=ys, xs=xs, vel=np.radians(10), HZ=500)
        print("q_traj", q_traj)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y in zip(xs,ys):
                    self.agent.plot_sphere(p=[x[0],y[0], 0.85],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_eval_reward()
        info = None 

        if save_path is not None:
            # Save action 
            path = save_path+file_name
            with open(path, "a") as f:
                content = {"epoch":epoch,
                            "x":xs.tolist(),
                            "y":ys.tolist(),
                            'reward':reward,
                            'done':done}
                
                f.write(json.dumps(content)+'\n')

        return obs, reward, done, info


if __name__=="__main__":
    xml_path = '../env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
        
    RENDER = False 
    if RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
        
    agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
    agent.reset() # reset
    sweep_init_pose= np.array([-0.73418, -1.08485, 2.7836, -1.699, 0.8366, 0])
    # Move tables and robot base
    agent.model.body('base_table').pos = np.array([0,0,0.395])
    agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
    agent.model.body('base').pos = np.array([0.18,0,0.79])
    print ("[UR5] parsed.")
    env = SweepEnvrionment(agent=agent, init_pose=sweep_init_pose, RENDER=RENDER)