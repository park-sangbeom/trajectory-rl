import os
import numpy as np
import mujoco
from env.mujoco.mujoco_kinematics import MujocoKinematics
from env.mujoco.mujoco_visualization import MujocoVisualizer

class MujocoSim:
    """
    Main MuJoCo simulation class with integrated kinematics, visualization, and IK solving.
    """
    def __init__(self, xml_path, MODE="window"):
        """
        Initialize the MuJoCo simulation environment.
        
        Parameters
        ----------
        xml_path : str
            Path to the MuJoCo XML model file
        MODE : str, default="window"
            Rendering mode, either "window" or "offscreen"
        """
        self.xml_path = xml_path
        self.MODE = MODE
        
        # Parse XML and setup model
        self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(), self.xml_path))
        self.model = mujoco.MjModel.from_xml_path(self.full_xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize state
        self.qpos0 = self.data.qpos.copy()
        
        # Create kinematics handler
        self.kin = MujocoKinematics(self.model, self.data)
        
        # Reset simulation
        self.reset()
        
        # Create visualizer
        self.viz = MujocoVisualizer(self.model, self.data, MODE=self.MODE)
        
        # Simulation counters
        self.tick = 0
        
    def reset(self):
        """Reset the simulation to its initial state."""
        mujoco.mj_resetData(self.model, self.data)
        # To initial position
        self.data.qpos = self.qpos0.copy()
        mujoco.mj_forward(self.model, self.data)
        self.tick = 0
        
    def step(self, ctrl=None, ctrl_idxs=None, nstep=1, INCREASE_TICK=True):
        """
        Advance the simulation by one or more steps.
        
        Parameters
        ----------
        ctrl : np.ndarray, optional
            Control signals
        ctrl_idxs : List[int], optional
            Indices for control signals
        nstep : int, default=1
            Number of steps to take
        INCREASE_TICK : bool, default=True
            Whether to increase the tick counter
        """
        if ctrl is not None:
            if ctrl_idxs is None:
                self.data.ctrl[:] = ctrl
            else:
                self.data.ctrl[ctrl_idxs] = ctrl
                
        mujoco.mj_step(self.model, self.data, nstep=nstep)
        
        if INCREASE_TICK:
            self.tick = self.tick + 1 * nstep
    
    def forward(self, q=None, joint_idxs=None, INCREASE_TICK=True):
        """
        Perform forward kinematics.
        
        Parameters
        ----------
        q : np.ndarray, optional
            Joint angles
        joint_idxs : List[int], optional
            Indices of joints to set angles for
        INCREASE_TICK : bool, default=True
            Whether to increase the tick counter
        """
        self.kin.forward(q, joint_idxs)
        
        if INCREASE_TICK:
            self.tick = self.tick + 1
    
    def get_sim_time(self):
        """Get simulation time (sec)."""
        return self.data.time
    
    def loop_every(self, HZ=1):
        """Check if current tick is a multiple of a frequency."""
        FLAG = (self.tick-1) % (int(1/self.model.opt.timestep/HZ)) == 0
        return FLAG
    
    # --- Object handling methods ---
    
    def set_objects(self, xyzs_dict, COLORS=False, VERBOSE=True):
        """
        Set objects to the given positions.
        
        Parameters
        ----------
        xyzs_dict : Dict[str, np.ndarray]
            Dictionary of object names and positions
        COLORS : bool, default=False
            Whether to set colors
        VERBOSE : bool, default=True
            Whether to print information
        """
        n_obj = len(xyzs_dict.keys())
        # if COLORS:
        #     colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])

        for obj_idx, obj_name in enumerate(xyzs_dict.keys()):
            jntadr = self.model.body(obj_name).jntadr[0]
            self.model.joint(jntadr).qpos0[:3] = xyzs_dict[obj_name]
            if COLORS:
                geomadr = self.model.body(obj_name).geomadr[0]
                self.model.geom(geomadr).rgba = colors[obj_idx]

        if VERBOSE:
            for obj_idx, obj_name in enumerate(xyzs_dict.keys()):
                if obj_idx == (n_obj):
                    break
                print(f"{obj_name}: {xyzs_dict[obj_name]}")
    
    # --- Delegate visualization methods to the visualizer ---
    
    def render(self, render_every=1):
        """Render the scene."""
        return self.viz.render(render_every)
    
    def update_viewer(self, **kwargs):
        """Update viewer properties."""
        self.viz.update_viewer(**kwargs)
    
    def get_viewer_cam_info(self, VERBOSE=False):
        """Get viewer camera information."""
        return self.viz.get_viewer_cam_info(VERBOSE)
    
    def grab_image(self, resize_rate=None, interpolation=None):
        """Grab the rendered image."""
        if interpolation is None:
            import cv2
            interpolation = cv2.INTER_NEAREST
        return self.viz.grab_image(resize_rate, interpolation)
    
    def close_viewer(self):
        """Close the viewer."""
        self.viz.close_viewer()
    
    def plot_sphere(self, p, r, rgba=[1, 1, 1, 1], label=''):
        """Add a sphere to the visualization."""
        self.viz.plot_sphere(p, r, rgba, label)
    
    def plot_T(self, p, R, **kwargs):
        """Plot coordinate axes and/or spheres at a given pose."""
        self.viz.plot_T(p, R, **kwargs)
    
    def is_viewer_alive(self):
        """Check whether the viewer is alive."""
        return self.viz.is_viewer_alive()
    
    # --- Delegate kinematics methods to the kinematics handler ---
    
    def get_p_body(self, body_name):
        """Get body position."""
        return self.kin.get_p_body(body_name)
    
    def get_R_body(self, body_name):
        """Get body rotation matrix."""
        return self.kin.get_R_body(body_name)
    
    def get_pR_body(self, body_name):
        """Get body position and rotation matrix."""
        return self.kin.get_pR_body(body_name)
    
    def get_p_joint(self, joint_name):
        """Get joint position."""
        return self.kin.get_p_joint(joint_name)
    
    def get_R_joint(self, joint_name):
        """Get joint rotation matrix."""
        return self.kin.get_R_joint(joint_name)
    
    def get_pR_joint(self, joint_name):
        """Get joint position and rotation matrix."""
        return self.kin.get_pR_joint(joint_name)
    
    def get_q(self, joint_idxs=None):
        """Get joint position in (radian)."""
        return self.kin.get_q(joint_idxs)
    
    def get_J_body(self, body_name):
        """Get Jacobian matrices of a body."""
        return self.kin.get_J_body(body_name)
    
    def get_qpos_joint(self, joint_name):
        """Get joint position."""
        return self.kin.get_qpos_joint(joint_name)
    
    def get_qvel_joint(self, joint_name):
        """Get joint velocity."""
        return self.kin.get_qvel_joint(joint_name)
    
    def get_qpos_joints(self, joint_names):
        """Get multiple joint positions from 'joint_names'."""
        return self.kin.get_qpos_joints(joint_names)
    
    def get_qvel_joints(self, joint_names):
        """Get multiple joint velocities from 'joint_names'."""
        return self.kin.get_qvel_joints(joint_names)
    
    def get_contact_info(self, must_exclude_prefix=None):
        """Get contact information from the simulation."""
        return self.kin.get_contact_info(must_exclude_prefix)
    
    # --- IK methods from kinematics with integrated visualization ---
    
    def get_ik_ingredients(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True, w_weight=1):
        """Calculate Jacobian and error for IK."""
        return self.kin.get_ik_ingredients(body_name, p_trgt, R_trgt, IK_P, IK_R, w_weight)
    
    def damped_ls(self, J, err, eps=1e-6, stepsize=1.0, th=5*np.pi/180.0):
        """Damped least squares solver for IK."""
        return self.kin.damped_ls(J, err, eps, stepsize, th)
    
    def onestep_ik(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True,
                  joint_idxs=None, stepsize=1, eps=1e-1, th=5*np.pi/180.0):
        """Solve IK for a single step."""
        return self.kin.onestep_ik(body_name, p_trgt, R_trgt, IK_P, IK_R, 
                                  joint_idxs, stepsize, eps, th)
    
    def solve_ik(self, body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                RESET=False, DO_RENDER=False, render_every=1, th=1*np.pi/180.0, err_th=1e-6, 
                w_weight=1.0, stepsize=1.0):
        """Solve inverse kinematics to reach a target pose."""
        return self.kin.solve_ik(body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                                RESET, DO_RENDER, render_every, th, err_th, w_weight, stepsize, self.viz)
    
    def solve_ik_repel(self, body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                      RESET=False, DO_RENDER=False, render_every=1, th=1*np.pi/180.0, err_th=1e-6,
                      w_weight=1.0, stepsize=1.0, eps=0.1, repulse=30, VERBOSE=False):
        """Solve IK with collision avoidance."""
        return self.kin.solve_ik_repel(body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                                      RESET, DO_RENDER, render_every, th, err_th, w_weight, stepsize, eps, 
                                      repulse, VERBOSE, self.viz)
    
    def solve_augmented_ik(self, ik_body_names, ik_p_trgts, ik_R_trgts,
                           IK_Ps, IK_Rs, q_init, idxs_forward, idxs_jacobian,
                           RESET=False, DO_RENDER=False, render_every=1, 
                           th=1*np.pi/180.0, err_th=1e-6):
        """Solve IK for multiple targets simultaneously."""
        return self.kin.solve_augmented_ik(ik_body_names, ik_p_trgts, ik_R_trgts, IK_Ps, IK_Rs, 
                                         q_init, idxs_forward, idxs_jacobian, RESET, DO_RENDER, 
                                         render_every, th, err_th, self.viz)
    
    def inverse_kinematics(self, x, y, L1, L2):
        """Find the joint angles given end-effector position (x, y)."""
        return self.kin.inverse_kinematics(x, y, L1, L2)