import numpy as np
import math
import mujoco
from env.mujoco.mujoco_utils import r2w, trim_scale  # Assuming utils.py contains necessary transformation functions

class MujocoKinematics:
    """
    Kinematics class for MuJoCo simulation.
    """
    def __init__(self, model, data):
        """
        Initialize the MuJoCo kinematics.
        
        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model
        data : mujoco.MjData
            MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Parse and store model information
        self._parse_model_info()
    
    def _parse_model_info(self):
        """Parse and store model information."""
        # Geometry info
        self.n_geom = self.model.ngeom
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, x)
                         for x in range(self.model.ngeom)]
        
        # Body info
        self.n_body = self.model.nbody
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, x)
                         for x in range(self.n_body)]
        self.body_name_idx = [self.body_names.index(x) for x in self.body_names]
        
        # Joint info
        self.n_dof = self.model.nv
        self.n_joint = self.model.njnt
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtJoint.mjJNT_HINGE, x)
                         for x in range(self.n_joint)]
        self.joint_types = self.model.jnt_type
        self.joint_ranges = self.model.jnt_range
        
        # Revolute joint info
        self.rev_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint = len(self.rev_joint_idxs)
        self.rev_joint_mins = self.joint_ranges[self.rev_joint_idxs, 0]
        self.rev_joint_maxs = self.joint_ranges[self.rev_joint_idxs, 1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        
        # Prismatic joint info
        self.pri_joint_idxs = np.where(self.joint_types == mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.pri_joint_mins = self.joint_ranges[self.pri_joint_idxs, 0]
        self.pri_joint_maxs = self.joint_ranges[self.pri_joint_idxs, 1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.n_pri_joint = len(self.pri_joint_idxs)
        
        # Actuator info
        self.n_ctrl = self.model.nu
        # Hardcoded information from original code
        self.ctrl_names = ['joint0', 'joint1']
        self.ctrl_joint_idxs = [0, 1]
        self.ctrl_joint_names = ['joint0', 'joint1']
        self.ctrl_qpos_idxs = self.ctrl_joint_idxs
        self.ctrl_qvel_idxs = [0, 1]
        self.ctrl_ranges = self.model.actuator_ctrlrange
        
        # Sensor info
        self.n_sensor = self.model.nsensor
        self.sensor_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, x)
                          for x in range(self.n_sensor)]
        
        # Site info
        self.n_site = self.model.nsite
        self.site_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, x)
                        for x in range(self.n_site)]
        
        # IK-related indices
        self.idxs_forward = [self.model.joint(joint_name).qposadr[0] for joint_name in self.rev_joint_names[:2]]
        self.idxs_jacobian = [self.model.joint(joint_name).dofadr[0] for joint_name in self.rev_joint_names[:2]]
        list1, list2 = self.ctrl_joint_idxs, self.idxs_forward
        self.idxs_step = []
        for i in range(len(list2)):
            if list2[i] in list1:
                self.idxs_step.append(list1.index(list2[i]))
    
    # --- Basic kinematics methods ---
    
    def get_p_body(self, body_name):
        """Get body position."""
        return self.data.body(body_name).xpos.copy()
    
    def get_R_body(self, body_name):
        """Get body rotation matrix."""
        return self.data.body(body_name).xmat.reshape([3, 3]).copy()
    
    def get_pR_body(self, body_name):
        """Get body position and rotation matrix."""
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p, R
    
    def get_p_joint(self, joint_name):
        """Get joint position."""
        body_id = self.model.joint(joint_name).bodyid[0]  # first body ID
        return self.get_p_body(self.body_names[body_id])
    
    def get_R_joint(self, joint_name):
        """Get joint rotation matrix."""
        body_id = self.model.joint(joint_name).bodyid[0]  # first body ID
        return self.get_R_body(self.body_names[body_id])
    
    def get_pR_joint(self, joint_name):
        """Get joint position and rotation matrix."""
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p, R
    
    def get_q(self, joint_idxs=None):
        """Get joint position in (radian)."""
        if joint_idxs is None:
            q = self.data.qpos
        else:
            q = self.data.qpos[joint_idxs]
        return q.copy()
    
    def get_J_body(self, body_name):
        """Get Jacobian matrices of a body."""
        J_p = np.zeros((3, self.model.nv))  # nv: nDoF
        J_R = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, J_p, J_R, self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p, J_R]))
        return J_p, J_R, J_full
    
    def get_qpos_joint(self, joint_name):
        """Get joint position."""
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self, joint_name):
        """Get joint velocity."""
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: 
            L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel
    
    def get_qpos_joints(self, joint_names):
        """Get multiple joint positions from 'joint_names'."""
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_qvel_joints(self, joint_names):
        """Get multiple joint velocities from 'joint_names'."""
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
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
        if q is not None:
            if joint_idxs is not None:
                self.data.qpos[joint_idxs] = q
            else:
                self.data.qpos = q
                
        mujoco.mj_forward(self.model, self.data)
    
    # --- IK methods ---
    
    def get_ik_ingredients(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True, w_weight=1):
        """
        Calculate Jacobian and error for IK.
        
        Parameters
        ----------
        body_name : str
            Name of the target body
        p_trgt : np.ndarray, optional
            Target position
        R_trgt : np.ndarray, optional
            Target rotation matrix
        IK_P : bool, default=True
            Whether to include position in IK
        IK_R : bool, default=True
            Whether to include rotation in IK
        w_weight : float, default=1.0
            Weight for rotation error
            
        Returns
        -------
        J : np.ndarray
            Jacobian matrix
        err : np.ndarray
            Error vector
        """
        J_p, J_R, J_full = self.get_J_body(body_name=body_name)
        p_curr, R_curr = self.get_pR_body(body_name=body_name)
        
        if IK_P and IK_R:
            p_err = (p_trgt - p_curr)
            R_err = np.linalg.solve(R_curr, R_trgt)
            w_err = R_curr @ r2w(R_err)
            J = J_full
            err = np.concatenate((p_err, w_weight * w_err))
        elif IK_P and not IK_R:
            p_err = (p_trgt - p_curr)
            J = J_p
            err = p_err
        elif not IK_P and IK_R:
            R_err = np.linalg.solve(R_curr, R_trgt)
            w_err = R_curr @ r2w(R_err)
            J = J_R
            err = w_err
        else:
            J = None
            err = None
            
        return J, err
    
    def damped_ls(self, J, err, eps=1e-6, stepsize=1.0, th=5*np.pi/180.0):
        """
        Damped least squares solver for IK.
        
        Parameters
        ----------
        J : np.ndarray
            Jacobian matrix
        err : np.ndarray
            Error vector
        eps : float, default=1e-6
            Damping factor
        stepsize : float, default=1.0
            Step size for update
        th : float, default=5*np.pi/180.0
            Threshold for joint updates
            
        Returns
        -------
        dq : np.ndarray
            Joint angle updates
        """
        # Calculate joint updates using damped least squares
        dq = stepsize * np.linalg.solve(
            a=(J.T @ J) + eps * np.eye(J.shape[1]), 
            b=J.T @ err
        )
        
        # Trim updates to prevent excessive movement
        dq = trim_scale(x=dq, th=th)
        return dq
    
    def onestep_ik(self, body_name, p_trgt=None, R_trgt=None, IK_P=True, IK_R=True,
                  joint_idxs=None, stepsize=1, eps=1e-1, th=5*np.pi/180.0):
        """
        Solve IK for a single step.
        
        Parameters
        ----------
        body_name : str
            Name of the end effector body
        p_trgt : np.ndarray, optional
            Target position
        R_trgt : np.ndarray, optional
            Target rotation matrix
        IK_P : bool, default=True
            Whether to include position in IK
        IK_R : bool, default=True
            Whether to include rotation in IK
        joint_idxs : List[int], optional
            Joint indices to update
        stepsize : float, default=1
            Step size for updates
        eps : float, default=1e-1
            Damping factor
        th : float, default=5*np.pi/180.0
            Threshold for joint updates
            
        Returns
        -------
        q : np.ndarray
            Updated joint angles
        err : np.ndarray
            Error vector
        """
        # Get Jacobian and error
        J, err = self.get_ik_ingredients(
            body_name=body_name, 
            p_trgt=p_trgt, 
            R_trgt=R_trgt, 
            IK_P=IK_P, 
            IK_R=IK_R
        )
        
        # Calculate joint updates
        dq = self.damped_ls(
            J=J, 
            err=err, 
            stepsize=stepsize, 
            eps=eps, 
            th=th
        )
        
        # Determine joint indices
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        
        # Get current joint positions
        q = self.get_q(joint_idxs=joint_idxs)
        
        # Update joint positions
        q = q + dq[joint_idxs]
        
        # Apply forward kinematics
        self.forward(q=q, joint_idxs=joint_idxs)
        
        return q, err
    
    def solve_ik(self, body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                RESET=False, DO_RENDER=False, render_every=1, th=1*np.pi/180.0, err_th=1e-6, 
                w_weight=1.0, stepsize=1.0, visualizer=None):
        """
        Solve inverse kinematics to reach a target pose.
        
        Parameters
        ----------
        body_name : str
            Name of the end effector body
        p_trgt : np.ndarray
            Target position
        R_trgt : np.ndarray
            Target rotation matrix
        IK_P : bool
            Whether to include position in IK
        IK_R : bool
            Whether to include rotation in IK
        q_init : np.ndarray
            Initial joint angles
        idxs_forward : List[int]
            Indices for forward kinematics
        idxs_jacobian : List[int]
            Indices for Jacobian
        RESET : bool, default=False
            Whether to reset data before solving
        DO_RENDER : bool, default=False
            Whether to render during solving
        render_every : int, default=1
            Render frequency
        th : float, default=π/180.0
            Joint update threshold
        err_th : float, default=1e-6
            Convergence error threshold
        w_weight : float, default=1.0
            Weight for rotation error
        stepsize : float, default=1.0
            Step size for updates
        visualizer : MujocoVisualizer, optional
            Visualizer instance for rendering during IK
            
        Returns
        -------
        q_ik : np.ndarray
            Solved joint angles
        """
        if RESET:
            mujoco.mj_resetData(self.model, self.data)
            
        # Save original joint angles
        q_backup = self.get_q(joint_idxs=idxs_forward)
        
        # Start from initial joint configuration
        q = q_init.copy()
        self.forward(q=q, joint_idxs=idxs_forward)
        
        # Iterative IK solving
        iteration = 0
        max_iterations = 500
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get Jacobian and error
            J, err = self.get_ik_ingredients(
                body_name=body_name, 
                p_trgt=p_trgt, 
                R_trgt=R_trgt, 
                IK_P=IK_P, 
                IK_R=IK_R, 
                w_weight=w_weight
            )
            
            # Calculate joint updates
            dq = self.damped_ls(
                J=J, 
                err=err, 
                stepsize=stepsize, 
                eps=1e-1, 
                th=th
            )
            
            # Update joint angles
            q = q + dq[idxs_jacobian]
            
            # Apply forward kinematics
            self.forward(q=q, joint_idxs=idxs_forward)
            
            # Check convergence
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
            
            # Render if requested
            if DO_RENDER and visualizer is not None:
                if ((iteration - 1) % render_every) == 0:
                    p_current, R_current = self.get_pR_body(body_name=body_name)
                    visualizer.plot_T(
                        p=p_current, R=R_current, 
                        PLOT_AXIS=True, axis_len=0.1, axis_width=0.005
                    )
                    visualizer.plot_T(
                        p=p_trgt, R=R_trgt, 
                        PLOT_AXIS=True, axis_len=0.2, axis_width=0.005
                    )
                    visualizer.render(render_every=render_every)
        
        # Get final solution
        q_ik = self.get_q(joint_idxs=idxs_forward)
        
        # Restore original configuration
        self.forward(q=q_backup, joint_idxs=idxs_forward)
        
        return q_ik

    def solve_ik_repel(self, body_name, p_trgt, R_trgt, IK_P, IK_R, q_init, idxs_forward, idxs_jacobian,
                      RESET=False, DO_RENDER=False, render_every=1, th=1*np.pi/180.0, err_th=1e-6,
                      w_weight=1.0, stepsize=1.0, eps=0.1, repulse=30, VERBOSE=False, visualizer=None):
        """
        Solve IK with collision avoidance.
        
        Parameters
        ----------
        body_name : str
            Name of the end effector body
        p_trgt : np.ndarray
            Target position
        R_trgt : np.ndarray
            Target rotation matrix
        IK_P : bool
            Whether to include position in IK
        IK_R : bool
            Whether to include rotation in IK
        q_init : np.ndarray
            Initial joint angles
        idxs_forward : List[int]
            Indices for forward kinematics
        idxs_jacobian : List[int]
            Indices for Jacobian
        RESET : bool, default=False
            Whether to reset data before solving
        DO_RENDER : bool, default=False
            Whether to render during solving
        render_every : int, default=1
            Render frequency
        th : float, default=π/180.0
            Joint update threshold
        err_th : float, default=1e-6
            Convergence error threshold
        w_weight : float, default=1.0
            Weight for rotation error
        stepsize : float, default=1.0
            Step size for updates
        eps : float, default=0.1
            Damping factor
        repulse : float, default=30
            Repulsion factor for collision avoidance
        VERBOSE : bool, default=False
            Whether to print debug information
        visualizer : MujocoVisualizer, optional
            Visualizer instance for rendering during IK
            
        Returns
        -------
        q_ik : np.ndarray
            Solved joint angles
        """
        if RESET:
            mujoco.mj_resetData(self.model, self.data)
            
        # Save original joint angles
        q_backup = self.get_q(joint_idxs=idxs_forward)
        
        # Start from initial joint configuration
        q = q_init.copy()
        self.forward(q=q, joint_idxs=idxs_forward)
        
        # Iterative IK solving
        iteration = 0
        max_iterations = 500
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get Jacobian and error
            J, err = self.get_ik_ingredients(
                body_name=body_name, 
                p_trgt=p_trgt, 
                R_trgt=R_trgt, 
                IK_P=IK_P, 
                IK_R=IK_R, 
                w_weight=w_weight
            )
            
            # Calculate joint updates
            dq = self.damped_ls(
                J=J, 
                err=err, 
                stepsize=stepsize, 
                eps=eps, 
                th=th
            )
            
            # Clip joint updates to avoid large steps
            clipped_dq = np.clip(dq[idxs_jacobian], -0.1, 0.1)
            q = q + clipped_dq
            
            # Apply joint limits
            q = np.clip(
                q, 
                self.joint_ranges[idxs_forward, 0], 
                self.joint_ranges[idxs_forward, 1]
            )
            
            # Apply forward kinematics
            self.forward(q=q, joint_idxs=idxs_forward)
            
            # Check for collisions
            p_contacts, f_contacts, geom1s, geom2s, body1s, body2s = self.get_contact_info(
                must_exclude_prefix='obj_')

            body1s_filtered = [obj_ for obj_ in body1s if obj_ not in [
                "rg2_gripper_finger1_finger_tip_link", "rg2_gripper_finger2_finger_tip_link"]]
            body2s_filtered = [obj_ for obj_ in body2s if obj_ not in [
                "rg2_gripper_finger1_finger_tip_link", "rg2_gripper_finger2_finger_tip_link"]]
            
            if len(body1s_filtered) > 0:
                if VERBOSE:
                    print(f"Collision detected with {body1s_filtered[0]} and {body2s_filtered}")
                
                # Apply repulsion in the opposite direction
                clipped_dq = np.clip(dq[idxs_jacobian], -0.1, 0.1)
                q = q - clipped_dq * repulse
                
                # Apply joint limits again
                q = np.clip(
                    q, 
                    self.joint_ranges[idxs_forward, 0], 
                    self.joint_ranges[idxs_forward, 1]
                )
                
                # Apply forward kinematics again
                self.forward(q=q, joint_idxs=idxs_forward)
            
            # Check convergence
            err_norm = np.linalg.norm(err)
            if err_norm < err_th:
                break
                
            # Render if requested
            if DO_RENDER and visualizer is not None:
                if ((iteration - 1) % render_every) == 0:
                    p_current, R_current = self.get_pR_body(body_name=body_name)
                    visualizer.plot_T(
                        p=p_current, R=R_current, 
                        PLOT_AXIS=True, axis_len=0.1, axis_width=0.005
                    )
                    visualizer.plot_T(
                        p=p_trgt, R=R_trgt, 
                        PLOT_AXIS=True, axis_len=0.2, axis_width=0.005
                    )
                    visualizer.render(render_every=render_every)
                    
                    if VERBOSE:
                        visualizer.plot_T(
                            p=np.array([0, 0, 2.5]), 
                            R=np.eye(3, 3),
                            PLOT_AXIS=False, 
                            label=f'[{err_norm:.4f}] err'
                        )
        
        # Get final solution
        q_ik = self.get_q(joint_idxs=idxs_forward)
        
        # Restore original configuration
        self.forward(q=q_backup, joint_idxs=idxs_forward)
        
        return q_ik

    def solve_augmented_ik(self, ik_body_names, ik_p_trgts, ik_R_trgts,
                           IK_Ps, IK_Rs, q_init, idxs_forward, idxs_jacobian,
                           RESET=False, DO_RENDER=False, render_every=1, 
                           th=1*np.pi/180.0, err_th=1e-6, visualizer=None):
        """
        Solve IK for multiple targets simultaneously.
        
        Parameters
        ----------
        ik_body_names : List[str]
            Names of target bodies
        ik_p_trgts : List[np.ndarray]
            Target positions
        ik_R_trgts : List[np.ndarray]
            Target rotation matrices
        IK_Ps : List[bool]
            Whether to include position for each target
        IK_Rs : List[bool]
            Whether to include rotation for each target
        q_init : np.ndarray
            Initial joint angles
        idxs_forward : List[int]
            Indices for forward kinematics
        idxs_jacobian : List[int]
            Indices for Jacobian
        RESET : bool, default=False
            Whether to reset data before solving
        DO_RENDER : bool, default=False
            Whether to render during solving
        render_every : int, default=1
            Render frequency
        th : float, default=π/180.0
            Joint update threshold
        err_th : float, default=1e-6
            Convergence error threshold
        visualizer : MujocoVisualizer, optional
            Visualizer instance for rendering during IK
            
        Returns
        -------
        q_ik : np.ndarray
            Solved joint angles
        """
        if RESET:
            mujoco.mj_resetData(self.model, self.data)
            
        # Save original joint angles
        q_backup = self.get_q(joint_idxs=idxs_forward)
        
        # Start from initial joint configuration
        q = q_init.copy()
        self.forward(q=q, joint_idxs=idxs_forward)
        
        # Iterative IK solving
        iteration = 0
        max_iterations = 500
        
        while iteration < max_iterations:
            iteration += 1
            
            # Collect Jacobian and error for all targets
            J_aug, err_aug = [], []
            
            for i, body_name in enumerate(ik_body_names):
                p_trgt = ik_p_trgts[i]
                R_trgt = ik_R_trgts[i]
                IK_P = IK_Ps[i]
                IK_R = IK_Rs[i]
                
                J, err = self.get_ik_ingredients(
                    body_name=body_name,
                    p_trgt=p_trgt,
                    R_trgt=R_trgt,
                    IK_P=IK_P,
                    IK_R=IK_R
                )
                
                if (J is None) or (err is None):
                    continue
                    
                if len(J_aug) == 0:
                    J_aug, err_aug = J, err
                else:
                    J_aug = np.concatenate((J_aug, J), axis=0)
                    err_aug = np.concatenate((err_aug, err), axis=0)
            
            # Calculate joint updates
            dq = self.damped_ls(
                J=J_aug, 
                err=err_aug, 
                stepsize=1.0, 
                eps=1e-1, 
                th=th
            )
            
            # Update joint angles
            q = q + dq[idxs_jacobian]
            
            # Apply forward kinematics
            self.forward(q=q, joint_idxs=idxs_forward)
            
            # Check convergence
            err_norm = np.linalg.norm(err_aug)
            if err_norm < err_th:
                break
                
            # Render if requested
            if DO_RENDER and visualizer is not None:
                if ((iteration - 1) % render_every) == 0:
                    # Visualize all targets
                    for i, body_name in enumerate(ik_body_names):
                        p_trgt = ik_p_trgts[i]
                        R_trgt = ik_R_trgts[i]
                        IK_P = IK_Ps[i]
                        IK_R = IK_Rs[i]
                        
                        if (not IK_P) and (not IK_R):
                            continue
                            
                        # Show current body position and target
                        visualizer.plot_T(
                            p=self.get_pR_body(body_name=body_name)[0],
                            R=self.get_pR_body(body_name=body_name)[1],
                            PLOT_AXIS=IK_R,
                            axis_len=0.2,
                            axis_width=0.01,
                            PLOT_SPHERE=IK_P,
                            sphere_r=0.05,
                            sphere_rgba=[1, 0, 0, 0.9],
                            label=f'augmented error: {np.linalg.norm(err_aug)}'
                        )
                        
                        visualizer.plot_T(
                            p=p_trgt,
                            R=R_trgt,
                            PLOT_AXIS=IK_R,
                            axis_len=0.2,
                            axis_width=0.01,
                            PLOT_SPHERE=IK_P,
                            sphere_r=0.05,
                            sphere_rgba=[0, 0, 1, 0.9]
                        )
                        
                    # Show world frame
                    visualizer.plot_T(
                        p=[0, 0, 0],
                        R=np.eye(3, 3),
                        PLOT_AXIS=True,
                        axis_len=1.0
                    )
                    
                    visualizer.render()
        
        # Get final solution
        q_ik = self.get_q(joint_idxs=idxs_forward)
        
        # Restore original configuration
        self.forward(q=q_backup, joint_idxs=idxs_forward)
        
        return q_ik
    
    def get_contact_info(self, must_exclude_prefix=None):
        """
        Get contact information from the simulation.
        
        Parameters
        ----------
        must_exclude_prefix : str, optional
            Prefix to exclude from contacts
            
        Returns
        -------
        p_contacts : List[np.ndarray]
            Contact positions
        f_contacts : List[np.ndarray]
            Contact forces
        geom1s : List[str]
            First geom names
        geom2s : List[str]
            Second geom names
        body1s : List[str]
            First body names
        body2s : List[str]
            Second body names
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        
        for i in range(self.data.ncon):
            # Get contact
            contact = self.data.contact[i]
            
            # Get geom1 info
            geom1_id = contact.geom1
            geom1_name = self.geom_names[geom1_id]
            body1_id = self.model.geom_bodyid[geom1_id]
            body1_name = self.body_names[body1_id]
            
            # Get geom2 info
            geom2_id = contact.geom2
            geom2_name = self.geom_names[geom2_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            body2_name = self.body_names[body2_id]
            
            # Skip if any name contains the excluded prefix
            if must_exclude_prefix is not None:
                if (must_exclude_prefix in body1_name) or (must_exclude_prefix in body2_name):
                    continue
            
            # Get contact position
            p_contact = np.zeros(3)
            mujoco.mj_contactForce(self.model, self.data, i, p_contact)
            
            # Get contact force
            f_contact = self.data.contact[i].dist
            
            # Append information
            p_contacts.append(p_contact)
            f_contacts.append(f_contact)
            geom1s.append(geom1_name)
            geom2s.append(geom2_name)
            body1s.append(body1_name)
            body2s.append(body2_name)
            
        return p_contacts, f_contacts, geom1s, geom2s, body1s, body2s
    
    def inverse_kinematics(self, x, y, L1, L2):
        """
        Find the joint angles given end-effector position (x, y).
        
        Parameters
        ----------
        x : float
            X coordinate
        y : float
            Y coordinate
        L1 : float
            Length of first link
        L2 : float
            Length of second link
            
        Returns
        -------
        tuple
            ((theta1_1, theta2_1), (theta1_2, theta2_2)) - two possible solutions
        """
        # Distance from origin to end-effector
        r = math.sqrt(x**2 + y**2)
        
        # Check if the point is reachable
        if r > L1 + L2:
            print("The point is not reachable.")
            return None

        # Calculate theta2 using the law of cosines
        cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta2_1 = math.acos(cos_theta2)  # Elbow up
        theta2_2 = -math.acos(cos_theta2)  # Elbow down
        
        # Calculate theta1
        theta1_1 = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2_1), L1 + L2 * math.cos(theta2_1))
        theta1_2 = math.atan2(y, x) - math.atan2(L2 * math.sin(theta2_2), L1 + L2 * math.cos(theta2_2))
        
        return ((theta1_1, theta2_1), (theta1_2, theta2_2))