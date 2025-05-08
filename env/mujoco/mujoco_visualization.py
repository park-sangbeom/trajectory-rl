import numpy as np
import cv2
import mujoco
import mujoco_viewer
from env.mujoco.mujoco_utils import rpy2r  # Assuming utils.py contains necessary transformation functions

class MujocoVisualizer:
    """
    Visualization class for MuJoCo simulation.
    """
    def __init__(self, model, data, MODE="window", viewer_title='MuJoCo', 
                 viewer_width=1200, viewer_height=800, viewer_hide_menus=True):
        """
        Initialize the MuJoCo visualizer.
        
        Parameters
        ----------
        model : mujoco.MjModel
            MuJoCo model
        data : mujoco.MjData
            MuJoCo data
        MODE : str, default="window"
            Rendering mode, either "window" or "offscreen"
        viewer_title : str, default='MuJoCo'
            Title for the viewer window
        viewer_width : int, default=1200
            Width of the viewer window
        viewer_height : int, default=800
            Height of the viewer window
        viewer_hide_menus : bool, default=True
            Whether to hide the menus in the viewer
        """
        self.model = model
        self.data = data
        self.MODE = MODE
        self.viewer = None
        self.render_tick = 0
        
        self.init_viewer(
            MODE=MODE,
            viewer_title=viewer_title,
            viewer_width=viewer_width,
            viewer_height=viewer_height,
            viewer_hide_menus=viewer_hide_menus
        )
    
    def init_viewer(self, MODE="window", viewer_title='MuJoCo', viewer_width=1200, 
                   viewer_height=800, viewer_hide_menus=True):
        """Initialize the MuJoCo viewer."""
        self.MODE = MODE
        if MODE == "window":
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model, self.data, mode='window', title=viewer_title,
                width=viewer_width, height=viewer_height, hide_menus=viewer_hide_menus)
        elif MODE == "offscreen":
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, mode='offscreen')
    
    def update_viewer(self, azimuth=None, distance=None, elevation=None, lookat=None,
                     VIS_TRANSPARENT=None, VIS_CONTACTPOINT=None,
                     contactwidth=None, contactheight=None, contactrgba=None,
                     VIS_JOINT=None, jointlength=None, jointwidth=None, jointrgba=None,
                     CALL_MUJOCO_FUNC=True):
        """Update viewer properties."""
        if self.viewer is None:
            return
            
        if azimuth is not None:
            self.viewer.cam.azimuth = azimuth
        if distance is not None:
            self.viewer.cam.distance = distance
        if elevation is not None:
            self.viewer.cam.elevation = elevation
        if lookat is not None:
            self.viewer.cam.lookat = lookat
        if VIS_TRANSPARENT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = VIS_TRANSPARENT
        if VIS_CONTACTPOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = VIS_CONTACTPOINT
        if contactwidth is not None:
            self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None:
            self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None:
            self.model.vis.rgba.contactpoint = contactrgba
        if VIS_JOINT is not None:
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_JOINT
        if jointlength is not None:
            self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None:
            self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None:
            self.model.vis.rgba.joint = jointrgba
            
        # Call MuJoCo functions for immediate modification
        if CALL_MUJOCO_FUNC:
            # Forward
            mujoco.mj_forward(self.model, self.data)
            # Update scene and render
            mujoco.mjv_updateScene(
                self.model, self.data, self.viewer.vopt, self.viewer.pert, self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value, self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx)
    
    def get_viewer_cam_info(self, VERBOSE=False):
        """Get viewer camera information."""
        if self.viewer is None:
            return None, None, None, None
            
        cam_azimuth = self.viewer.cam.azimuth
        cam_distance = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat = self.viewer.cam.lookat.copy()
        
        if VERBOSE:
            print(f"cam_azimuth:[{cam_azimuth:.2f}] cam_distance:[{cam_distance:.2f}] "
                  f"cam_elevation:[{cam_elevation:.2f}] cam_lookat:{cam_lookat}]")
                  
        return cam_azimuth, cam_distance, cam_elevation, cam_lookat

    def render(self, render_every=1):
        """Render the scene."""
        if self.viewer is None:
            print(f"Viewer NOT initialized.")
            return None
            
        if self.MODE == "window":
            if ((self.render_tick % render_every) == 0) or (self.render_tick == 0):
                self.viewer.render()
            self.render_tick = self.render_tick + 1
        elif self.MODE == "offscreen":
            rgbd = self.grab_rgb_depth_img_offscreen()
            self.render_tick = self.render_tick + 1
            return rgbd
    
    def grab_image(self, resize_rate=None, interpolation=cv2.INTER_NEAREST):
        """Grab the rendered image."""
        if self.viewer is None:
            return None
            
        img = np.zeros((self.viewer.viewport.height, self.viewer.viewport.width, 3), dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport, self.viewer.scn, self.viewer.ctx)
        mujoco.mjr_readPixels(img, None, self.viewer.viewport, self.viewer.ctx)
        img = np.flipud(img)  # flip image
        
        # Resize
        if resize_rate is not None:
            h = int(img.shape[0] * resize_rate)
            w = int(img.shape[1] * resize_rate)
            img = cv2.resize(img, (w, h), interpolation=interpolation)
            
        return img.copy()
    
    def close_viewer(self):
        """Close the viewer."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def plot_sphere(self, p, r, rgba=[1, 1, 1, 1], label=''):
        """Add a sphere to the visualization."""
        if self.viewer is None:
            return
            
        self.viewer.add_marker(
            pos=p,
            size=[r, r, r],
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            label=label)
    
    def plot_T(self, p, R, PLOT_AXIS=True, axis_len=1.0, axis_width=0.01,
              PLOT_SPHERE=False, sphere_r=0.05, sphere_rgba=[1, 0, 0, 0.5], 
              axis_rgba=None, label=None):
        """Plot coordinate axes and/or spheres at a given pose."""
        if self.viewer is None:
            return
            
        if PLOT_AXIS:
            if axis_rgba is None:
                rgba_x = [1.0, 0.0, 0.0, 0.9]
                rgba_y = [0.0, 1.0, 0.0, 0.9]
                rgba_z = [0.0, 0.0, 1.0, 0.9]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
                
            # X axis
            R_x = R @ rpy2r(np.deg2rad([0, 0, 90])) @ rpy2r(np.pi/2 * np.array([1, 0, 0]))
            p_x = p + R_x[:, 2] * axis_len/2
            self.viewer.add_marker(
                pos=p_x,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[axis_width, axis_width, axis_len/2],
                mat=R_x,
                rgba=rgba_x,
                label=''
            )
            
            # Y axis
            R_y = R @ rpy2r(np.deg2rad([0, 0, 90])) @ rpy2r(np.pi/2 * np.array([0, 1, 0]))
            p_y = p + R_y[:, 2] * axis_len/2
            self.viewer.add_marker(
                pos=p_y,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[axis_width, axis_width, axis_len/2],
                mat=R_y,
                rgba=rgba_y,
                label=''
            )
            
            # Z axis
            R_z = R @ rpy2r(np.deg2rad([0, 0, 90])) @ rpy2r(np.pi/2 * np.array([0, 0, 1]))
            p_z = p + R_z[:, 2] * axis_len/2
            self.viewer.add_marker(
                pos=p_z,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[axis_width, axis_width, axis_len/2],
                mat=R_z,
                rgba=rgba_z,
                label=''
            )
            
        if PLOT_SPHERE:
            self.viewer.add_marker(
                pos=p,
                size=[sphere_r, sphere_r, sphere_r],
                rgba=sphere_rgba,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label='')
                
        if label is not None:
            self.viewer.add_marker(
                pos=p,
                size=[0.0001, 0.0001, 0.0001],
                rgba=[1, 1, 1, 0.01],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                label=label)
    
    def is_viewer_alive(self):
        """Check whether the viewer is alive."""
        return self.viewer is not None and self.viewer.is_alive