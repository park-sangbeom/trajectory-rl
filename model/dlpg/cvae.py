import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from torch.nn.utils import spectral_norm

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


class ConditionalVariationalAutoEncoderClass(nn.Module):
    def __init__(
        self,
        name     = 'CVAE',              
        x_dim    = 784,              # input dimension
        c_dim    = 10,               # condition dimension
        z_dim    = 16,               # latent dimension
        h_dims   = [64,32],          # hidden dimensions of encoder (and decoder)
        actv_enc = nn.LeakyReLU(),   # encoder activation
        actv_dec = nn.LeakyReLU(),   # decoder activation
        actv_out = None,             # output activation
        actv_q   = nn.Softplus(),    # q activation
        var_max  = None,             # maximum variance
        device   = 'cpu',
        spectral_norm = False
        ):
        """
            Initialize
        """
        super(ConditionalVariationalAutoEncoderClass,self).__init__()
        self.name = name
        self.x_dim    = x_dim
        self.c_dim    = c_dim
        self.z_dim    = z_dim
        self.h_dims   = h_dims
        self.actv_enc = actv_enc
        self.actv_dec = actv_dec
        self.actv_out = actv_out
        self.actv_q   = actv_q
        self.var_max  = var_max
        self.device   = device
        self.sn       = spectral_norm
        # Initialize layers
        self.init_layers()
        self.init_params()
        
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        
        # Encoder part
        h_dim_prev = self.x_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['enc_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            if self.actv_enc is not None:
                self.layers['enc_%02d_actv'%(h_idx)] = \
                    self.actv_enc
            h_dim_prev = h_dim
        if self.sn:
            self.layers['z_mu_lin']  = spectral_norm(nn.Linear(h_dim_prev,self.z_dim,bias=True))
            self.layers['z_var_lin'] = spectral_norm(nn.Linear(h_dim_prev,self.z_dim,bias=True))
        else:
            self.layers['z_mu_lin']  = nn.Linear(h_dim_prev,self.z_dim,bias=True)
            self.layers['z_var_lin'] = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        # Decoder part
        h_dim_prev = self.z_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['dec_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            if self.actv_dec is not None:
                self.layers['dec_%02d_actv'%(h_idx)] = \
                    self.actv_dec
            h_dim_prev = h_dim
        self.layers['out_lin'] = nn.Linear(h_dim_prev,self.x_dim,bias=True)
        
        # Append parameters
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.cvae_parameters = nn.ParameterDict(self.param_dict)
        
    def xc_to_z_mu(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_mu
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            if self.actv_enc is not None:
                net = self.layers['enc_%02d_actv'%(h_idx)](net)
        z_mu = self.layers['z_mu_lin'](net)
        return z_mu
    
    def xc_to_z_var(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_var
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            if self.actv_enc is not None:
                net = self.layers['enc_%02d_actv'%(h_idx)](net)
        net = self.layers['z_var_lin'](net)
        if self.var_max is None:
            net = torch.exp(net)
        else:
            net = self.var_max*torch.sigmoid(net)
        z_var = net
        return z_var
    
    def zc_to_x_recon(
        self,
        z = torch.randn(2,16),
        c = torch.randn(2,10)
        ):
        """
            z and c to x_recon
        """
        if c is not None:
            net = torch.cat((z,c),dim=1)
        else:
            net = z
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            if self.actv_dec is not None:
                net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon
    
    def xc_to_z_sample(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_sample
        """
        z_mu,z_var = self.xc_to_z_mu(x=x,c=c),self.xc_to_z_var(x=x,c=c)
        eps_sample = torch.randn(
            size=z_mu.shape,dtype=torch.float32).to(self.device)
        z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        # Vmf
        if self.sn:
            z_sample   = F.normalize(z_sample, p=2,  dim = -1)
        return z_sample
    
    def xc_to_x_recon(
        self,
        x             = torch.randn(2,784),
        c             = torch.randn(2,10), 
        STOCHASTICITY = True
        ):
        """
            x and c to x_recon
        """
        if STOCHASTICITY:
            z_sample = self.xc_to_z_sample(x=x,c=c)
        else:
            z_sample = self.xc_to_z_mu(x=x,c=c)
        x_recon = self.zc_to_x_recon(z=z_sample,c=c)
        return x_recon
    
    def sample_x(
        self,
        c        = torch.randn(5,10),
        n_sample = 5
        ):
        """
            Sample x
        """
        z_sample = torch.randn(
            size=(n_sample,self.z_dim),dtype=torch.float32).to(self.device)
        return self.zc_to_x_recon(z=z_sample,c=c),z_sample
    
    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def loss_recon(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        q               = torch.randn(2,1),
        LOSS_TYPE       = 'L1',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True
        ):
        """
            Recon loss
        """
        EPS = 1e-15
        x_recon = self.xc_to_x_recon(x=x,c=c,STOCHASTICITY=STOCHASTICITY)
        if (LOSS_TYPE == 'L1') or (LOSS_TYPE == 'MAE'):
            errs = torch.mean(torch.abs(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L2') or (LOSS_TYPE == 'MSE'):
            errs = torch.mean(torch.square(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L1+L2') or (LOSS_TYPE == 'EN'):
            errs = torch.mean(
                0.5*(torch.abs(x+EPS-x_recon+EPS)+torch.square(x+EPS-x_recon+EPS)),axis=1)
        else:
            raise Exception("VAE:[%s] Unknown loss_type:[%s]"%
                            (self.name,LOSS_TYPE))
        if self.actv_q is not None: q = self.actv_q(q)
        unscaled_err = errs 
        errs = errs*q 
        return recon_loss_gain*torch.mean(errs), torch.mean(unscaled_err)
    
    def loss_kl(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10),
        q = torch.randn(2,1)
        ):
        """
            KLD loss
        """
        z_mu     = self.xc_to_z_mu(x=x,c=c)
        z_var    = self.xc_to_z_var(x=x,c=c)
        z_logvar = torch.log(z_var)
        errs     = 0.5*torch.sum(z_var + z_mu**2 - 1.0 - z_logvar,axis=1)
        if self.actv_q is not None: q = self.actv_q(q)
        errs     = errs*q
        return torch.mean(errs)
        
    def loss_total(
        self,
        x               = torch.randn(2,2),
        c               = torch.randn(2,5),
        q          = torch.randn(2,1),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True,
        beta            = 1.0
        ):
        """
            Total loss
        """
        loss_recon_out, unscaled_loss_recon_out = self.loss_recon(
                                        x               = x,
                                        c               = c,
                                        q               = q,
                                        LOSS_TYPE       = LOSS_TYPE,
                                        recon_loss_gain = recon_loss_gain,
                                        STOCHASTICITY   = STOCHASTICITY
                                        )
        loss_kl_out    = beta*self.loss_kl(x=x,c=c,q=q)
        loss_total_out = loss_recon_out + loss_kl_out
        info           = {'loss_recon_out' : loss_recon_out,
                          'loss_kl_out'    : loss_kl_out,
                          'loss_total_out' : loss_total_out,
                          'beta'           : beta,
                          "unscaled_loss_recon_out":unscaled_loss_recon_out}
        return loss_total_out,info