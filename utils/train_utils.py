import numpy as np
import torch
import random
import os 


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def init_before_training(seed=3407):
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benckmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    
def aug_fft_1d(y,mage):
    # mage = np.random.rand(y.shape[-2],1)
    y_fft = np.fft.fft(y)
    y_abs, y_pha = np.abs(y_fft), np.angle(y_fft)
    y_abs = np.fft.fftshift(y_abs,axes=(-1))

    y_abs_aug = np.fft.ifftshift(y_abs*mage,axes=(-1))
    y_aug = y_abs_aug * (np.e ** (1j * y_pha))
    y_aug = np.real(np.fft.ifft(y_aug))
    
    return y_aug

def aug_fft_numpy(ys):
    """
    Args:
        ys: [D,H] D H-dimension signal
    Outs:
        ys_aug: [D,H]
    """
    res = []
    d = ys.shape[-2]
    mages = np.random.rand(d,1)
    for i in range(d):
        res.append(aug_fft_1d(ys[i],mages[i]))
        
    return np.stack(res,axis=-2)
    
    
def save_dynamic_model(model, optimizer, savepath, name):
    torch.save({
            # "state_dict": model.module.state_dict(),
            "cfn_state_dict": model.ode_solver.func.state_cfn.state_dict(),
            "dfn_state_dict": model.ode_solver.func.state_dfn.state_dict(),
            "int_state_dict": model.ode_solver.func.intensity_fn.state_dict(),
            "adj_dict" : model.adjs.data,
            "init_state": model._init_hstate.data,
            "optim_state_dict": optimizer.state_dict(),
        }, os.path.join(savepath, name))
    
    
def save_policy_model(model, optimizer, savepath, name):
    torch.save({
            # "state_dict": model.module.state_dict(),
            "rep_state_dict": model.ode_solver.func.represent_fn.state_dict(),
            "pol_state_dict": model.ode_solver.func.policy_fn.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        }, os.path.join(savepath, name))
    
    
def load_dynamic_model(model, checkpt):
    model.ode_solver.func.state_cfn.load_state_dict(checkpt["cfn_state_dict"])
    model.ode_solver.func.state_dfn.load_state_dict(checkpt["dfn_state_dict"])
    model.ode_solver.func.intensity_fn.load_state_dict(checkpt["int_state_dict"])
    model.adjs.data = checkpt["adj_dict"]
    model._init_hstate.data = checkpt["init_state"]
    

def load_policy_model(model, checkpt):
    model.ode_solver.func.policy_fn.load_state_dict(checkpt["pol_state_dict"])
    model.ode_solver.func.represent_fn.load_state_dict(checkpt["rep_state_dict"])