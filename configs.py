import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


## change the root path to save model the logging
root_path = os.getcwd()
model_path = os.path.join(root_path, "pre_trained_models")


parser = argparse.ArgumentParser(description="Controlled Neural Jump Process")
parser.add_argument("--root_dir",type=str, default=root_path)


########### training parameters ###################
parser.add_argument("--epoches","-e",type=int, default=500)  
parser.add_argument("--pretrain_epo",type=int, default=200)
parser.add_argument("--warmup_epo",type=int, default=10)
parser.add_argument("--decay_epo",type=int, default=100)

# parser.add_argument("--epoches","-e",type=int, default=5)  
# parser.add_argument("--pretrain_epo",type=int, default=0)
# parser.add_argument("--warmup_epo",type=int, default=1)
# parser.add_argument("--decay_epo",type=int, default=3)


parser.add_argument("--resume_dyna", type=str2bool, default='false')
parser.add_argument("--resume_policy", type=str2bool, default='false')
parser.add_argument("--policy_learning", type=str2bool, default='true')
parser.add_argument("--dynamic_learning", type=str2bool, default='true')
parser.add_argument("--amortized_state_list", type=str, default="")


parser.add_argument("--lr_dyna",type=float, default=1e-3)
parser.add_argument("--lr_pg",type=float, default=1e-4)
parser.add_argument("--lr_decay",type=float, default=0.99)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument("--save_every", type=float, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port",type=int, default=5555)
parser.add_argument("--ode_solver",type=str, default="dopri5")


parser.add_argument("--dyna_patience",type=int, default=10)
parser.add_argument("--pg_patience",type=int, default=10)
parser.add_argument("--pg_early_stop",type=str2bool, default='false')
parser.add_argument("--dyna_early_stop",type=str2bool, default='false')

parser.add_argument("--aug_fft",type=str2bool, default='false')
parser.add_argument("--aug_perm",type=str2bool, default='false')
####################################################


############### model parameters ###################
parser.add_argument("--hstate_dim",type=int, default=64)
parser.add_argument("--hidden_dims", type=str, default="64-64-64")

parser.add_argument("--controls_limit", type=int ,default = 8)
parser.add_argument("--closure_limit", type=int, default = 3)

parser.add_argument("--intervention_coeff", type=float, default=0.0)
parser.add_argument("--smooth_coeff", type=float, default=0.0)
parser.add_argument("--perminv_coeff", type=float, default=0.0)
#####################################################


############## dataset parameters ###################
parser.add_argument("--window_size", type=int, default=7)
parser.add_argument("--batch_size","-b",type=int, default=8)
parser.add_argument("--state", type=str, default="Kansas")
parser.add_argument("--split",type=int, default=0)
######################################################


############### meta parameters ######################
parser.add_argument("--niterations",type=int, default=100)
parser.add_argument("--innerepochs",type=int, default=30)
######################################################