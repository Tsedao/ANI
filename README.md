# Amortized Network Intervention to Steer the Excitatory Point Process
## Prerequisite
`torch`, `torchdiffeq`
## Usage
### Dataset preparation
First store the historical data under `synthetic_data` folder, we need first train a dynamic model for different regions, then moves to policy learning, i.e.,
```bash
├── synthetic_data
│   ├── YOUR_DATASET_NAME_1
│   │   └── train_count.npy
│   │   └── env_params.json
│   ├── YOUR_DATASET_NAME_2
│   │   └── train_count.npy
│   │   └── env_params.json
```

### Train dynamic model using neural ODE
```py
python train_model.py \
    --data_path = YOUR_DATASET_NAME \ # your dataset under synthetic_data folder
    --learning_influence_matrix  # learning influence matrix if ground truth adjacency matrix is not given in env_params.json
```
The above code will save the dynamic model under `synthetic_data_model` folder
### Train policy on a single region 
```py
python train_rl.py \
    --data_path = YOUR_DATASET_NAME
```
The above code will save the dynamic model under `synthetic_policy_model` folder
### Train meta-policy on multiple regions using constrastive learning loss
```py
python train_meta_rl.py \
    --regions = YOUR_DATASET_NAME_1 YOUR_DATASET_NAME_2 YOUR_DATASET_NAME_3 \
    --model_learning ## adaptive learning dynamic model
```
The above code will save the dynamic model under `synthetic_policy_model` folder