# Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline Reinforcement Learning (NeurIPS 2023)

TSRL (https://arxiv.org/abs/2306.04220) introduces a new offline reinforcement learning (RL) algorithm that leverages the fundamental symmetry of system dynamics to enhance performance under small datasets. The proposed Time-reversal symmetry (T-symmetry) enforced Dynamics Model (TDM) establishes consistency between forward and reverse latent dynamics, providing well-behaved representations for small datasets. TSRL achieves impressive performance on small benchmark datasets with as few as 1% of the original samples, outperforming recent offline RL algorithms in terms of data efficiency and generalizability



#### Usage
To install the dependencies, use 
```python
    pip install -r requirements.txt
```

#### 1.Create small samples
Before start trainig, you should create small samples by yourself:

```python
    bash utils/generate_loco.sh # For the locomotion tasks
```
and
```python
    bash utils/generate_adroit.sh # For the adroit tasks
```
#### 2.Train TDM models
You can train TDM simply from:

```python
    bash TDM/train_loco.sh # For the locomotion tasks 
```
and
```python
    bash TDM/train_adroit.sh #  For the adroit tasks
```

#### 3.Run TSRL on Benchmark experiments
After you have your own small samples as well as a trained TDM model, you can start run TSRL on D4RL tasks by:

```python
    bash tsrl_loco.sh # For the locomotion tasks 
```
and
```python
    bash tsrl_adroit.sh # For the locomotion tasks 
```

#### Visulization of Learning curves
You can resort to [wandb](https://wandb.ai/site) to login your personal account via export your own wandb api key.
```
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```
and run 
```
wandb online
```
to turn on the online syncronization.


