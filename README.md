# Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline Reinforcement Learning (NeurIPS 2023)


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
