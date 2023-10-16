# Look Beneath the Surface: Exploiting Fundamental Symmetry for Sample-Efficient Offline Reinforcement Learning (NeurIPS 2023)

#### Usage
To install the dependencies, use 
```python
    pip install -r requirements.txt
```

#### 1.Create small samples
Before trainig, you should set up a small sample dataset by yourself:
```python
    python utils/generate_samples.py 
```

#### 2.Train TDM models
Before running TSRL, you should set up a small sample dataset by:
```python
    python utils/generate_samples.py 
```

#### 3.Benchmark experiments
You can run Mujoco tasks and AntMaze tasks like so:
```python
    python train_distance_mujoco.py --env_name halfcheetah-medium-v2 --alpha 7.5
```
```python
    python train_distance_antmaze.py --env_name antmaze-umaze-v2 --alpha 5.0
```
#### Modified AntMaze tasks

You can run the modified AntMaze medium/large tasks like so:
```python
    python train_distance_antmaze.py --env_name antmaze-large-play-v2 --alpha 70 --toycase True
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
