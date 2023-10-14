import os
import sys
sys.path.append(os.getcwd())
sys.path.append('TDM/')
from turtle import done
import wandb
import argparse
from tsrl_algos import TSRL
import datetime

def main():
  wandb.init(project="TSRL-test", entity="pcheng")
  # Parameters
  parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
  parser.add_argument('--device', default='cuda', help='cuda or cpu')
  parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
  parser.add_argument('--ratio', default=1, type=float, help='choose your mujoco env')
  parser.add_argument('--alpha', default=2.5, type=float, help='choose your mujoco env')
  parser.add_argument('--gamma', default=0.99, type=float, help='choose your mujoco env')
  parser.add_argument("--seed", default=111, type=int)  # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--num_hidden", default=512, type=int) 
  parser.add_argument('--lr_actor', default=3e-4, type=float)
  parser.add_argument('--lr_critic', default=3e-4, type=float)
  parser.add_argument('--z_act_weight', default=0.0, type=float)
  parser.add_argument('--inconsis_weight', default=0.0, type=float)
  parser.add_argument('--quantile', default=0.0, type=float)
  parser.add_argument('--Drop_prob', default=0.1, type=float)
  parser.add_argument('--eval_iter', default=5, type=int)
  parser.add_argument('--augment', default=True, type=bool)
  parser.add_argument('--store_prams', default=False, type=bool)
  args = parser.parse_args()
  wandb.config.update(args)

  # setup mujoco environment
  env_name = args.env_name
  wandb.run.name = f"TSRL-Alpha_{args.alpha}-{env_name}-seed_{args.seed}-ratio_{args.ratio}-Baidu"

  agent_TD3_BC = TSRL(env_name,
          device=args.device,
          ratio=args.ratio,
          gamma=args.gamma,
          alpha=args.alpha,
          num_hidden=args.num_hidden,
          z_act_weight=args.z_act_weight,
          inconsis_weight = args.inconsis_weight,
          quantile=args.quantile,
          Drop_prob=args.Drop_prob,
          store_prams=args.store_prams,
          augment=args.augment,
          eval_iter=args.eval_iter,
          seed=args.seed,
          )
  agent_TD3_BC.tsrl_learn(total_time_step=int(1e+6))
  # agent_TD3_BC.online_exploration(exploration_step=int(1e+3))


if __name__ == '__main__':
  main()

# hopper-random-v2  hopper-medium-v2    hopper-medium-replay-v2    hopper-medium-expert-v2   hopper-expert-v2 
# halfcheetah-random-v2 halfcheetah-medium-v2 halfcheetah-medium-replay-v2 halfcheetah-medium-expert-v2  halfcheetah-expert-v2 
# walker2d-random-v2  walker2d-medium-v2  walker2d-medium-replay-v2  walker2d-medium-expert-v2   walker2d-expert-v2 
# pen-human-v0    pen-cloned-v0     pen-expert-v0 
# hammer-human-v0   hammer-cloned-v0    hammer-expert-v0 
# door-human-v0   door-cloned-v0    door-expert-v0 
# relocate-human-v0   relocate-cloned-v0  relocate-expert-v0 

# kitchen-complete-v0: 3680  s:60 a:9
# kitchen-partial-v0: 130k
# kitchen-mixed-v0: 130k

# LOCOMOTION            10k   50k
# expert/medium/random:   1M    100	  20	  10	
# medium-expert:      2M    200	  40	  20	
# halfcheetah-medium-replay:  200k    20	  4	  2
# walker2d-medium-replay:   300k    30	  6	  3
# hopper-medium-replay:   400k    40	  8	  4
# -----------------------------------------------------
# Adroit            10k  5k
# human-v0:       5k      
# cloned-v1:        500k    50   100
# expert-v1:        500k    50   100 
# kitchen-complete-v0:    3680 
# kitchen-partial-v0:     130k    10   2
# kitchen-mixed-v0:     130k    10   2

# export WANDB_API_KEY=ea1eb211c31bd14320ec5da1ed751ff43eaed121

# export CUDA_VISIBLE_DEVICES=1 

#------- Walker2d -------#
# *10k*:
# nohup python tsrl_train.py --ratio 100 --seed 333 --quantile 0.5 --env_name walker2d-medium-v2 --z_act_weight 200 --inconsis_weight 100  > logs/TSRL-Walker2d-medium-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 30 --seed 111 --quantile 0.5 --env_name walker2d-medium-replay-v2 --z_act_weight 100 --inconsis_weight 100  > logs/TSRL-Walker2d-medium-replay-v2.log 2>&1 &

# nohup python tsrl_train.py  --ratio 200  --seed 222 --quantile 0.5 --env_name walker2d-medium-expert-v2 --z_act_weight 200 --inconsis_weight 100 > logs/TSRL-Walker2d-medium-expert-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 100 --seed 111 --quantile 0.5 --env_name walker2d-expert-v2 --z_act_weight 200 --inconsis_weight 100  > logs/TSRL-Walker2d-random-v2.log 2>&1 &

# *Full*:
# nohup python tsrl_train.py --ratio 1 --seed 333  --env_name walker2d-medium-v2 --z_act_weight 10 --inconsis_weight 1  > logs/TSRL-Walker2d-random-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 1 --quantile 0.9 --seed 222  --env_name walker2d-medium-replay-v2 --z_act_weight 5 --inconsis_weight 100  > logs/TSRL-Walker2d-random-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 1 --quantile 0.9 --env_name walker2d-medium-expert-v2 --z_act_weight 5 --inconsis_weight 1  > logs/TSRL-Walker2d-random-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 1 --seed 222 --quantile 0.9  --env_name walker2d-expert-v2 --z_act_weight 200 --inconsis_weight 100  > logs/TSRL-Walker2d-random-v2.log 2>&1 &

#------- Halfcheetah -------#
# *10k*:
# nohup python tsrl_train.py --ratio 100 --quantile 0.7 --seed 111  --env_name halfcheetah-medium-v2 --z_act_weight 100 --inconsis_weight 100  > logs/TSRL-halfcheetah-random-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 20 --quantile 0.7 --seed 111 --env_name halfcheetah-medium-replay-v2 --z_act_weight 100 --inconsis_weight 100  > logs/TSRL-halfcheetah-random-v2.log 2>&1 &

# nohup python tsrl_train.py  --ratio 200  --quantile 0.7 --seed 111  --env_name halfcheetah-medium-expert-v2 --z_act_weight 100 --inconsis_weight 100 > logs/TSRL-halfcheetah-random-v2.log 2>&1 &

# nohup python tsrl_train.py --ratio 100 --quantile 0.7 --Drop_prob 0.1 --seed 111 --env_name halfcheetah-expert-v2 --z_act_weight 200 --inconsis_weight 10  > logs/TSRL-halfcheetah-random-v2.log 2>&1