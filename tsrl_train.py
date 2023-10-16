import os
import sys
sys.path.append(os.getcwd())
from turtle import done
import wandb
import argparse
from tsrl_algos import TSRL
import datetime

def main():
  wandb.init(project="TSRL", entity="")
  # Parameters
  parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with TD3_BC')
  parser.add_argument('--device', default='cuda', help='cuda or cpu')
  parser.add_argument('--env_name', default='hopper-medium-v2', help='choose your mujoco env')
  parser.add_argument('--ratio', default=1, type=float, help='choose the data ratio')
  parser.add_argument('--alpha', default=2.5, type=float)
  parser.add_argument('--gamma', default=0.99, type=float)
  parser.add_argument("--seed", default=111, type=int)  # Sets Gym, PyTorch and Numpy seeds
  parser.add_argument("--num_hidden", default=512, type=int) 
  parser.add_argument('--lr_actor', default=3e-4, type=float)
  parser.add_argument('--lr_critic', default=3e-4, type=float)
  parser.add_argument('--z_act_weight', default=0.0, type=float)
  parser.add_argument('--inconsis_weight', default=0.0, type=float)
  parser.add_argument('--quantile', default=0.0, type=float)
  parser.add_argument('--drop_prob', default=0.1, type=float)
  parser.add_argument('--eval_iter', default=5, type=int)
  parser.add_argument('--augment', default=True, type=bool)
  parser.add_argument('--store_prams', default=False, type=bool)
  args = parser.parse_args()
  wandb.config.update(args)

  # setup mujoco environment
  env_name = args.env_name
  wandb.run.name = f"TSRL-Alpha_{args.alpha}-{env_name}-seed_{args.seed}-ratio_{args.ratio}"

  agent_TD3_BC = TSRL(env_name,
          device=args.device,
          ratio=args.ratio,
          gamma=args.gamma,
          alpha=args.alpha,
          num_hidden=args.num_hidden,
          z_act_weight=args.z_act_weight,
          inconsis_weight = args.inconsis_weight,
          quantile=args.quantile,
          drop_prob=args.drop_prob,
          store_prams=args.store_prams,
          augment=args.augment,
          eval_iter=args.eval_iter,
          seed=args.seed,
          )
  agent_TD3_BC.tsrl_learn(total_time_step=int(1e+6))


if __name__ == '__main__':
  main()