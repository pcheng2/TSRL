for ENV in 'pen' 'hammer' 'door' 'relocate'; do
for DATASET in 'human' 'cloned' 'expert' ; do
for SEED in 111 222 333; do
ENV_NAME=$ENV'-'$DATASET'-v1'
W_INCONSIS=100
W_Z_ACT=10000
QUANTILE=0.7
RATIO=50
if [[ "$ENV_NAME" =~ "human" ]]; then
RATIO=1
fi
echo $RATIO $ ENV_NAME
python tsrl_train.py --env_name $ENV_NAME --ratio $RATIO --z_act_weight $W_Z_ACT --inconsis_weight $W_INCONSIS --quantile $QUANTILE --seed $SEED
done
done
done