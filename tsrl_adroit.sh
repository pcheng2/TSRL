for ENV in 'walker2d' 'halfcheetah' 'hopper'; do
for DATASET in 'medium' 'medium-replay' 'medium-expert' 'expert'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
W_INCONSIS=100
W_Z_ACT=200
QUANTILE=0.7
RATIO=100
if [[ "$ENV_NAME" =~ "hopper-medium-replay" ]]; then
RATIO=40
fi

if [[ "$ENV_NAME" =~ "walker2d-medium-replay" ]]; then
RATIO=30
fi

if [[ "$ENV_NAME" =~ "halfcheetah-medium-replay" ]]; then
RATIO=20
fi

if [[ "$ENV_NAME" =~ "medium-expert" ]]; then
RATIO=200
fi

if [[ "$ENV_NAME" =~ "walker2d" ]]; then
QUANTILE=0.5
fi

echo $ENV_NAME $RATIO

python tsrl_train.py --env_name $ENV_NAME --ratio $RATIO --z_act_weight $W_Z_ACT --inconsis_weight $W_INCONSIS --quantile $QUANTILE
done
done