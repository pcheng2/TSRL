for ENV in 'walker2d' 'halfcheetah' 'hopper'; do
for DATASET in 'medium' 'medium-replay' 'medium-expert' 'expert'; do
ENV_NAME=$ENV'-'$DATASET'-v2'
PRE_TRAIN=200
TRAIN_EPOCH=2000
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

echo $ENV_NAME $RATIO

python tdm_train.py --env_name $ENV_NAME --ratio $RATIO --pre_train_epoch $PRE_TRAIN --epoch $TRAIN_EPOCH
done
done