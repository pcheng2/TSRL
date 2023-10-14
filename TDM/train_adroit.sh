for ENV in 'pen' 'hammer' 'door' 'relocate'; do
for DATASET in 'cloned' 'expert' ; do
ENV_NAME=$ENV'-'$DATASET'-v1'
TRAIN_EPOCH=2000
RATIO=50
echo $RATIO $ ENV_NAME
python tdm_train.py --env_name $ENV_NAME --ratio $RATIO --epoch $TRAIN_EPOCH
done
done