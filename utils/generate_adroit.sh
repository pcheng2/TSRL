for ENV in 'pen' 'hammer' 'door' 'relocate'; do
for DATASET in 'cloned' 'expert' ; do
ENV_NAME=$ENV'-'$DATASET'-v1'
RATIO=50
echo $RATIO $ENV_NAME
python generate_samples.py --env_name $ENV_NAME --ratio $RATIO
done
done