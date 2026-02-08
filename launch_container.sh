gpu=$1
WANDB_API_KEY=$(cat ./dev/wandb)

container_name="mfax_$(echo $gpu | tr ',' '_')"

# --- multiple GPUs ---
if [[ $gpu == *","* ]]; then
    gpu_flag="--gpus all"
else
    gpu_flag="--gpus device=$gpu"
fi

docker run \
    $gpu_flag \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e CUDA_VISIBLE_DEVICES=$gpu \
    -v $(pwd):/home/duser/mfax \
    --name $container_name \
    --user $(id -u) \
    -it mfax bash