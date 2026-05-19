if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env not found. STOP"
    exit 1
fi


CUDA_VISIBLE_DEVICES=4,5 accelerate launch --debug --config_file ./configs/accelerate2.yaml unwrap.py
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --debug --config_file ./configs/accelerate2.yaml train.py