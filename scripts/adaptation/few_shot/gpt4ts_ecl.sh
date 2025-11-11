model_name=gpt4ts
token_len=24
seq_len=96
pred_len=24
temperature=3
distillation_lambda=0.2
synthetic_size=2000
# a smaller batch size chosen due to large memory usage

export LD_LIBRARY_PATH=/Extra/env_temp/openltm/lib:$LD_LIBRARY_PATH

nohup torchrun --nproc-per-node=2 run.py \
    --task_name forecast \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --is_training 1 \
    --origin_root_path ./dataset/ETT-small/ \
    --origin_data_path ETTh1.csv \
    --model_id ECL \
    --model gpt4ts \
    --data MultivariateDatasetBenchmark  \
    --use_lwf \
    --use_synthetic \
    --seq_len $seq_len \
    --input_token_len $token_len \
    --output_token_len $token_len \
    --test_seq_len $seq_len \
    --test_pred_len $pred_len \
    --d_model 768 \
    --d_ff 768 \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --cosine \
    --train_epochs 10 \
    --use_norm \
    --gpu 0 \
    --ddp \
    --gpt_layers 6 \
    --patch_size 16 \
    --stride 8 \
    --tmax 10 \
    --valid_last \
    --nonautoregressive \
    --visualize \
    --subset_rand_ratio 0.1 \
    > logs/train_gpt4ts_ecl.log 2>&1 &