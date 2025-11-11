# export CUDA_VISIBLE_DEVICES=3
model_name=timer_xl
token_num=30
token_len=96
seq_len=$[$token_num*$token_len]

export LD_LIBRARY_PATH=/Extra/env_temp/openltm/lib:$LD_LIBRARY_PATH

nohup torchrun --nproc-per-node=2 run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --origin_root_path ./dataset/ETT-small/ \
  --origin_data_path ETTh1.csv \
  --use_replay \
  --use_synthetic \
  --model_id ETTh1_few_shot \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 1024 \
  --learning_rate 5e-6 \
  --train_epochs 10 \
  --gpu 0 \
  --ddp \
  --cosine \
  --tmax 10 \
  --use_norm \
  --visualize \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \
  --subset_rand_ratio 0.5 \
  > logs/few_shot_${model_name}_etth1.log 2>&1 &