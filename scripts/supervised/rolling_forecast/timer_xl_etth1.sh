export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
token_num=7
token_len=96
seq_len=$[$token_num*$token_len]
# training one model with a context length
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len $test_pred_len \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --d_model 1024 \
  --d_ff 2048 \
  --gpu 0 \
  --lradj type1 \
  --use_norm \
  --e_layers 1 \
  --valid_last \
  --test_dir forecast_ETTh1_timer_xl_ETTh1_Multi_sl672_it96_ot96_lr0.0001_bt32_wd0_el1_dm1024_dff2048_nh8_cosFalse_test_0
done
