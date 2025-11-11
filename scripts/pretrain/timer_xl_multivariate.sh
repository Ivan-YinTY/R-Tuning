export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
model_name=timer_xl
token_num=32
token_len=96
seq_len=$[$token_num*$token_len]

# starting multivariate pre-training only when when the datasest is structurally unified
python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./path/to/dataset/ \
  --data_path dataset.csv \
  --model_id multivariate_pretrain \
  --model $model_name \
  --data MultivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 4 \
  --d_model 512 \
  --d_ff 2048 \
  --batch_size 4096 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --dp \
  --devices 0,1,2,3,4,5,6,7