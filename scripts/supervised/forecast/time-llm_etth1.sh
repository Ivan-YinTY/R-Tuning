export CUDA_VISIBLE_DEVICES=0
model_name=time_llm
token_num=7
token_len=16
seq_len=$[$token_num*$token_len]

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
  --output_token_len 96 \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.02 \
  --train_epochs 10 \
  --d_model 32 \
  --d_ff 128 \
  --lradj type1 \
  --dropout 0.2 \
  --use_norm \
  --valid_last \
  --llm_model GPT2 \
  --nonautoregressive