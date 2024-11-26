#model_name='LSTM'
#
#root_path_name=./dataset/
#data_path_name=electricity.csv
#model_id_name=Electricity
#data_name=electricity
#
#seq_len=96
#for pred_len in 96 192 336 720
#do
#for random_seed in 1024
#do
#    python -u run.py \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id $model_id_name'_'$seq_len'_'$pred_len \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 320 \
#      --d_model 512 \
#      --e_layers 2 \
#      --dropout 0.1 \
#      --train_epochs 30 \
#      --patience 5 \
#      --itr 1 \
#      --batch_size 32 \
#      --learning_rate 0.001 \
#      --random_seed $random_seed
#done
#done

#!/bin/bash

## 设置模型名称和数据路径等参数
#model_name="LSTM"  # 使用 LSTM 模型
#root_path_name="./dataset/"
#data_path_name="electricity.csv"
#model_id_name="Electricity"
#data_name="electricity"
#model_type="lstm"  # 指定模型类型为 lstm
#seq_len=96
#
## 循环不同的预测长度（pred_len）
#for pred_len in 96 192 336 720
#do
#  # 循环不同的随机种子（random_seed）
#  for random_seed in 1024
#  do
#    echo "Running experiment with LSTM, pred_len=${pred_len}, random_seed=${random_seed}"
#
#    # 执行 Python 脚本
#    python -u run.py \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id ${model_id_name}_${seq_len}_${pred_len} \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 320 \
#      --cycle 168 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --itr 1 \
#      --batch_size 64 \
#      --learning_rate 0.01 \
#      --random_seed $random_seed
#  done
#done

#!/bin/bash

## 设置模型名称
#model_name=LSTM
#
## 数据集路径
#root_path_name=./dataset/
#data_path_name=electricity.csv
#
## 实验配置
#model_id_name=Electricity
#data_name=custom
#model_type='lstm'  # 使用 LSTM 作为模型类型
#seq_len=720  # 输入序列长度
#
## 循环尝试不同的预测序列长度
#for pred_len in 96 192 336 720
#do
#  # 固定随机种子
#  for random_seed in 1024
#  do
#    python -u run.py \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id $model_id_name'_'$seq_len'_'$pred_len \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 320 \
#      --cycle 168 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --itr 1 \
#      --batch_size 64 \
#      --learning_rate 0.001 \
#      --random_seed $random_seed
#  done
#done

#!/bin/bash

# 设置模型名称
model_name="LSTM"

# 数据集路径
root_path_name="./dataset/"
data_path_name="electricity.csv"

# 实验配置
model_id_name="Electricity"
data_name="custom"
model_type="lstm"  # 使用 LSTM 作为模型类型
seq_len=720  # 输入序列长度

# 循环尝试不同的预测序列长度
for pred_len in 96 192 336 720
do
  # 固定随机种子
  for random_seed in 1024
  do
    python -u run.py \
      --is_training 1 \
      --root_path "${root_path_name}" \
      --data_path "${data_path_name}" \
      --model_id "${model_id_name}_${seq_len}_${pred_len}" \
      --model "${model_name}" \
      --data "${data_name}" \
      --features M \
      --seq_len "${seq_len}" \
      --pred_len "${pred_len}" \
      --enc_in 320 \
      --cycle 168 \
      --model_type "${model_type}" \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 \
      --batch_size 64 \
      --learning_rate 0.001 \
      --random_seed "${random_seed}"
  done
done