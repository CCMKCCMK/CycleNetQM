#model_name=DoubleCycleNet
#
#root_path_name=./dataset/
#data_path_name=electricity.csv
#model_id_name=Electricity
#data_name=electricity
#
#model_type='linear'
#seq_len=96
#
#for pred_len in 192
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
#      --cycle 168 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed
#done
#done

model_name=DoubleCycleNet

root_path_name=./dataset/
residual_csv_path_name=trans_remain.csv
cycle_csv_path_name=Q.csv
model_id_name=ResidualCycleNet
data_name=residual_cycle

model_type='linear'
seq_len=96

for pred_len in 192
do
for random_seed in 1024
do
    python -u run_double.py \
      --is_training 1 \
      --root_path $root_path_name \
      --residual_csv_path $residual_csv_path_name \
      --cycle_csv_path $cycle_csv_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 320 \
      --cycle 168 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed
done
done