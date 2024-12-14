# #!/bin/bash

model_names=("CycleNet" "CycleNetQM" "CycleNetMM" "CycleNetQQ" "GRU" "Linear" "LSTM")

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=electricity

model_types=("mlp" "linear")
seq_lens=(336 720)
pred_lens=(336 720)
seq_lens=(96 336 720)
pred_lens=(96 336 720)
train_epochs=30
is_training=0

for model_name in "${model_names[@]}"
do
    if [ "$model_name" == "CycleNet" ] || [ "$model_name" == "CycleNetQQ" ] || [ "$model_name" == "CycleNetQM" ]; then
        model_types=("mlp" "linear")
    elif [ "$model_name" == "Linear" ]; then
        model_types=("linear")
    elif [ "$model_name" == "CycleNetMM" ]; then
        model_types=("mlp")
    elif [ "$model_name" == "LSTM" ] || [ "$model_name" == "GRU" ]; then
        model_types=("rnn")
    fi
    for model_type in "${model_types[@]}"
    do
        for seq_len in "${seq_lens[@]}"
        do
            for pred_len in "${pred_lens[@]}"
            do
                for random_seed in 1024
                do
                    train_epochs=30

                    python -u run.py \
                        --is_training $is_training \
                        --root_path $root_path_name \
                        --data_path $data_path_name \
                        --model_id ${model_id_name}_${seq_len}_${pred_len} \
                        --model $model_name \
                        --data $data_name \
                        --features M \
                        --seq_len $seq_len \
                        --pred_len $pred_len \
                        --enc_in 321 \
                        --cycle 168 \
                        --model_type $model_type \
                        --train_epochs $train_epochs \
                        --patience 5 \
                        --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed
                done
            done
        done
    done
done
