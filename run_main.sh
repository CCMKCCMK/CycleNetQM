# #!/bin/bash

# sh scripts/CycleNet/Linear-Input-96/electricity.sh;

# sh scripts/CycleNet/Linear-Input-336/electricity.sh;

# sh scripts/CycleNet/Linear-Input-720/electricity.sh;

# sh scripts/CycleNet/MLP-Input-96/electricity.sh;

# sh scripts/CycleNet/MLP-Input-336/electricity.sh;

# sh scripts/CycleNet/MLP-Input-720/electricity.sh;


model_names=("CycleNet" "Linear" "CycleNetMM")

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=electricity

model_types=("mlp" "linear")
seq_lens=(96 336 720)
pred_lens=(96)

for model_name in "${model_names[@]}"
do
    if [ "$model_name" == "CycleNet" ]; then
        for model_type in "${model_types[@]}"
        do
            for seq_len in "${seq_lens[@]}"
            do
                for pred_len in "${seq_lens[@]}"
                do
                    for random_seed in 1024
                    do
                        train_epochs=15
                        python -u run.py \
                            --is_training 1 \
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
    elif [ "$model_name" == "CycleNetMM" ]; then
        model_type="mlp"
        for seq_len in "${seq_lens[@]}"
        do
            for pred_len in "${seq_lens[@]}"
            do
                for random_seed in 1024
                do
                    train_epochs=15

                    python -u run.py \
                        --is_training 1 \
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
    elif [ "$model_name" == "Linear" ]; then
        for seq_len in "${seq_lens[@]}"
        do
            for pred_len in "${seq_lens[@]}"
            do
                for random_seed in 1024
                do
                    train_epochs=30
                    python -u run.py \
                        --is_training 1 \
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
                        --train_epochs $train_epochs \
                        --patience 5 \
                        --itr 1 --batch_size 64 --learning_rate 0.01 --random_seed $random_seed
                done
            done
        done
    fi
done