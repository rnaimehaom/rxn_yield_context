# Train the first model (multi-task multi-label classfication).
python -u Multitask_train_LCY_morgan.py --activation ReLU --epochs 100 --dropout 0.5 \
    --train_path ~/rxn_yield_context/rxn_yield_context/All_LCC_Data/processed_data_12 \
    --batch_size 512 --weight_decay 0.0001 --fpsize 4096 --radius 2 \
    --init_lr 0.00001 --max_lr 0.01 --final_lr 0.0001 --warmup_epochs 4.0 \
    --save_dir ~/rxn_yield_context/rxn_yield_context/save_model/first_model_12_final_3 \
    --num_last_layer 1 --num_shared_layer 1 \
    --loss Focal --alpha 0.5 --gamma 2 \

# Train the second model (multi-task, temperature prediction + ranking).
python -u Multitask_train_LCY_morgan.py --activation ReLU --epochs 100 --dropout 0.5 \
    --train_path ~/rxn_yield_context/rxn_yield_context/All_LCC_Data/processed_data_12 \
    --batch_size 512 --weight_decay 0.0001 --fpsize 4096 --radius 2 \
    --init_lr 0.00001 --max_lr 0.01 --final_lr 0.0001 --warmup_epochs 4.0 \
    --save_dir ~/rxn_yield_context/rxn_yield_context/save_model/first_model_12_final_3 \
    --num_last_layer 1 --num_shared_layer 1 \
    --loss Focal --alpha 0.5 --gamma 2 \
