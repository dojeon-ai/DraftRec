
####################
## 110k-benchmark ##
####################
# Interaction Models
python main.py --exp_name 270k_benchmark --model_name pop --op train_interaction --data_type full --model_type pop --epochs 1
python main.py --exp_name 270k_benchmark --model_name nmf_imp --op train_interaction --data_type full --model_type nmf --target_type implicit --epochs 100
python main.py --exp_name 270k_benchmark --model_name nmf_exp --op train_interaction --data_type full --model_type nmf --target_type explicit --epochs 100

# Sequential Models
python main.py --exp_name 270k_benchmark --model_name spop --op train_user_rec --data_type full --model_type spop --epochs 1
python main.py --exp_name 270k_benchmark --model_name sas --op train_user_rec --data_type full --model_type sas --lmbda 0.5 --use_game_specific_info False --epochs 500 --evaluate_every 10
python main.py --exp_name 270k_benchmark --model_name sas_plus --op train_user_rec --data_type full --model_type sas --lmbda 0.5 --epochs 500 --evaluate_every 10

# Draftrec Models
python main.py --exp_name 270k_benchmark --model_name draftrec_no_history --op train_draftrec_rec --data_type full --model_type draftrec --lmbda 0.5 --evaluate_every 1
python main.py --exp_name 270k_benchmark --model_name draftrec --op train_draftrec_rec --data_type full --model_type draftrec --lmbda 0.5 --evaluate_every 1
python main.py --exp_name 270k_benchmark --model_name draftrec --op train_draftrec_rec --data_type full --model_type draftrec --lmbda 0.5 --evaluate_every 1
python main.py --exp_name 270k_benchmark --model_name draftrec --op train_draftrec_rec --data_type full --model_type draftrec --lmbda 0.5 --evaluate_every 1