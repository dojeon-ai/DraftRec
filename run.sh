
##################
## 1M-benchmark ##
##################





###################
## 70k-benchmark ##
###################
# Interaction Models
python main.py --exp_name 70k_benchmark --model_name pop --op train_interaction --data_type toy --model_type pop --epochs 1
python main.py --exp_name 70k_benchmark --model_name nmf_imp --op train_interaction --data_type toy --model_type nmf --target_type implicit --epochs 100
python main.py --exp_name 70k_benchmark --model_name nmf_exp --op train_interaction --data_type toy --model_type nmf --target_type explicit --epochs 100

# Sequential Models
python main.py --exp_name 70k_benchmark --model_name spop --op train_user_rec --data_type toy --model_type spop --epochs 1
python main.py --exp_name 70k_benchmark --model_name sas --op train_user_rec --data_type toy --model_type sas --lmbda 0.5 --use_game_specific_info False --epochs 1000
python main.py --exp_name 70k_benchmark --model_name bert --op train_user_rec --data_type toy --model_type bert --lmbda 0.5 --use_game_specific_info False --epochs 1000
python main.py --exp_name 70k_benchmark --model_name sas_plus --op train_user_rec --data_type toy --model_type sas --lmbda 0.5 --epochs 1000
python main.py --exp_name 70k_benchmark --model_name bert_plus --op train_user_rec --data_type toy --model_type bert --lmbda 0.5 --epochs 1000

# DraftRec
python main.py --exp_name 70k_benchmark --model_name draftrec --op train_draft_rec --data_type toy --lmbda 0.5 --epochs 100
python main.py --exp_name 70k_benchmark --model_name draftrec_history --op train_draft_rec --data_type toy --lmbda 0.5 --epochs 100







#################
## Max_seq_len ##
#################




######################
## Model Complexity ##
######################