config_path=$1
num_folds=$2

tsp exec_train_and_cv $config_path --num_partial_folds $num_folds