config=$1
bash install.sh
if [ -d outputs/models/debug ]; then
    rm -r outputs/models/debug
fi
if [ -d outputs/logs/debug ]; then
    rm -r outputs/logs/debug
fi

cp $config configs/debug.yaml
exec_train_and_cv configs/debug.yaml --num_partial_folds 1
