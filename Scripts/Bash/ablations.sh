python Exp/run_experiment.py -grid Configs/Ablations/DSS_to_small_GIN.yaml -dataset ZINC --repeats 4 --x 100
python Exp/run_experiment.py -grid Configs/Ablations/GIN_big.yaml -dataset ZINC --repeats 4 
python Exp/run_experiment.py -grid Configs/Ablations/DSS_to_GIN_no_layer_matching.yaml -dataset ZINC --repeats 4 --x 100
