# Knowledge Distillatoin for (Expressive) GNNs

Code repository for our paper [_"Is Expressivity Essential for the Predictive Performance of Graph Neural Networks?"_](https://openreview.net/pdf?id=5v7hpSy3Ir) (SciForDL @ NeurIPS 2024).

## Setup
Clone this repository and open the directory

Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

Create a conda environment (this assume miniconda is installed)
```
conda create --name ET && conda activate ET
```

Install dependencies
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch && conda install -c pyg pyg=2.2.0 && conda install -c conda-forge graph-tool=2.44 && python -m pip install -r requirements.txt

```


## Recreating our Experiments
Training teachers is not necessary, we provide trained models. If you want you can train teachers with `Scripts/train_basic_model.py` and evaluate models with `Scripts/eval_baseline_models.py`.

**ZINC**
```
bash Scripts/Bash/transfer_experiment.sh CWN ZINC
bash Scripts/Bash/transfer_experiment.sh DSS ZINC
bash Scripts/Bash/transfer_experiment.sh L2GNN ZINC
bash Scripts/Bash/transfer_experiment.sh GSN9 ZINC
```

**ZINC ablations**
```
python Exp/run_experiment.py -grid Configs/Ablations/DSS_to_GIN_no_layer_matching.yaml -dataset ZINC --x 100 --repeats 3
python Exp/run_experiment.py -grid Configs/Ablations/DSS_to_small_GIN.yaml -dataset ZINC --x 100 --repeats 3
python Exp/run_experiment.py -grid Configs/Ablations/GIN_big.yaml -dataset ZINC --x -1 --repeats 3
python Scripts/eval_baseline_models.py -model GIN -dataset ZINC --repeats 3

```


**MOLHIV**
```
python Exp/run_experiment.py -grid Configs/Knowledge_Distil/CWN_to_GIN_ogbg-molhiv.yaml -dataset ogbg-molhiv --x 0 --repeats 3```
python Exp/run_experiment.py -grid Configs/Knowledge_Distil/GSN9_to_GIN_ogbg-molhiv.yaml -dataset ogbg-molhiv --x 0 --repeats 3```
python Exp/run_experiment.py -grid Configs/Knowledge_Distil/L2GNN_to_GIN_ogbg-molhiv.yaml -dataset ogbg-molhiv --x 0 --repeats 3```
```

