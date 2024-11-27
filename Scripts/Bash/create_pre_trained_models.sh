rm Models/trained_models/*
python Scripts/train_basic_model.py -model GIN -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model CWN -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model DSS -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN4 -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN5 -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN6 -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN7 -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN8 -dataset ZINC --store_model 1
python Scripts/train_basic_model.py -model GSN9 -dataset ZINC --store_model 1


python Scripts/train_basic_model.py -model GIN -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model CWN -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN4 -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN5 -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN6 -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN7 -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN8 -dataset ogbg-molhiv --store_model 1
python Scripts/train_basic_model.py -model GSN9 -dataset ogbg-molhiv --store_model 1