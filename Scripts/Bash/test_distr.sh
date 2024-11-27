declare -a sizes=(1 5 10 25 50)
declare -a generators=("m" "er" "ct")

for size in "${sizes[@]}"
do
    for generator in "${generators[@]}"
    do
        echo "$size"
        echo "$generator"
        python Exp/run_experiment.py -grid Configs/Knowledge_Distil/Distribution_Testing/CWN_to_GIN_ZINC_$generator.yaml -dataset ZINC --repeats 2 --x "$size"
    done
done