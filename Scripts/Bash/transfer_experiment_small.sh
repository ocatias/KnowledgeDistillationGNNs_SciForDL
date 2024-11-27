#  Use with bash Scripts/Bash/transfer_experiment.sh teacher dataset
#  e.g.:

# $1 ... teacher
# $2 ... dataset
declare -a sizes=(0 1 2 3)

for size in "${sizes[@]}"
do
    echo "$ds"
    python Exp/run_experiment.py -grid Configs/Knowledge_Distil/$1_to_GIN_$2.yaml -dataset "$2" --repeats 5 --x "$size"
   done