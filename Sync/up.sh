
# Upload datasets and embds to other servers

declare -a hosts=('hufflepuff' 'slytherin' 'horcrux' 'crookshanks')
declare -a dirs=('datasets' 'Embs')


# Send to hosts
for host in "${hosts[@]}"
do  
    for dir in "${dirs[@]}"
    do
        echo "$host"
        echo "$dir"
        rsync -a --info=progress2 --include "*.pt" --exclude '*/' $dir/ fjogl@$host.ml.tuwien.ac.at:/home/fjogl/ExpressivenessTransferGNNs/$dir/ 
    done
done