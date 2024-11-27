# $1 host from which to pull data

# Send to hosts
for dir in "${dirs[@]}"
do
    echo "$host"
    echo "$dir"
    rsync -a --info=progress2 --include "*.pt" --exclude '*/' fjogl@$1.ml.tuwien.ac.at:/home/fjogl/ExpressivenessTransferGNNs/$dir/ $dir/ 
done