DIR=$1
OUT_FILE=$2
#COUNTER=0

for file in `ls $DIR | sort -n -t _ -k 1`; do
    #echo $DIR/$file
    cat $DIR/$file >> $OUT_FILE
    #let COUNTER=COUNTER+`wc -l $DIR/$file`
done

#echo $COUNTER
