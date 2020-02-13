for F in `ls -d */`; do
    for G in `ls $F`; do
        A=`ls $F$G/checkpoints | sort -V | head -n -3`
        cd $F$G/checkpoints
        rm $A
        cd -
    done
done
