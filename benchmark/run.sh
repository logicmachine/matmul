#!/bin/bash

MATMUL=$1

run () {
	echo $1 $1 $1 | $MATMUL | grep '\[GFLOPS\]' | sed 's/ \[GFLOPS\]//'
}

echo "N,FLOPS1,FLOPS2,FLOPS3"

for n in `seq $2 $3 $4`; do
	flops1=`run $n`
	flops2=`run $n`
	flops3=`run $n`
	echo "$n,$flops1,$flops2,$flops3"
done
