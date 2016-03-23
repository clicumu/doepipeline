#!/usr/bin/env bash

while [[ $# > 1 ]]
do
key="$1"

case ${key} in
    -o|--output)
    OUTPUT="$2"
    shift
    ;;
    --factor_a)
    FACTOR_A="$2"
    shift
    ;;
    --factor_b)
    FACTOR_B="$2"
    ;;
    *)

    ;;
esac
shift
done

echo "Factor A: $FACTOR_A, Factor B: $FACTOR_B" > ${OUTPUT}/output.txt