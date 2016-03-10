#!/usr/bin/env bash

while [[ $# -gt 1 ]]
do
key="$1"

case ${key} in
    --expressionWithFactors)
    FACTOR="$2"
    shift
    ;;
    *)

    ;;
esac
shift
done

echo "Factor two: $FACTOR" > "step2.txt"