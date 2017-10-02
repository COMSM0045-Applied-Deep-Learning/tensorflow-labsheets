#!/usr/bin/env bash

for notebook in *.ipynb; do
    echo "Running $notebook"
    jupyter nbconvert \
        --inplace \
        --execute \
        --allow-errors \
        "$notebook"
done
