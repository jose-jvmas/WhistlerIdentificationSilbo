#!/bin/bash

# Whistler identification using mfcc
for param in mfcc_mean mfcc_delta mfcc_deltadelta; do
    python -u src/main.py -e $param -p 20  > logs/$param.log
done
