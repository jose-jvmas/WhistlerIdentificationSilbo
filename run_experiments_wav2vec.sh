#!/bin/bash

# Whistler identification using Wav2Vec
for param in 64 256 1024 4096; do
    python -u src/main.py -e wav2vec -p $param > logs/wav2vec_$param.log
done
