#!/bin/bash

if [ ! -d logs ]; then
  mkdir logs
fi

# Whistler identification using Whisper
for param in tiny base small medium; do
    python -u src/main.py -e whisper -p $param > logs/whisper_$param.log
done
