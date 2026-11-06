#!/bin/bash

while true
do
    python main.py monitor --debug --llm-all-chunks "$@"
    sleep 1
done
