#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1 -A comsm0018 --reservation=comsm0018-lab1  -t 0-02:00 --mem=4G --pty bash
