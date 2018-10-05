module add libs/tensorflow/1.2
srun -p gpu_veryshort --gres=gpu:1 -A comsm0018 --reservation=comsm0018-lab1 -t 0-00:15 --mem=4G  python testing_tensorflow.py