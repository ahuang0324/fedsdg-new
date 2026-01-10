#!/bin/bash
# TensorBoard 启动脚本
# 显示 fedavg, fedlora, fedsdg 三种算法的训练日志

cd "$(dirname "$0")/../.."

tensorboard --logdir_spec=fedavg:./logs/fedavg,fedlora:./logs/fedlora,fedsdg:./logs/fedsdg 
