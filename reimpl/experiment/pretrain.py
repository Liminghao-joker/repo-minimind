"""Pretrain"""
# 1. 手写一次 mini forward + loss 计算，参考配置：batch_size = 4, seq_len = 16
# 2. 简单做一次 backward 与 optimizer step，并观察参数梯度