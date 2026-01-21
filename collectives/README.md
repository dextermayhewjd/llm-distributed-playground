# 这个folder干什么

本目录用于白盒理解分布式训练中的基础通信原语（collective communication）  
所有后续的 Data Parallel（DP）、Tensor Parallel（TP）以及 FSDP / ZeRO，本质上都由这些通信操作组合而成。  

## 限制是什么  

本目录 仅支持 GPU + NCCL
假设运行环境为单机多 GPU
所有示例均在 CUDA device 上运行

## 这个目录包含的通信原语  

| 文件                  | Collective     | 通信语义         | 典型用途         |
| ------------------- | -------------- | ------------ | ------------ |
| `broadcast.py`      | broadcast      | 1 → N        | 参数初始化        |
| `reduce.py`         | reduce         | N → 1        | 统计 / 汇总      |
| `all_reduce.py`     | all_reduce     | N → N（求和/平均） | DP 梯度同步      |
| `all_gather.py`     | all_gather     | N → N（拼接）    | TP 前向        |
| `reduce_scatter.py` | reduce_scatter | N → N（分片）    | TP 反向 / FSDP |
