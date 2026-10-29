# 完整轨迹搜索入口与独立审核

`probe_tracking_v5_*`为生产入口，`check_tracking_v5_*`为独立完整响应复算，
`*_audit.py`为独立账本及数学检查。可复用算法在
`src/s2a/research/trajectory_search/`，不再从脚本目录混同导入。

当前进行迁移准入的是最新dual-recovery两入口。所有命令从仓库根执行：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONPATH=src:scripts \
outputs/libero_env311/bin/python -m scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery --dry

CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONPATH=src:scripts \
outputs/libero_env311/bin/python -m scripts.tracking.trajectory_search.check_tracking_v5_state_vector_dual_recovery --dry
```

dry只读取与核验输入，不运行新的QP、模型响应或物理控制。真实生产需要另有
新登记和独占输出；已有完整及失败输出不能覆盖。

旧家族源码与独立数学实现保留，历史入口的执行资格分别记录，不能因路径迁移
推断全部旧CLI已获新准入。11个直接依赖旧脚本暂保留原bare身份和位置，由集中
legacy边界核来源及双载；其完整依赖迁移后再移除此边界。

[全局设计](../../../docs/architecture/repository_design.md) ·
[迁移身份协议](../../../docs/migrations/README.md) ·
[当前状态](../../../docs/status.md)
