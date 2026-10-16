# 测试

`LD_LIBRARY_PATH='' .venv/bin/python -m pytest` 执行独立单元与合成控制测试，不读取外部数据；临时产物放在 `outputs/pytest/`。

覆盖 column-rot6d 的列约定与退化、非对易 SE(3) 积分、TCP 力臂、base-frame SO(3)、DROID Euler 差与向量限幅、独立夹爪、尾行 mask、provenance join、时序支持、PID 饱和/释放/滤波/状态隔离/重复反馈/陈旧观测/时间倒退、chunk 衔接、延迟辨识与因果 history，以及实际推进新状态的合成 FF/P/PI/PD/PID 控制。

外部数据测试默认 deselected，不算通过。显式执行：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
```

该测试重新抽取只读 episode，验证物理时钟、来源、masks、夹爪、几何与 R1 flange 交叉一致性。缺少配置、源文件或契约不符会失败。完整 P2/P3 实验由 `scripts/run_experiment.py --seed 42 --stage all --plots` 执行，结果不等同于单元测试或真机控制通过。
