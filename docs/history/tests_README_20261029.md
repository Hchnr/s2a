# 测试

`LD_LIBRARY_PATH='' .venv/bin/python -m pytest` 执行独立单元与合成控制测试，不读取外部数据；临时产物放在 `outputs/pytest/`。

覆盖 column-rot6d 的列约定与退化、非对易 SE(3) 积分、TCP 力臂、base-frame SO(3)、DROID Euler 差与向量限幅、独立夹爪、尾行 mask、provenance join、时序支持、PID 饱和/释放/滤波/状态隔离/重复反馈/陈旧观测/时间倒退、chunk 衔接、延迟辨识与因果 history，以及实际推进新状态的合成 FF/P/PI/PD/PID 控制。

外部数据测试默认 deselected，不算通过。显式执行：

```bash
LD_LIBRARY_PATH='' .venv/bin/python -m pytest tests -m integration --data-config configs/droid_episode0.json
```

该测试重新抽取只读 episode，验证物理时钟、来源、masks、夹爪、几何与 R1 flange 交叉一致性。缺少配置、源文件或契约不符会失败。完整 P2/P3 实验由 `scripts/run_experiment.py --seed 42 --stage all --plots` 执行，结果不等同于单元测试或真机控制通过。

跨数据集新增测试覆盖 native wxyz 四元数、M1 effector→link8固定方向与TCP力臂、native source row非零起点/乱序join、invalid观测边界的独立积分，以及URDF轴/原点/fixed joint（允许无效或零axis）的名义FK。真实60条跨数据集检查由 `scripts/run_cross_dataset.py --seed 42` 执行，证据分类和报告独立于pytest软件通过数。

LIBERO扩展的5项OSC单元测试覆盖：非零初速度下固定PD响应的独立前向积分、历史affine inverse/clipping一致性、不可辨识短周期拒绝、最终发送命令变化率，以及大角度回退。独立MuJoCo的37项真实控制执行通过 `run_libero_matrix.py` 组织，`summarize_libero_simulation.py --seed 42` 复核状态误差/时钟/接口约束/成对输入和指纹，不计入pytest软件测试数。原生expert末行有效命令会执行，缺失的未来观测标签不进入pose/opening评分。

2026-10-19新增测试验证开度趋势在接触平台的保持、释放滞回、非法开度拒绝、position actuator映射、post-action参考首末端、Action128按mask反归一化与时间合同，以及actuator delay的因果顺序、copy隔离和初始目标保持。实际MuJoCo矩阵、发布数据核查与模型响应审计分别由实验脚本执行，未计入pytest通过数；终态保持结果与首次环境success分开。
