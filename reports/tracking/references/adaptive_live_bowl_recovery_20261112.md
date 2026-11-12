# 连续重规划：独立恢复试验与重放

2026-11-12。按[预登记](../../../plans/v0.0.1/tracking/references/adaptive_live_bowl_recovery_20261112.md)
在已有开发初态 `task0_init0_s42_raw`、seed 42 上重建 107 个控制周期；在完成的
第 107 和 111 周期各使用实际当前图像请求一次 WAM，并在两个参考窗口中各执行
4 个自适应周期。首个响应的状态、图像与采样身份及 action 数组与既有同图结果
逐项相同；校验逻辑见 [adaptive_live_bowl.py:20](../../../scripts/simulation/runtime/adaptive_live_bowl.py#L20)。
在第 115 周期达到登记的周期上限后停止，没有派发下一动作。

## 生产者与独立重放

| 指标 | 生产者与独立 checker |
| --- | --- |
| 已完成控制周期 / 内步 | 各 115 / 2,875，其中 107 为重建前缀、8 为自适应周期 |
| TCP 位置 / SO(3) 旋转 RMSE | 0.0030543726506808196 m / 0.017726035030739988 rad，按全部 115 周期计算 |
| 位姿端点约束 | 115/115 满足；最小实际内步关节余量 0.3871836971502849 rad |
| 原生任务成功 | 共 14/115 周期；端点 102–115 连续 14 个为真，其中新增窗口 8/8 |
| 目标 grasp | 共 40/115 周期；新增自适应窗口 1/8 |
| 独立逐值比较 | command、state、TCP error、inner q/qvel/time 和 gripper 最大绝对差均为 0，成功与 grasp 标签一致 |

checker 从同一 reset 只重放生产者已派发动作，重新采集图像与状态，没有重新选择
命令、加载候选策略或运行世界模型；它没有独立重新求解生产者的停止原因。
生产者与 checker 各创建 3 个仿真模型及 3 个 context，所有 worker 均已实际
wait/reap；模型使用登记 UUID，结束后该 GPU 显存及计算进程均已释放。

## 实际资源和验证

- 本次新增物理工作为生产者与 checker 合计 230 控制周期、5,750 内步、6 个
  模型及 6 个 context。WAM 初始化 1 次，2/2 请求完成、70/70 次去噪完成。
- 先前失败没有完整仿真收据，累计范围仍为 1,761–1,876 控制周期、
  44,025–46,900 内步及各 84–87 个模型/context，另有 2 个静态模型；累计 WAM
  初始化 10 次、请求尝试/完成 23/20 次、完成去噪 700 次。
- 真实预检核验 1,004 个模型源码文件、163 个权重文件及空闲 GPU；最终 64 并发
  相关 CPU 回归 283 项通过，无跳过和收集错误，源码前后一致，全部 worker 已回收。
  Ruff lint 和格式检查以及零模型、零物理步 dry 通过。

## 收据指纹

以下路径均在仓库的 `outputs/` 下，SHA-256 对应实际结果字节：

| 文件 | SHA-256 |
| --- | --- |
| `adaptive_live_bowl_recovery_20261112/preflight.json` | `fcc6381c501c02a444004dcdd3e504e1e32e98885d0bbce63fcc3b9de03d7b6b` |
| `adaptive_live_bowl_recovery_20261112/producer/result.json` | `b2c1df055bd76fb8cdd5c0f948f8f26ae3074b9ffefcb3075a2860c37eeb57be` |
| `adaptive_live_bowl_recovery_20261112/producer/simulation/result.json` | `942fe94ce05ac50f76f89e9b8e7f0d6927dbff36cc144273a0bf93b76dfe277b` |
| `adaptive_live_bowl_recovery_20261112/producer/model/result.json` | `4398b78163d4f5a27a063eb2c7edaa016e1cdc883f8c2189ef976ecae584cabb` |
| `adaptive_live_bowl_recovery_20261112/checker/result.json` | `c546979a78877233fc60aa591eee4d08d6710862de31581ed001b50b97f2aadc` |
| `adaptive_live_bowl_recovery_20261112/checker/simulation/result.json` | `8a29e12d48678434aabea74098ef521a60c44fdcd182a2effed04e088cff4187` |
| `qa/adaptive_live_bowl_recovery_final_20261112/result.json` | `b62089ce92fcea5b1a4082e028cca383080545719e24a76e1f87627571378dcb` |
| `adaptive_live_bowl_recovery_dry_20261112/result.json` | `02b5165b1ff4d52e925cf358350e0e44b05524645e82b054183d14c4fb6585d1` |

这是单个旧开发初态的有界短窗，等待模型时仿真时钟暂停。末尾连续 14 周期尚不足
原目标的末尾 40 周期；24 条留出新初态、600 周期评估以及实时/真机验证仍未执行，
不能把本结果当作最终目标达成或控制资格。
