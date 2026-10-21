# 两条历史episode的部分命令内容身份（2026-10-21）

R1 episode384/720的native translation XYZ与gripper四列，分别与cream-cheese官方HDF的demo27/24唯一连续片段逐值一致。旋转仍不通过原完整projection检验，因此这里只提供用于**保守排除来源**的部分内容证据，完整native control、原始初态及图像/pose物理同步身份仍未证明。

| R1 / R2 episode | 唯一HDF demo | R1范围（左闭右开） | HDF范围（左闭右开） | 四列最大绝对差 | 原旋转6列最大残差 | 旋转超阈值行 | 其它匹配窗口 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 384 / 850 | cream `demo_27` | `[0,112)` | `[0,112)`（HDF共113行） | `2.895083e-8` | `0.007499593806` | 3 | 0 |
| 720 / 1176 | cream `demo_24` | `[0,103)` | `[0,103)`（HDF共104行） | `2.895083e-8` | `0.007499929518` | 1 | 0 |

四列比较为`R1.action.native_control[0,1,2,9]`与`HDF.actions[0,1,2]`、`(1-HDF.actions[6])/2`。严格阈值保持`max_abs < 1e-6`；两条的gripper均为逐值0差。搜索3个已记录来源共150个demo，对整个较短序列搜索较长序列的所有固定偏移连续片段，分别检查5020/4966个窗口；每条只有1个窗口/1个demo满足阈值。没有插值、任意跳帧、移除前缀或时间缩放，也没有以nearest误差认定身份。

旋转诊断沿用已有审计投影：`Rotation.from_rotvec(HDF.actions[3:6])`的前两列与R1的`[3:9]`比较。未尝试其它旋转格式、比例或轴变换。两条的rotation RMSE分别为`0.0003579384634`、`0.0003016935325`（六列元素量纲）；这些残差超过原门槛，不能升级为全10列命令身份。

数据只读，未读取或运行draft配置，未读取候选评测结果，未新增模型请求/物理步。HDF完整SHA复用原审计结果，并核对读取前后size/mtime/ctime未变；未声称本次重新hash全部HDF字节。

- cream源文件：`outputs/wam_checkpoint_acceptance_20261019v2/assets/put_the_cream_cheese_in_the_bowl_demo.hdf5`，SHA256 `ee58a7d72c143a71bf1448674e70940ac7781c1d1bb10a1b0c7a5c69b26abbef`。
- 结果：`outputs/wam_stability_20261021/fresh_offline_episode_exclusion/partial_native_four_column_identity.json`，SHA256 `3c6df9fa3fa1478510406573eb3c99eae4a96c58e8dd778fcd7a067600cca90c`。
- 新只读入口：`scripts/audit_wam_partial_native_identity.py`；原reviewer审计脚本及原始证据保持原字节。该入口ruff check/format通过。

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/audit_wam_partial_native_identity.py --output outputs/wam_stability_20261021/fresh_offline_episode_exclusion/partial_native_four_column_review.json
```
