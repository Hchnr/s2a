# 新初态预留的独立metadata核验（2026-10-21）

12条新初态预留通过独立只读核验，尚未运行任何预留任务或选择候选控制器。核验涉及已保存fixture、来源metadata/XML/初态与原始expert actions文件，不读取任务成功结果；模型请求和物理步均为0。

配置逐字匹配commit `ac2663f`，SHA256为`a5d5f87fe28885f09112b18b20cd8ea79e18e0a37d16949abc958f03748d8d65`；fixture中配置副本与代码指纹的配置SHA相同。5份历史实际fixture共有36条记录/18个不同初态SHA；新12条SHA各自唯一，与这些旧初态无交叉。

| 任务 | 保守排除demo数 | 剩余候选数 | seed42独立重算所得4条demo |
| --- | ---: | ---: | --- |
| bowl → plate | 25 | 25 | `20, 2, 28, 31` |
| wine → rack | 9 | 41 | `34, 10, 5, 28` |
| cream cheese → bowl | 9 | 41 | `35, 23, 39, 6` |

按原顺序共享同一个`default_rng(42)`，从数值排序且按demo/旧初态SHA过滤后的候选列表无放回选4条，独立重算与已保存fixture完全相同。没有重新运行prepare入口、生成新fixture或执行候选任务。

3份源HDF本次各独立完整计算一次SHA，读取前后size/mtime/ctime不变；12条fixture XML逐字等于对应HDF demo的model XML，初态数组和SHA与HDF第一条state一致，expert `.npy`文件SHA与fixture记录一致且数组逐值等于对应demo的actions。44份输入文件SHA保留于审计JSON。

通过项仅限旧物理初态SHA排除、已登记demo排除（包含两个疑似来源的保守额外排除），以及这些新预留记录到HDF的metadata关联。**不认证历史R1旋转身份、历史图像/pose/初态物理同步、预训练未见或任务能力。** 两条仅XYZ+gripper匹配的来源仍未升级为full native control身份。

- fixture：`outputs/wam_stability_20261021/fresh_acceptance_reservation/simulation/fixtures.json`，SHA256 `6e279c4db07a06f7399f53eb298cb77b671305accfe5b520598f1ee1c8e134c6`。
- 独立审计：`outputs/wam_stability_20261021/fresh_reservation_independent_metadata_audit.json`，SHA256 `b1b5bef48457c3ff3e688a8a399a48745ce4ffcf98ae7423d51439c2f903816d`。
- 新只读入口：`scripts/audit_wam_fresh_reservation_metadata.py`；ruff check/format通过，原prepare和主模型配置未修改。

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' outputs/libero_env311/bin/python scripts/audit_wam_fresh_reservation_metadata.py --output outputs/wam_stability_20261021/fresh_reservation_metadata_review.json
```
