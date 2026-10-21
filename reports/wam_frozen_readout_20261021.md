# 025：冻结WAM主体的残差readout结果

**结论：本候选未通过事前teacher保护门槛，不进入96次完整生成、物理仿真或fresh验收。** 留出开发集active10速度MSE总体下降28.53%，但sigma=.8误差变为原来的2.043倍；生产与独立审计的门槛结论均为False。目标“稳定成功”尚未达成，本项没有新增state预测或任务成功证据。

固定原22-source non-EMA iter8400、H40/20Hz、SDPA、guidance1/35步/shift5/active10，权重位置沿用[024正式报告](wam_current_training_inference_numerics_20261021.md:5)。原WAM所有参数冻结；新readout为12,810参数，针对完整LIBERO domain、normalization31、共享projection0及active10同时匹配的请求。其余域与118槽保持，未原地修改共享head；见[上下文](../src/s2a/wam_readout.py:25)与[hook](../src/s2a/wam_readout.py:193)。它属于生成模型适配，不能归为S2A机器人响应补偿或原checkpoint通过验收。

按[025登记](../plans/v0.0.1/025_frozen_wam_readout_20261021.md)，只用12条此前已查看并从fresh保守排除的episode。训练六条bowl1172/1098、wine893/1151、cream898/850；开发验证六条bowl1254/1126、wine868/1139、cream1209/1176。每EP四个固定完整H40 anchor、sigma=.2/.5/.8、noise seed42；训练/验证各72次前向、2880 endpoint行。完整EP隔离，验证不参与fit；这些数据不称预训练未见或新无偏验收。

实际执行3份zero readout完整生成，40×128 raw及normalized均逐值复现011原缓存，之后144次真实future监督的single denoise。全部249次边界计数和144份hidden/base/target/xt/mask/条件/RNG均经冻结独立CPU checker核验。真实future RGB/action只用于teacher监督；此阶段没有在线闭环任务。

固定phi=[hidden/sqrt1280,1]，拟合active10(v_target-v_base)，episode/anchor/sigma/endpoint等权、权重总和1，ridge=.1且bias同样正则。另fit一次10参数constant-bias对照，共两个参数组，无网格或重试。[fit与评分](../scripts/fit_wam_readout.py:266)使用实际TorchCPU部署路径的BF16输出；数学单次FP64→BF16 ties-even见[实现](../src/s2a/wam_readout.py:84)和[数值补充](../plans/v0.0.1/025_bfloat16_rounding_addendum_20261021.md)。指定GPU6真实合成CPU/CUDA bitwise资格已通过；本次非零拟合权重未进入GPU完整采样。

下表为六个开发验证EP的量化后active10标准化flow速度MSE，**不是米/rad误差、未来state误差或任务成功率**。

| 范围 | 原模型 | 残差readout | constant bias | readout/原模型 |
| --- | ---: | ---: | ---: | ---: |
| pooled | 0.01013023 | 0.00723977 | 0.00727350 | 0.714670 |
| sigma=.2 | 0.02577898 | 0.01656281 | 0.01662608 | 0.642493 |
| sigma=.5 | 0.00332227 | 0.00252171 | 0.00252941 | 0.759031 |
| sigma=.8 | 0.00128945 | 0.00263479 | 0.00266500 | 2.043343 |

事前要求pooled≤.95原模型、严格优于bias、每个sigma≤1.05原模型；最后一项失败。[生产门槛](../scripts/fit_wam_readout.py:253)与[独立门槛](../scripts/verify_wam_readout.py:809)一致。训练集同样出现sigma=.8退化2.155倍，pooled改善26.74%。constant-bias已解释大部分总体收益；readout相对bias仅再降低0.464%验证MSE。本固定目标/正则/特征方案尚未获得跨噪声时刻的合格改进，不据此否定所有更广模型训练方向。

独立checker不导入生产fit/apply/评分，用另一种增广最小二乘重建两个参数组，按rtol=atol=2e-12核查参数及全部分组量化评分；实际参数最大差未保存，不补造该数字。其CPU验证通过表示来源和重算一致，eligible_for_sampling仍False。根据信息完整且失败的teacher门槛，没有执行条件96次生成，因此没有readout H8/H16/H40 state指标、效果比较或物理成功率。

原GPU6模型正常wait0/reap且未发信号，原owner在模型退出后10秒资源观测未见显存0而TimeoutError；原owner False、日志、prepared与53执行源均保留。随后root和两份独立审查均确认GPU6显存0、compute-apps空、原两birth已终止，原超时每次内存值未持久化，原因未知。按[资源接续补充](../plans/v0.0.1/025_release_observation_addendum_20261021.md)使用新隔离入口，只补正式CPU审计/原两fit；没有重复features。新接续admit、fit及其CPU子进程均正常退出；后续sampling实际CLI dry-run在任何模型加载前被teacher门槛拒绝，输出目录未建立。

| 实际阶段 | 数量/结果 |
| --- | --- |
| zero完整生成 / teacher single | 3 / 144，共249 denoise，全部独立核验 |
| 生产fit / 独立参数重建 | 2 / 2，固定同一6train/6validation |
| 条件完整生成 / 物理步 / fresh | 0 / 0 / 0，门槛未过 |
| 原冻结快照QA | 499 passed / 1外部integration deselected，305 Python文件Ruff通过 |
| 新隔离接续回归 | 32 passed，含新增6项；当前307文件Ruff check/format通过，不与499相加 |

| 主要证据 | SHA256 |
| --- | --- |
| [完整feature producer](../outputs/wam_stability_20261021/frozen_wam_readout/features/result.json) | `b24be09d2d099f303b86492cf66b399e7445e01782b0b65cb07343c30412233b` |
| [全249/144独立数学审计](../outputs/wam_stability_20261021/frozen_wam_readout/review/features_verified.json) | `1582f5e37c4ff166da32483db3f30258ead410c136d15b9b96f24ed67cd84a83` |
| [两组fit及量化评分](../outputs/wam_stability_20261021/frozen_wam_readout/fit/fit_result.json) | `8753010c199fd83760aadb1924b86e2b43b5ff59394b5fc4a915caa6670c5405` |
| [独立参数、评分与False门槛](../outputs/wam_stability_20261021/frozen_wam_readout/review/fit_verified.json) | `3456708a75821a1c69afc20622aa8eaabd8c8c95a5569609b3d7d3efb12841bf` |
| [新CPU接续正常完成](../outputs/wam_stability_20261021/frozen_wam_readout/orchestration/release_continuation/fit/result.json) | `db98ace0fc695c6917a5fb6924e1e23bdc4e317065287e3bd246bf88f10dd78e` |
| [实际sampling准入拒绝](../outputs/wam_stability_20261021/readout_release_review/sampling_teacher_gate_rejected.json) | `50e00722b7b044077ff25d7a66f88f2c368b7f16a4148c76d25bc69f80dc2389` |

本轮未改lambda、容量、sigma、anchor或门槛以挽救025，原失败候选不部署。下一步根据噪声时刻误差与flow的物理定义选择另立项的模型学习方案；任何新方案仍须训练EP隔离、独立teacher与完整生成state验收，随后冻结执行链路再进行多任务双seed闭环。最终预留12初态×2seed、≥20/24且每任务≥4/8及末40端点连续成功的目标保持，fresh性能未打开，无真机。
