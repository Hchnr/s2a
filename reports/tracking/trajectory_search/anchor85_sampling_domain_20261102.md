# anchor85局部模型采样域与实现检查

2026-11-02，设计先行提交23a1456。此轮完成命令域检查及采样/拟合基础实现，
没有执行新的机器人响应或重建实际机器人局部模型，不授予控制资格。

## 发现与实现

原±1e-4单轴探针只能覆盖64/72轴。零基展开轴4/6/12/31/32/38/44/48的两符号
都至少违反一个原command门，因此不能用旧完整单轴stencil合同假称新anchor已刷新。

按长度1..12、起点、通道、符号固定顺序枚举连续区间同通道联动扰动：936个中
605个满足原box/rate/global/trust及非零门。响应盲选的独立方向为64个单时段、
7个双时段、1个三时段。实际浮点位移矩阵rank72，condition约4.04891734；
12个因果前缀列rank6/12/…/72，各condition≤4.049。没有clip、改幅度或随机补选。
命令域合格不证明非线性安全，所有将来的数值探针仍不得下发。

`src/s2a/research/trajectory_search/command_domain_stencil.py` 新增：

- `select_interval_basis`：只使用原命令几何选择72方向；原STEP1e-4、radius.005，
  与旧preflight相同的1e-12命令边界数值比较，未放宽原安全/跟踪容差。
- `prefix_geometry`：实际位移矩阵前缀必须满列秩且condition≤10，否则拒绝。
- `fit_causal_response`：直接在各row时段的命令前缀中求最小二乘；未来参数不参与
  求解，保留逐row残差和rank/condition。这是模型结构约束，不是实验证明物理因果。

实际anchor重算逐值复现保存检查的全部72commands、实际displacements及选择顺序，
枚举/准入计数相同。旧单轴build_response_stencil及旧算法/配置/结果未修改。

## 验证与计费

24项jobs32专项全部通过、0skip/collection错误，worker实际回收、来源前后不变。
覆盖速率饱和链的联动采样、不可辨识域拒绝、秩/条件数拒绝、合成因果线性响应
恢复、未来不一致观测保留残差，以及原固定尺度准入。Ruff/format、Git空白通过，
未运行全仓回归。合成拟合测试不作为机器人模型或未见数据验证。

本轮实际机器人响应0、动力学查询0、QP/NNLS0、实际机器人模型拟合0；
命令域936次几何枚举与软件线性代数检查单列。0物理/GPU/WAM/fresh/最终留出。
所有数值库CPU单线程、空CUDA/LD，无共享环境改动，未启动模型采集owner。

| 产物 | outputs相对路径 | SHA256 |
| --- | --- | --- |
| 单轴可用域 | anchor85_sampling_domain_20261102/result.json | 779b9c325b3bb21f44bf63f4287bd29287dcc798dc5419dd376cdc7d65461a98 |
| 联动方向/前缀几何 | anchor85_sampling_domain_20261102/coupled_basis.json | a35a48af6fed010eea746c4e6c4d4463de7a4ed817d88c9cfeed4c86ad22ddc7 |
| 实现逐值复现 | anchor85_sampling_domain_20261102/implementation_comparison.json | 917364a5bcf03783bed214997a0c399f63d9c5f19ee858988602a149626125f6 |
| 24项QA | qa/command_domain_stencil_20261102/result.json | 306bd93438871100698bffc0bdc771283dd8cf47f11c3c6b11c6e09457785880 |

下一步独立登记CPU并行完整响应采集预算、独立审核、源码/输入绑定和owner回收，
再实际重建Js/Jq/Je并用未参与拟合的新方向验证。不能把本轮满秩与合成检查当作
实际模型已刷新；原可靠跟踪和18/12/24、稳定20/24且每task4/8总验收仍未达成。
