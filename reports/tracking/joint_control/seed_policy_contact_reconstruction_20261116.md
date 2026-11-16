# 两种 seed 排序的已执行端点接触与放置

2026-11-16。按[登记](../../../plans/v0.0.1/tracking/joint_control/seed_policy_contact_reconstruction_20261116.md)，
从同一 18 例的两条已独立重放轨迹恢复全部 **9,458 个实际端点**：seed-first
4,778 个、有界排序 4,680 个。原综合资格分别为 16/18 和 15/18；这是非盲
正确参考开发集，不是 WAM 或新初态的成功率。

静态 `mj_forward` 重建的双侧指垫接触与原生 grasp 标签、目标区域和承载物
接触重建的原生成功标签，在 9,458 个端点均一致。重建 TCP 与保存的目标误差
最大差 `2.23e-16 m`，目标物体位置与已核验状态列最大差为 0；恢复时最大
四元数归一化变化 `2.23e-16`。接触列表按原生 `check_contact` 语义计数，
不是接触力，也不覆盖每周期中间 25 个内步。下表数值是两个完整**已执行**
轨迹各自的计数；早停时两组长度不同，不能直接比较其总数。

| case | seed-first / 有界执行周期 | grasp 端点 | 承载物接触端点 | 区域条件成立端点 | 任务成功端点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 03 | 170 / 170 | 0 / 0 | 91 / 112 | 113 / 0 | 91 / 0 |
| 06 | 334 / 230 | 86 / 85 | 170 / 79 | 120 / 6 | 99 / 3 |
| 07 | 86 / 215 | 2 / 44 | 0 / 88 | 0 / 92 | 0 / 86 |
| 11 | 281 / 158 | 83 / 60 | 95 / 4 | 112 / 0 | 95 / 0 |
| 12 | 228 / 228 | 60 / 61 | 103 / 103 | 5 / 95 | 4 / 91 |

step 为从 0 开始的动作索引，其保存端点为该动作完成后；不把短轨迹结束处
当作完整任务末尾。case 07 的 seed-first 在 86/215 周期停止，后续没有数据；
case 06 / 11 的有界策略也分别在 230/334、158/281 周期停止。

## 两个完整长度的任务差异

**case 12：两组都接触 rack，但放置区域条件不同。** 两组各有 103 个
目标物体/rack 接触端点；最后 40 个端点均持续接触 rack，grasp 均为 0。
seed-first 的 40 个端点都在原生 rack 区域第二个横向边界外，中位余量
`−2.339 mm`；有界排序均在边界内，中位余量 `+4.183 mm`。其余横向及高度
边界在该末尾均满足。两个轨迹的目标物体位置最大差 `8.838 mm`，TCP 位置
最大差 `2.521 mm`；物体位置在 step 101 首次相差超过 1 mm，grasp 标签在
step 78 首次不同，任务成功在 step 118 首次不同。step 116 两组均有 grasp、
rack 接触及任务成功；step 141 有界组再次成功并保持，而 seed-first 未进入
区域。rack 的“区域坐标”严格沿用保存版本 `SiteObject.under` 中的矩阵
乘法和严格不等式；它是环境评分语义，不把该公式改称一般坐标变换。

**case 03：有接触仍可能未放到目标位置。** 最后 40 个端点两组都接触盘子，
但有界组的碗始终不满足目标空间条件：相对 `30 mm` 水平阈值的中位余量
`−79.520 mm`，且碗原点比盘子低约 `2.483 mm`。seed-first 分别为
`+14.057 mm` 和高出约 `5.029 mm`，40/40 成功。两组物体位置在 step 43
首次相差超过 1 mm、step 51 超过 10 mm；最大差 `146.465 mm`，TCP 最大差
`2.476 mm`。端点接触与单周期 TCP 精度均不能单独替代放置成功。

两组在首次动作分歧后持续各自反馈，不能把最终结果归因于某一个早期动作或
某一次接触。物体位置、目标区域与接触是仿真特权信息，只用于离线评分；
不能将上述 `2.339 mm` 等信息直接输入在线控制器，也不能据此按案例拼接策略。

## 独立复核、资源与限制

[独立复核登记](../../../plans/v0.0.1/tracking/joint_control/seed_policy_contact_independent_audit_20261116.md)
重新读取原始状态和标签，对全部 9,458 个端点复算状态指纹、目标状态列、
已保存接触条件及原生标签，并从各自 XML 的 site 尺寸以标量公式复算区域
边界；共同前缀差异和末 40 个已执行端点统计一致。复核不重新运行 MuJoCo
forward，不能独立证明接触列表没有遗漏或局部坐标重建本身正确。

正式重建使用 18 个 CPU 模型/数据和 9,458 次 forward，0 新积分物理步、
渲染 context、GPU、WAM 或 fresh；先完成 03 / 07 / 12 三任务 worker，再
执行其余 15 个。重建和独立复核各 18 个 worker 均实际 wait/reap，源码与
输入前后一致。两次 64 并发相关回归分别为 69 / 73 项通过、0 跳过；
32 并发专项分别为 26 / 17 项通过。三次包含真实 MuJoCo 合成测试的 QA
共新建 6 个微型合成模型、执行 9 次合成 forward，均无积分步或渲染；不计入
正式 9,458 次实际端点重建。

首次专项检查的 28 通过 / 3 失败及其 OSMesa 与历史源码指纹原因保留在
[事件记录](seed_policy_contact_qa_incident_20261116.md)；后续独立路径通过
不表示旧历史来源检查已通过。正式重建源码为 `c8668c97`，独立复核源码为
`2716637e`。下一步应设计仅使用实时可用观测、机器人状态和参考的统一候选，
并在原 18 例从 reset 执行与重放；不能用本次离线真值或接触诊断替代 WAM
开发、最终留出或真机验证。

| 本仓库 `outputs/` 路径 | SHA-256 |
| --- | --- |
| `seed_policy_contact_reconstruction_dry_20261116.json` | `30bd2bb8dd40c86a96be336970e778234fc404072144c689eba16dfa2c3524ac` |
| `qa/seed_policy_contact_reconstruction_unit_v2_20261116/result.json` | `b865982e186c46bad6dc68004710047f8a1cfc0b94d2a190c30743248b93d03f` |
| `qa/seed_policy_contact_reconstruction_final_20261116/result.json` | `497105d1e69538a5037910dc39a4384774da94eede383582c0cb4a98128c8084` |
| `seed_policy_contact_reconstruction_20261116/workers.json` | `9edbb5fd3fc25f6c9ee11186bd205ee2adeef26b52a09e44c9130758a51bc82a` |
| `seed_policy_contact_reconstruction_20261116/result.json` | `498b3d32cb40e1aa65b61e27c1ca091472b4a5bbab07e8b12dbda44c937ee00d` |
| `seed_policy_contact_independent_audit_dry_20261116.json` | `0ec535fe3b8c1b2c19fc5a82a9368381282306f9020cee16b9134018b60ca988` |
| `qa/seed_policy_contact_independent_audit_unit_20261116/result.json` | `60579e33f350bed73c56f810b1a2bbb64f4a437b657333471f92241296dd81af` |
| `qa/seed_policy_contact_independent_audit_final_20261116/result.json` | `4214044307828f27f2b2cb709ed2d76b76ff78bcbb389e016bc301d49c5dedc6` |
| `seed_policy_contact_independent_audit_20261116/workers.json` | `d5366c7549e8539fd2ff2b96aabe60974adac2c4b00abd1cd21f8653a2cd6d47` |
| `seed_policy_contact_independent_audit_20261116/result.json` | `9d06cb7e8f2049deefbf08c9d6a67bf27ce02b70eec82518126c2e0de23cfbd3` |
