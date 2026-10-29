# 保存3543的末周期平移修复

2026-10-29：修订后的155项CPU QA、四项准入、14响应生产及14响应完整独立复算均已正常关闭。
12个固定候选中2个通过原安全约束和全部保留条件；选中请求2，即末周期x命令减0.001。
terminal从`-1.20453558423459e-6`恢复至`2.7622793190069904e-5`，保留历史3543的
cursor9微小改善，但完整H12 witness仍为0。实际仍cursor8、task0/grasp0，未取得控制资格。
GPU继续禁用，原18正确参考/12 WAM开发/24最终留出门槛及总目标不变。

本轮只验证一个有限修复假设。生产和独立复算合计28完整响应、8400数值内步、8738查询；
新增物理积分、世界模型调用、新初态验收和参考消费均0。首次环境引用错误的零响应失败完整保留。

## 输入与固定合同

[执行前计划](../plans/v0.0.1/terminal_only_tail_repair_20261029.md)选取已完整复算的
[H12 SQP结果](restored_anchor_sqp_20261029.md)中的
`answer.phases[50].ray_attempts[3].entry`，即response3543。
前项因果分析先筛出40条prefix9保持、cursor9严格下降、仅terminal失败的ray，再按
cursor9归一化最大平方误差和响应序号排序；3543为首项。该选择没有使用本轮响应。

输入封装位于`outputs/terminal_only_tail_repair_inputs_20261029/bundle.json`，SHA256为
`637d172935c3dd852b03347f1d9b97f863089dd85a637ca5f539c6a66b19ee26`。
其中保存完整历史entry、原source/registration、controller合同和已准入的网格。
实际历史entry恰有11个数值字段，包含完整rollout和audit；没有通过省略这两项取得exact。

各响应从同一actual368的实测q/v重新开始，使用completed367命令、旋转latch、原已生成
12点参考、固定finger与控制器、2ms内步和每周期25内步。保持原10mm/0.05rad姿态caps、
command box/rate 0.25、关节裕量0.05rad、braking 0.1s及e7 terminal normal。
全部13条不同命令先准入，任一失败即停止，不裁剪或换步长。

固定请求次序为首anchor，末周期x/y/z分别依次`+0.001,-0.001,+0.0005,-0.0005`，
最后付费anchor repeat，共14响应。每次完整H12为300数值内步和312查询，另有一次当前状态查询。
每条保存12点误差、原九组audit和五组非线性安全slack。候选必须同时满足原audit、
所有安全slack至少`1e-8`、prefix至少9、前11完整period/275内步/11端点exact、
12个旋转goal exact、历史首响应exact和首末repeat exact。
排序使用terminal降序、实际扰动Linf升序、请求序号升序，选择还要求terminal严格改善。
本轮没有使用旧E17经验值，也没有“充分误差余量”资格标签。

## 首次失败、修复和来源保留

提交`aa74655`的153项组件QA通过（128.03秒），四项准入亦通过，但首次真实producer
在读取`source_reg.control_ref`的`cv2_guard_registration`时发生KeyError。
该文件是仅含controller contract的记录；环境guard实际来自
`s2a.terminal_development_execution.CONTROL_REF`。原测试fixture把两种结构合在一个
文件中，未覆盖真实结构差异。故原QA和dry成功不能证明环境入口可执行。

首次子进程1191209由原session23050实际wait/reap，exit1，无信号、无自动重试。
异常发生于模型构建前，模型/响应/查询均0；旧结果未初始化这几个计数字段，其零成本由
原代码执行顺序和错误位置确定，不将缺失字段伪称为已记录计数。
旧结果`complete=false, passed=false`，原日志、QA及owner receipt均保留。

提交`9a69049`将环境guard与controller参数分开，在dry阶段就读取并绑定guard、
exact probe及runtime source refs。新增真实artifact结构测试和controller-only记录的dry
拒绝测试，且初始化失败成本计数字段。修订后的155项测试通过（39.57秒）；
117来源由原110数学来源加本轮7来源组成，执行6个针对性测试模块。
这不表示重跑了原110来源对应的全部历史测试。

原7来源以`aa74655`和
`outputs/terminal_only_tail_repair_v1_sources_20261029/manifest.json`保留。
manifest SHA256为`de14feb6532a1f79e6d343c76170b4b7a540b1c01258400424a5bb6d32b7f86c`，
逐项记录原路径、原SHA及当前仓库内字节一致的副本路径。只有下列3项在修订中改变：

| 文件 | 首次版本SHA256 | 修订版本SHA256 |
| --- | --- | --- |
| `scripts/probe_tracking_v5_terminal_only_tail_repair.py` | `7bb483ffe7c5659d67e5ae9726e0c00a40806204086b1fec038d820a7af75504` | `6c4dc3027ab220c57b9ddda94cd30de2f4812cc2d05bbb4b99bba92caaba588d` |
| `tests/test_terminal_only_tail_repair_execution.py` | `3b02a2c412e35061fd04f14f91aa9f385f82f29aeb103d3ba9678de94dc7d5c6` | `5239987d3b33b7968c80ff217432c5785d71f2a29adecb3575c79d49e406b8ae` |
| `plans/v0.0.1/terminal_only_tail_repair_20261029.md` | `6e5b7ab538d21a5752355b663d08aa8a6cdbabceb5c288097db4818eafa6a9e5` | `b1b36147a9ee3760ffa27b6634de57c08f7908dfa02038629d986feed0f86871` |

原110数学来源和此前29项bootstrap来源未改动。修复执行使用新独占输出目录
`outputs/tracking_v5_terminal_only_tail_repair_v2_20261029`；输入3543、网格、预算和科学门槛保持。
这是一项登记后的入口修订执行，未覆盖首次失败或追加搜索预算。

## 修复结果与独立复算

生产请求2和4合格，排序为`[2,4]`；分别为末周期x减0.001和减0.0005。
选中请求2的实际扰动Linf为`0.0010000000000000009`，terminal改善
`2.8827328774304494e-5`。其他安全最小slack仍为`6.134673746771568e-8`，
没有因终端修复变宽。请求4的terminal为`1.3208972937890128e-5`。

12候选全部通过prefix9和因果exact检查，但10条未满足完整安全条件。例如y加0.001
的terminal达到`7.648150778421291e-5`，比选中项更大，却因其他braking约束失败而拒绝，
非terminal最小slack为`-8.277248836563444e-6`。不能仅按末端速度选择命令。

选中项cursor9的位置误差为`0.011465200140606792`m、旋转误差为
`0.05732641137643227`rad，分别仍超过原0.01m和0.05rad门槛。
cursor9的最大归一化平方误差为`1.314526976519977`，与输入3543完全保留。
相对前项buffer合格anchor3557的`1.3145505007230824`，只降低
`2.3524203105473873e-5`，约`0.0017895%`。这项微小改善没有增加达标前缀长度。

生产首响应与历史3543、全部因果前缀/旋转goal和付费repeat的exact门均通过。
独立checker重新构造14次完整响应，并独立重算网格、全部保存响应、audit、strict标签和选择。
其`grid_check`、`ledger_check`、`independent_repair_check`均通过，选择仍为请求2；
`strict_disagreements=[]`和`boundary_classification_disagreements=[]`。

exact的适用范围需分开记录：历史exact在保存的生产数值路线内检查，独立路线的前缀
和repeat在该路线内部检查；两条数值路线之间沿原浮点容差比较，不宣称逐字节相等。
完整安全slack的最大跨路线差为`1.7763568394002505e-15`，归一化平方误差最大差为
`1.1879386363489175e-13`，请求/施加力矩最大差为`1.3855583347321954e-12`。
原生四元数JIT与Python goal路径有112项差异，最大`2.9802322387695312e-8`，
沿既有原生路径核对保留；没有以该差异放宽原姿态或安全门槛。

生产完成时重核2536来源，checker完成时重核2538来源；各owner的117来源与driver均保持。
没有为本轮另外新增全来源闭合阶段。producer与checker都为完整witness0、
`control_qualified=false`、`empirical_error_bound_used=false`。

| 实际阶段 | 响应 | 数值内步 | 查询 | 完成状态 |
| --- | ---: | ---: | ---: | --- |
| 首次producer | 0 | 0 | 0 | guard引用失败，exit1 |
| 修订producer | 14 | 4200 | 4369 | exit0，partial查询0 |
| 独立checker | 14 | 4200 | 4369 | exit0，完整核验14条 |
| 合计 | 28 | 8400 | 8738 | 无自动重试或额外搜索 |

新QA的session72970、四命令准入session46373、producer session23282及checker
session87081均已由root实际wait到exit0。producer子进程1213241、checker子进程1217335
均actual_wait/reaped true，无信号、无重试。全部运行使用CPU、空CUDA/LD、单线程和seed42。

## 结果引用

以下均为当前仓库下路径，完整SHA从对应结果文件读取；旧结果保持其原声明。

| 产物 | 路径 | SHA256 |
| --- | --- | --- |
| 首次QA | `outputs/terminal_only_tail_repair_qa_20261029/result.json` | `852d5a0b2337ffbef04b75a08cfedc074459431b379cf700d6f793e790fef225` |
| 首次准入 | `outputs/terminal_only_tail_repair_admission_20261029/result.json` | `5eab54a1e372a16e52a5791d831d1e7052e59bd01c133d8dcd20efb9b9c75faf` |
| 首次失败owner | `outputs/terminal_only_tail_repair_producer_20261029/result.json` | `e05269e208471995366831acbdfe9b5264bd44cec415aa7783144bfd08821a7e` |
| 首次失败结果 | `outputs/tracking_v5_terminal_only_tail_repair_20261029/verified.json` | `75f89ac3b1365efa4b61582bd8749329f92db6f423ee9b1a932105526219669c` |
| 修订QA | `outputs/terminal_only_tail_repair_v2_qa_20261029/result.json` | `cc9eed3635f58e84a985c6b079c82288b18cc1e1154f6d347c7c795c16590395` |
| 修订准入 | `outputs/terminal_only_tail_repair_v2_admission_20261029/result.json` | `e2f88333d9a55d24e5d2f9b758cc4e3acfe56b0eeb5d2f683194da0131ec87dc` |
| 修订producer owner | `outputs/terminal_only_tail_repair_v2_producer_20261029/result.json` | `ce5ade1bd72b4a60dcf7bc3f7434dee2482ac45c1e0a9944a731a7bd2d3a6cc4` |
| 修订producer结果 | `outputs/tracking_v5_terminal_only_tail_repair_v2_20261029/verified.json` | `5d346adc26e53da29c4ea5a1b06da161c53b5979e8863bad8ad056b8b8e139d1` |
| checker owner | `outputs/terminal_only_tail_repair_v2_checker_20261029/result.json` | `261f5837fad35a3015d8ac6db8a77863cefb4c70d41dc37390f0748bb13afc77` |
| checker结果 | `outputs/tracking_v5_terminal_only_tail_repair_check_20261029/verified.json` | `49ed51615a9fb9124303cb94118ba811558fd36ab94579bfa2e4ad22cdc7c6c4` |

有限网格已经关闭。证据支持“末周期平移可修复3543的终端行，同时保留前缀及微小cursor9
改善”，不支持完整轨迹或真实控制成功。其他安全余量未增加，cursor9仍超caps，故不能
自动重启同一4000响应SQP或进入物理pilot；后续需要另行提出能解决剩余跟踪误差的实质性假设。
本报告不消耗最终留出，也不改变总目标或GPU禁用状态。
