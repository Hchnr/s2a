# 全布局执行细化

本页落实[总体设计](repository_design.md)的D3–D5；首批D2已提交`b3df0e7`。
原研究目标、原验收门槛和GPU禁令保持。[逐文件设计清单](../migrations/layout_plan_20261029.json)
明确每个原文件及已有新增包的当前位置、计划位置和职责。它是执行前的布局设计，
不是执行完成证明；实际位置/SHA另由累计current清单登记。

## 领域边界

| 领域 | 分块与定位 |
| --- | --- |
| 几何与命令 | `geometry`保留公共包身份；`controllers`区分DROID与OSC，`adapters`维护反馈、辨识及在线入口 |
| 参考 | `references`处理已生成轨迹、时序、融合、漂移；不混入机器人响应辨识 |
| 动力学与环境 | `dynamics`保存局部前向原语；`backends`保存通用观察/推进/环境构建 |
| 研究控制 | `research/joint_control`按preview、limits、tracking、inverse等分块；`trajectory_search`独立保存整轨迹优化与账本 |
| 响应研究 | `research/response_models`区分causal、forward、inverse、wrench、nonlinear；不以误差拟合代替控制成功 |
| 世界模型 | `research/wam`区分training、readout和tracking；DROID初始研究单列`research/droid` |
| 固定运行时 | 固定368分支及其bootstrap/replay属于`research/native_runtime`；不升格成通用后端 |
| 基础设施与评价 | 文件、指纹、产物、生命周期与评价分层；基础层不得导入研究层或scripts |
| 脚本 | `data`、`simulation`、`tracking`、`wam`、`verification`、`maintenance`；大领域再按实际问题分组，统一`scripts.*`身份 |
| 测试 | 镜像领域；fake CLI属于unit/entrypoints，真实外部集成保留原integration marker；共享fixture进support |
| 证据 | 配置/计划/报告按相同研究问题归组；跨领域历史、总目标和状态单列，原字节保持 |

接口合同当前由真实的controller/reference类及其测试维护；本阶段不创建空的
contracts包。测试根保留`__init__.py`、`conftest.py`和简明`README.md`三项，
以保留pytest配置作用域；结构检查的测试根阈值据此为3。配置/计划/报告根各保留
一份导航，其余按主题归组。历史报告正文的相对链接以原声明路径的Git视图解释。

详细代码设计覆盖2348个基线source/script/test文件及22个已有新增文件，代码包目录
205个，设计中最大直接文件数73。171项初始不确定分类已有逐项决定。独立验证
仍保留各自实现；本次不合并算法、不调整容差、不比较新策略。

## 执行与提交

1. 先抽取两处库反向依赖的最小真实函数闭包：STATE终端/worker核验、native
   endpoint与render观察。公共publication的来源函数下沉到基础设施。
2. D3a迁移src领域，修改全部当前调用方的明确导入及根定位，脚本和测试暂保留
   当前物理位置；核导入来源、全量节点及对应CPU回归后提交。
3. D3b先处理高扇入脚本的依赖闭包，再逐领域迁移调用方。保留同一函数对象与
   独立审核边界；不留上千根wrapper，不装全局import或文件API hook。
4. D3c迁移测试及fixture，核原节点的函数名、参数和数量；不删测试来消除错误。
5. D4按已列出的928份配置/计划/报告设计归组，核原SHA。根README收敛为目标、
   当前状态、接口和开发导航；旧README已保留原文，历史链接按原声明位置解释。
6. D5验证每个基线文件去向、历史身份、当前导入、节点发现、样式、CPU回归和
   当前入口dry，登记剩余历史执行边界后提交收尾。

采用单份累计`docs/migrations/repository_layout_20261029.json`描述当前布局；
每个阶段更新它的current SHA并保存对应提交。D2的已提交清单作为历史记录保留，
不会与累计清单同时传给locator形成重复旧身份。新增guard/输入层在基线后创建，
可在生成累计清单后pin其SHA，再通过当前源码收据绑定自身，避免自引用哈希。

独立metadata审核若原合同要求仅标准库，则保留独立的标准库anchor根定位；
不为统一路径API而引入生产库，也不放宽其禁止producer/数组导入的测试。

所有历史实验入口的路径和源码变动都需要新的运行登记。目录迁移、可导入或dry
通过不自动授权重开已有输出、恢复任务，或继承旧运行时与控制资格。当前入口
优先验证；历史原始配置和结果只保留身份，不伪装成已重新执行。
