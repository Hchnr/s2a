# S2A 仓库结构设计与迁移协议

状态：设计完成后按下述阶段执行。基线提交：`7b2712850dea54090d61e564380f88825ad8b8f6`。
本次任务是整理现有实现、入口、测试和研究证据；控制策略及验收标准保持。

## 1. 总目标与不能丢失的主线

世界模型给出已经生成的期望 state/delta_state 轨迹。S2A 结合当前实测 state、
因果历史和固定控制接口，构造可执行 action；通过同一控制器与机器人/环境后，
以实测轨迹、安全和任务稳定性判断效果。解析、PID、辨识逆、直接逆与残差模型
都属于待比较的方法，目录结构不预设某一种方案胜出。

```mermaid
flowchart LR
    W[世界模型已生成的参考] --> R[参考解码与时间轨迹]
    M[当前实测状态与因果历史] --> A[S2A adapter / planner]
    R --> A
    A --> I[明确的 controller command 接口]
    I --> C[固定控制器]
    C --> P[机器人 / 物理环境]
    P --> M
    P --> E[跟踪 / 安全 / 任务稳定评价]
```

约束：measured/desired/delta/command 分开；固定 frame、TCP、单位、时钟、mask，
旋转遵循 SO(3)/SE(3)。每个 episode 隔离积分、滤波、历史和参考游标。局部模型
rollout、日志回放、独立物理仿真和真机属于不同证据等级，不共用一个success。
原18正确参考、12 WAM开发、24最终留出门槛保持；最终稳定需≥20/24且各任务≥4/8。

原始依据：`plans/v0.0.1/001_state_to_action.md:36-56`；稳定验收沿用
`plans/v0.0.1/036_project_goal_20261023.md`及后续明确修订。

## 2. 当前问题与量化基线

仅统计 Git 跟踪文件，排除用户根目录未跟踪文件和生成产物。

| 区域 | 文件数 | 行数 | 当前直接平铺 |
| --- | ---: | ---: | ---: |
| src/s2a | 386 | 76,792 | 386 |
| scripts | 1,489 | 318,392 | 1,489 |
| tests | 473 | 76,200 | 465 |
| configs | 199 | 188,929 | 199 |
| plans/v0.0.1 | 350 | 7,722 | 350 |
| reports | 379 | 35,735 | 379 |

共3,276文件、703,770行，约100.35MB。问题并非只需改目录名：

- 2,578次裸脚本导入，涉及1,079个导入方、364个被导入脚本；脚本已兼作库。
- 147次裸测试辅助函数导入；移动会改变测试发现和导入来源。
- 1,393个文件涉及source manifest，592个含SHA字面量，历史路径参与证据身份。
- 54处REPO赋值，部分依赖固定`__file__.parents[n]`；增加目录层级会改变语义。
- 当前README混合安装说明、总目标、实时状态和数百条历史进度，缺少稳定入口。
- 日期、vN、失败修复和一次性实验与可复用算法混在同一层。

当前代码静态依赖未发现SCC循环，但有库反向依赖脚本、通用校验带入数据提取、
仿真入口兼管模型调用等层次问题。动态加载不在静态无循环结论覆盖范围内。

## 3. 目标结构

```text
s2a/
├── README.md                         # 简短：目标、当前状态、安装、导航、常用入口
├── AGENTS.md                         # 工作约束与新导航
├── pyproject.toml
├── src/s2a/
│   ├── contracts/                    # 状态/参考/命令、frame/TCP/units/time/mask
│   ├── geometry/                     # SO(3)/SE(3)、几何与运动学原语
│   ├── references/                   # 解码、锚定、插值、过期/乱序、参考游标
│   ├── adapters/                     # 解析、反馈、辨识逆、学习逆等可复用策略
│   ├── controllers/                  # DROID与OSC分别明确的控制接口、夹爪状态
│   ├── dynamics/                     # 前向响应与局部预测；不冒称实际物理执行
│   ├── backends/                     # 环境、observe/step、固定控制器运行时
│   ├── evaluation/                   # tracking、安全、task/stability与证据等级
│   ├── infrastructure/               # 配置、根定位、产物、来源、生命周期
│   └── research/                     # 研究实现，按主题分块；不默认作为公共API
│       ├── trajectory_search/
│       ├── joint_control/
│       ├── response_models/
│       ├── gripper/
│       └── wam/
├── scripts/                          # 薄实验入口及尚待抽取的历史入口
│   ├── data/
│   ├── simulation/
│   ├── tracking/
│   │   └── trajectory_search/
│   ├── wam/
│   ├── verification/                 # 保持独立数学/数值实现
│   └── maintenance/                  # 结构迁移、索引、完整性检查
├── tests/                            # 镜像上述领域，不按测试文件创建一层目录
│   ├── unit/
│   ├── research/
│   ├── integration/
│   └── support/                      # 明确的共享fixture/hand-certificate
├── configs/                          # 同主题分组；预登记的同字节文件也保留身份
│   ├── data/
│   ├── simulation/
│   ├── tracking/
│   └── wam/
├── docs/
│   ├── architecture/                # 总体结构、接口与依赖规则
│   ├── development/                 # 环境、命令、测试、贡献与迁移说明
│   ├── status.md                    # 唯一简明当前状态与待解决项
│   ├── research_index.md            # 按问题/阶段导航，不追加流水账
│   ├── history/                     # 旧README及历史导航，保留原字节
│   └── migrations/                  # 旧路径/SHA/新位置/迁移版本清单
├── plans/v0.0.1/                     # 按goal、tracking、wam等主题分组
├── reports/                         # 按同主题分组，报告和双语PR有明确角色
└── outputs/                         # 生成产物、缓存、环境、原运行证据；本次不搬
```

目录数以领域边界为准。若一个主题文件过多，再按实际问题分块；不创建每文件一个
目录，也不把全部旧文件简单塞到一个archive目录。具体旧→新路径由迁移清单逐项
确定，不能只靠文件名前缀猜测可复用性。

## 4. 模块职责与依赖方向

| 层 | 职责 | 明确排除 |
| --- | --- | --- |
| contracts/geometry | 数据语义与纯数值原语 | 历史outputs、模型服务、实验登记 |
| references | 已生成参考的时序处理 | 读取尚未生成的未来参考 |
| adapters/controllers | 因果决策与明确命令编码 | 隐式env.step、将规划结果写成已发生历史 |
| dynamics | 局部前向预测 | 冒称真实接触/固定控制系统的执行证据 |
| backends | 真实observe/step与状态恢复 | 选择实验胜者或放宽评价阈值 |
| evaluation | 原始指标、完整预算、稳定验收 | 以重建/action RMSE代替闭环成功 |
| infrastructure | 文件、配置、来源及进程生命周期 | 控制策略决策 |
| research/scripts | 候选、数据划分、预算、登记和独立复算 | 作为底层库的反向依赖 |

基础层不导入research或scripts；脚本可调用公共库，独立审核复用合同和纯几何，
不调用生产求解器来证明生产求解器正确。DROID与OSC的尺度/旋转合同分别保留。

`runtime.StateToActionAdapter`、`osc.OSCStateToAction`是现有明确入口；
`prepare → commit → observe`提供可参考的事务边界，但当前带研究refiner的实现
不直接升格为通用API。版本化搜索和固定case/预算/输出路径实现放research；
跨实验验证过的求解、账本原语未来可单独抽取。本次先保持数学及控制行为。

## 5. 历史身份与兼容协议

### 5.1 旧声明与实际位置分开

历史JSON、日志、快照、预登记及报告正文不批量改写。迁移清单至少包含：
`old_path`、原SHA256、Git来源commit/blob、`new_path`、迁移阶段、角色与主题。
身份按`(declared_path, sha256)`索引；同一路径的不同历史版本不能混为一份。
解析必须先确认身份已登记，再尝试存储位置；不能因为某个现存文件恰好匹配就
自动接纳未知历史版本。当前源码改字节时另记current_ref，原身份定位原Git blob。

新locator返回原`declared_ref`、实际`storage_ref`、`migration_ref`。流式核SHA，
拒绝缺失、歧义、越界、错误哈希；不按basename搜索，不访问旧清理根。原字节可
从明确Git blob或既有已登记snapshot定位；必要缓存写outputs，不能恢复旧仓库。
旧源码字节可核验不等于旧环境可执行，更不继承物理或任务资格。

历史Markdown相对链接按原declared_path解释，不能按移动后的物理目录解释。
新导航应指向映射后的目标或明确的基线Git视图；需要本地可点击正文时生成有
独立标记的派生视图，原正文保持字节。验收分开报告历史原文、派生链接和无法
定位的外部/ignored产物引用，不能仅创建索引就宣称全部旧链接可用。

### 5.2 新代码和旧代码分别登记

当前导入和入口改为明确包路径，新运行绑定迁移后的实际源码及迁移层。历史结果
继续引用原声明与原SHA；不能将新实现的路径替换进旧JSON后宣称旧链未变。
变更共同路径helper时，原版本保留在Git及清单中，新版本重新验证。

不保留上千根目录wrapper/symlink；确有外部使用的少量公共API才提供显式、集中、
可测试的过渡兼容。历史入口以索引及明确loader定位，不能自动启动模型或任务。
不使用全局Path/open monkey patch，也不通过伪造`__file__`隐藏实际导入位置。

### 5.3 不搬运行证据和外部环境

`outputs/`中的已完成/失败收据、源码快照、XML、Python环境及runtime manifest
保持原位置。内含绝对路径、canonical目录和no-symlink规则的运行时若以后迁移，
必须有独立派生协议及验证；本次源码整理不授予新runtime/bootstrap/control资格。
所有相关进程必须先实际wait/reap，本次不启动GPU或恢复历史排队任务。

## 6. 执行顺序与提交边界

1. **D0，设计与基线。** 提交本设计、当前状态快照、按Git文件生成的清单/依赖统计。
   记录已有失败；设计落地前不移动算法或入口。
2. **D1，迁移基础。** 实现统一根定位、显式身份locator/迁移清单验证和测试support
   边界；补真实路径/SHA/导入身份反例，不进行全局文件API替换。
3. **D2，首个完整切片。** 迁移最近8个trajectory search家族：8core、23scripts、
   24tests、7config、9plan、14report，共85文件。移动当前导入方，分开算法、独立
   审核和执行入口；确认源码/测试/文档根平铺数真实减少。通过后及时提交。
   85仅为种子文件数；实际变更还须覆盖脚本/测试导入方闭包及新增包/support，
   不为保持该数字遗漏调用方。迁移前核测试节点映射，迁移后核真实import位置，
   防止静默落回同名旧模块。
4. **D3，其余代码。** 按导入依赖闭包分批迁移controllers、references、dynamics、
   evaluation和其他research家族；先处理高扇入脚本里的可复用函数，再搬调用方。
   每批核验清单、导入来源和相应CPU回归，不一次搬走1,489脚本。
5. **D4，文档和配置。** 全部计划/报告/配置按主题归组，原正文/字节保持，导航用
   新索引；新维护文档使用新链接。旧README保存原字节，根README收敛为简明入口。
6. **D5，收尾。** 全量结构/引用/测试发现核验，记录剩余兼容项及历史执行边界，
   提交最终目录导航与迁移报告。此时才判断本次整理是否完成。

首批家族为constraint_residual_repair、interior_trajectory_search、rejected_response_search、
global_envelope_search、stage_budget_search、state_vector_gn_search、
state_vector_objective_scale、state_vector_dual_recovery。字节相同的配置仍保留独立
实验身份。本次不顺便合并算法、调参、缩小验收范围或修正ratio容差。

## 7. 验收标准

- 每个基线跟踪文件有唯一去向或明确保留理由；无丢失、无未说明覆盖。
- 各批清单核SHA；历史声明仍可定位并核原字节，实际新源码另有新指纹。
- 根README能定位总目标、当前未完成项、代码入口、常用命令和历史研究索引。
- public API及本批入口导入真实新位置；共享测试辅助函数使用明确support边界。
- 迁移前后pytest node集合按路径映射比较，避免移动让测试静默消失。
- 本批CPU测试、Ruff/format通过；全仓既有失败单列，不以专项通过代替全量通过。
- 纯数学/完整fake-QP ledger、预算/repeat/异常计费及因果历史回归保持。
- 真实输入dry通过且不产生模型、GPU、响应；需要仿真的验证另登记，不自动补跑。
- 控制/世界模型/fresh性能不会因为目录检查通过而升级。

基础命令仍从仓库执行，使用仓库内解释器，显式空CUDA/LD及单线程。每次pytest
使用独占basetemp。最终维护文档会列出迁移后的准确命令；迁移中不沿用失效路径。

## 8. 迁移前实验状态快照

GPU继续禁用，实际资格仍cursor8/task0/grasp0，总目标未达成。
最新唯一dual-recovery生产122响应、52QP、52拟合已关闭；原完整KKT独立账本通过，
51QP合格。首次checker输入路径缺失失败已保留；隔离check-only v2已实际退出，
全部122响应/36600内步/38065查询复算完成，但fresh_ratio差
`2.137171550842254e-10`超过原`1e-10`，最终complete/passed仍False。
完整数值witness仍0，不能写成“独立复算全部通过”。

待解决项是原严格决策验证及已达标前九点占用误差收益、后三点仍超caps的问题。
整理期间保留这些事实和原阈值，不启动新策略对照；结构就绪后再单独登记处理。
唯一producer SHA：`436fbe20f20cf6678aa329c05e928845b563f76cb0a976c11395e16481aba551`；
v2 checker SHA：`6036f0cd4a711ef44b556b489cf946252d2d30e8dafb8cb232cfef0c4d4fcb85`；
v2 owner SHA：`cc55b1626b3ef7c3667850eb9d80f86fe538e82c030da06e21b4da382dcb60e1`。
