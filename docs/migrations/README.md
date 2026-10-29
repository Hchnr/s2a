# 结构迁移与来源身份

[全局设计](../architecture/repository_design.md)规定顺序及验收。
[基线清单](baseline_20261029.json)只记录迁移前的3276项身份，不预判新路径；
基线commit为`7b2712850dea54090d61e564380f88825ad8b8f6`，清单SHA256为
`1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897`。

## D1：路径及历史字节定位

`s2a.infrastructure.repository_paths.repository_root()`从明确anchor查找项目标志，
不依赖固定目录深度；`repo_path()`校验仓库相对路径。

`s2a.infrastructure.source_identity.SourceIdentityLocator`接收明确root、baseline
commit、catalog_ref、migration_refs及outputs内cache目录。`resolve(declared_ref)`
首先确认原路径/SHA已登记，再返回declared_ref、storage_ref、migration_ref、
catalog_ref、原Git来源及resolution_kind。调用方最后执行`finish()`核验整批。

后续迁移清单绑定相同catalog_ref，moves每项包含：

```json
{
  "old_path": "src/s2a/example.py",
  "original_sha256": "<baseline SHA256>",
  "new_path": "src/s2a/research/example.py",
  "current_sha256": "<current storage SHA256>"
}
```

路径在迁移后改字节时，原身份定位原Git blob，当前源码另有指纹。已登记current
目标漂移会失败，不通过fallback掩盖。未知版本、错误blob、缓存损坏、路径越界
及旧根访问均拒绝；不会自动导入找到的源码。结果scope固定为`byte_identity_only`，
不授予历史代码执行、runtime或控制资格。旧JSON与旧helper目前保持原样。

D1专项36项CPU测试通过（最终session28118实际exit0，35.12s），新文件Ruff及
format检查通过。真实3276项catalog接线、active源码定位及本地Git收尾通过。

## 测试发现基线

`outputs/repository_baseline_collection_20261029/`保存迁移前实际collect-only收据、
日志与nodeids。session17648/child3375668实际wait/reap；收集到4976节点，5项既有
collection错误，退出码1。错误均为CPU环境PyArrow要求NumPy≥2.0，而当前用于
数值回归的解释器为NumPy1.26.4。没有执行测试用例；不能将此记录称为全仓通过。

后续按旧→新文件映射比较测试node集合，并分别报告新增测试和这些既有错误。
收集器中的模型/任务零计数字段没有独立运行时探针；本基线只用作测试发现与
源码未变的证据，不据此授予任何模型或仿真执行资格。

## D2：首批完整领域

[首批清单](trajectory_search_20261029.json)绑定85个实际移动文件和81个原位置修改。
8个研究实现、23个入口/独立审核、24个测试、7份配置、9份计划、14份报告分别位于
各自的 `tracking/trajectory_search` 或 `research/trajectory_search` 目录。配置、计划、
报告30份原字节保持；历史Markdown仍按旧声明位置解释相对链接。

[根文件补充清单](root_files_20261029.json)只登记原 `.gitignore`、`AGENTS.md`、
`README.md`、`pyproject.toml` 四项。它与3276项主清单采用同一基线commit，不扩大
到用户未跟踪文件。当前Python包导入及pytest路径另行记录新源码身份。

最新两个入口使用 `s2a.research.trajectory_search.inputs.SearchInputContext`，分开
历史源码、当前源码/输入和已固定的外部仿真资产。历史源码只通过catalog/locator
证明字节；32个外部mesh/纹理依据已绑定bundle的exact path/SHA只读核验，不作为
可执行外部代码。`finish()`重验已消费来源，未知版本、漂移及旧清理根均拒绝。
迁移未重跑生产或完整checker，也不升级原failed checker的资格。

当前命令从仓库执行，例如：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONPATH=src:scripts:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 \
outputs/libero_env311/bin/python -m \
  scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery --dry
```

两个真实入口dry均通过，QP/响应/查询各0。首次dry对32项外部资产的分类拒绝记录
保留在 `outputs/trajectory_search_migrated_dual_dry_20261029/`；修复后的独占v2记录
在 `outputs/trajectory_search_migrated_dual_dry_v2_20261029/`，owner和两child实际
wait/reap、exit0，175项来源前后一致。其适用范围是当前入口准入，不是重新执行
历史搜索或控制。

全量collect-only得到5096节点：旧4976节点按文件路径映射后完整保留（参数及顺序
保持），新增120；仅保留原5项PyArrow/NumPy错误，没有新增collection错误。
原退出码1与比较通过分开记录在 `outputs/repository_restructure_collection_20261029/`。

统一302项CPU QA最终通过；v2独占收据位于
`outputs/repository_restructure_unified_qa_v2_20261029/result.json`，SHA256
`fabd2a958a242ab5c303e0a9c5b9a2a9c9256dbdf094daf784889ef9759faf7c`。
owner及child实际wait/reap exit0，3317项来源/文档在本轮前后一致。首次统一QA的
302测试通过，但因进行中编辑了本说明导致owner的冻结检查False，原失败收据
保留；v2显式绑定该失败及唯一文档差异，没有修改测试门槛。153个修改/新增Python
文件的Ruff/format均通过。

最终结构及节点检查收据为
`outputs/trajectory_search_d2_final_structure_20261029/result.json`，SHA256
`33e568d2c423f4dc01140bb8ae8e8e6467d7c6a149901602c258cf14fa9f6e26`。
此时根平铺仍为src378/scripts1467/tests442/configs192/plans341/reports365，
`full_restructure_complete=False`。

维护检查入口为 `python -m scripts.maintenance.check_repository_structure`；
`--manifest`显式传迁移清单，`--additional-catalog`显式传根文件补充清单。
它核当前存储SHA、映射后的pytest节点及根平铺数量；首批结构通过不表示全仓整理
完成，更不表示控制或运行时资格通过。

## D3a：库分层与当前执行身份

累计清单 `repository_layout_20261029.json` 完整登记主清单及根补充清单的3280个
历史身份，原声明/SHA与当前路径/SHA分开。`CurrentLayout` 与当前脚本来源核验
只接受显式批准的清单；原输出及历史Git字节不因此取得新的运行资格。

其余377个库模块已按领域移动，物理移动时原字节逐项相同；导入与当前来源读取
单独更新。两处库到scripts的依赖抽取到真实库函数；独立审核仍保留独立实现。
脚本目前已统一scripts.*身份，但大多数物理位置将在D3b整理。

发现收据 `outputs/repository_d3a_collection_20261029/result.json` 中，5169节点保留
D2全部5096节点及其顺序/参数，新增73项；原5项环境错误单列，pytest实际退出1，
比较owner退出0。2337个修改/新增Python文件Ruff及format检查通过。

32进程CPU回归初次674项用时49.12秒，669通过、5项旧路径/导入匹配失败；原收据
`outputs/repository_d3a_cpu_qa_20261029/result.json` 保留，不覆盖成成功。修复只涉及
明确的当前路径与历史字节读取边界，后续复核收据另存。数值QP单元测试实际执行，
不能将该QA称为QP零活动。各进程保持数值库单线程，空CUDA/LD、独立临时目录，
全部实际wait/reap，节点无重复、无遗漏，来源前后一致。

两条当前入口真实dry并行通过，各核验5032条运行引用，QP/响应/查询均0；
`outputs/repository_d3a_latest_dry_20261029/result.json` SHA256为
`9867dd53dedfb0e5654240b2fc099563e5ef76488a3e6d255c372d53739dea57`。
适用范围仍为当前入口准入，未重跑生产、完整checker、模型或GPU任务。

原5项失败随后在新累计清单下独立复核全部通过，5个child实际wait/reap exit0，
2.37秒，3372项来源/文档前后一致；收据
`outputs/repository_d3a_failed_nodes_20261029/result.json`，SHA256
`475bfc2dd691d5e5a6459851c780482e265529a1ac6266a5cc8b7cd10f71c71d`。
另外temporal/readout两个模块并行10+11项通过，其中新增2项分别拒绝错误历史SHA
和用正确历史字节掩盖当前来源错误。没有将修复后的复核称作重新执行了全部674项。

修复后的最终发现为5171节点：5096原节点顺序/参数保持，新增75项；5项原环境
错误仍在。实际pytest退出1与比较owner退出0分别保存于
`outputs/repository_d3a_collection_v2_20261029/result.json`，SHA256
`0dcc9125ace894b6b675179b0ba03c9528b2151cfc75e6f136b2a100fe16a043`。
最终全仓2411个Python文件Ruff/format均通过；结构身份检查通过，脚本/测试/证据
区域仍有平铺文件，`full_restructure_complete=False`，不得提前称作整理完成。

## D3b首批：共享脚本与独立审核

本阶段775个脚本按已发布的领域设计移动，1141个调用方同步明确namespace、
当前源码读取与CLI路径；另新增59个真实包标记，不保留旧平铺wrapper。
最初静态复核1004个非test改动文件：在仅namespace/明确存储定位规范化后，
数学AST保持；775个rename阶段逐项字节保持。4份独立审核仍只使用标准库定位
当前仓库，不以引入生产库或放宽禁止导入断言来消除路径问题。

历史proof继续读取catalog登记的原Git字节并核原SHA，当前源码按累计清单独立
定位和核验。prefix reserve的完整逆AST比较仅多识别一个明确的依赖namespace
迁移，原差分、caps、预算、容差、函数和ledger断言保持。C++源码与原build、
flags、binary完全保持；当前存储与原编译声明分别核验。

全量发现5172节点，严格旧文件/函数/参数/顺序保持；原5项环境错误仍实际退出1。
收据 `outputs/qa/repository_d3b1_collection_v2_20261029/result.json` SHA256
`baf1dde6b0debfdcfebb24151d42cc7db66e455b9700844fb16a519ece5c4abd`。

2254节点的32进程初测2107通过、147失败，来源前后完全一致；首次collection的
新增5处定位错误与该回归失败原记录均保留。修复后完整35模块486节点32进程
复核482通过，4项剩余失败另修后独立复核全部通过，没有声称重新执行全部2254。
最终4项收据 `outputs/qa/repository_d3b1_final_four_20261029/result.json` SHA256
`93c2fdb8fb91482e4418cc2d37bdae078485b0d1856b0d2e679b173be8ca33dc`。
所有owner/child实际wait/reap，未发送信号、未自动重试；未运行GPU或新研究实验。

3.11数值解释器没有pidfd接口，pidfd专测显式使用仓库3.12解释器运行原真实
父子流程，原3.11 fallback测试保留。短命child测试等待真实退出再核reap，避免
并发启动延迟使固定sleep误判；完整生命周期门保持。

最终全仓2470个Python文件Ruff/format通过，结构/导入边界另行核验。剩余690
script、439test、898raw证据与README整理未执行，不能称为全仓整理完工或新控制
资格通过。本阶段提交后仍按已落盘设计推进。

本次最终结构/身份检查 `outputs/repository_d3b1_structure_20261029/result.json` 通过，
1237原文件已移动；根平铺为src1/scripts692/tests442/configs192/plans341/reports365，
完整整理标志仍False。静态包依赖检查2470个Python文件、10903条本地导入边通过，
无未解析的明确本地导入或裸脚本导入；基础层不导入research，库不导入scripts/tests。
该静态结果不包含动态导入/运行资格，收据位于 `outputs/repository_d3b1_module_graph_20261029/result.json`。
双语PR原文见[本阶段PR说明](../pull_requests/repository_layout_20261029.md)。

## D3b收尾：其余实验入口

按既定逐文件设计移动其余690个脚本，57个测试调用方更新明确包路径，新增17个
包标记；scripts根目录只保留README和包入口。685个rename相对上次提交原字节
保持，其余5个只更新当前ENTRY/WORKER或当前AST源码定位。历史注册、原始产物、
算法、容差、预算和控制门槛保持；未留下旧路径wrapper或符号链接。

完整累计清单登记3280个历史身份、1927个实际移动，当前批准SHA256为
`d8d808a6cc641fae55940f03e92c97827bf0d2dc29a57271d9ff3fdd22bc055b`。
当前身份与原声明/SHA分开核验。全量发现仍5172节点，旧节点/参数/顺序保持，
数值环境仍有原5项PyArrow/NumPy collection错误，实际退出1。

32进程816项初测781通过、34失败、1项按原条件跳过。修复后11个完整模块148项
复核139通过、9失败；9项再次复核8通过、1失败；最后1项修复后独立通过。
这些是分轮结果，不是最终版本重跑全部816项；所有owner/child实际wait/reap，
源码/文档各轮前后相同，原失败记录保持。收据如下：

| 范围 | 收据 | SHA256 |
| --- | --- | --- |
| 816项初测 | `outputs/qa/repository_d3b_complete_20261029/result.json` | `674996b54b07ce2eb89c9fb958e9c49d197a8df69fe187636d95608717856838` |
| 5172节点发现 | `outputs/qa/repository_d3b_complete_collection_20261029/result.json` | `afce065e56579b092ea2b22936212e92c9aee7876f7113a0d1f17736a29875e3` |
| 148项相关模块 | `outputs/qa/repository_d3b_complete_repair_20261029/result.json` | `071f95d99115b3a6f3d34e91e58d40d89e6dded1ea1df0691b146293d9858d18` |
| 9项复核 | `outputs/qa/repository_d3b_complete_final_nine_20261029/result.json` | `1a562594afcaeebf543788333c6361ae810e27ec99e8d0466a8b4bab2c4e29c1` |
| 最后1项复核 | `outputs/qa/repository_d3b_complete_final_one_20261029/result.json` | `44e6cf009a8a5740bb2f6c9e7c2295850a1534ba652b556e7cae1097af4575ba` |
| 兼容Arrow环境14项 | `outputs/qa/repository_d3b_arrow_environment_20261029/result.json` | `9b8f8fb4596608842a107ae169d8ef4a8411d28e010ea79e6f8c77b36aa09c84` |

历史CLI帮助和发布桥接使用catalog/Git原字节的隔离子进程，保留原断言、原__file__
和导入身份；显式禁止数组解码、模型及当前源码混入，只回归历史metadata合同。
外部只读config缺失的测试使用已登记迁移位置的相同SHA字节，并继续拒绝错误SHA
与旧仓库路径；原preflight字节保持，不追认旧运行资格。

PyArrow五模块在仓库现有Python3.12/NumPy2/PyArrow23解释器下实际14项全部通过；
该记录与3.11数值环境collection错误分开，不修改共享环境，不称完整pytest通过。
全仓2487个Python文件Ruff/format通过；静态包依赖检查10918条本地导入边通过。
结构/身份检查通过，根平铺为src1/scripts2/tests442/configs192/plans341/reports365，
`full_restructure_complete=False`。静态与结构收据分别为
`outputs/repository_d3b_complete_module_graph_20261029/result.json` 和
`outputs/repository_d3b_complete_structure_20261029/result.json`。

439个测试文件、898份配置/计划/报告、README导航及最终全量CPU验收仍待执行。
本阶段未运行GPU、新生产搜索或完整研究checker；实际控制资格和总目标保持未达成。
双语PR原文见[脚本迁移收尾PR](../pull_requests/scripts_layout_20261029.md)。

## D3c：测试领域包与分轮回归

已实际迁移439个测试文件，原fixture不移动；新增71个包标记，测试根保留
README、__init__、原字节conftest。原测试函数参数、装饰器及断言在明确namespace
归一化后保持，当前fixture/peer读路径显式定位。pytest的testpaths、not-integration
scope及addopts保持，仅移除旧scripts搜索路径。标准库并发入口为
`python -m scripts.maintenance.run_cpu_tests --jobs 32 --output outputs/qa/<新的独占目录>`；
全量使用64进程，每个子进程数值库单线程、空CUDA/LD、独立临时目录、实际回收。

初次受影响88个完整模块1205项回归1141通过、64失败，收据
`outputs/qa/repository_d3c_modified_modules_20261029/result.json`。全量发现5172节点
完整保留，仍只有原5项PyArrow/NumPy环境错误。随后的来源修复分开原Git字节、
原声明/SHA、当前文件与批准SHA，保留完整修复逆AST及原数值/安全/预算门，新增
5项拒绝原生运行时计数/import/cleanup等篡改的测试；节点预期扩展为5177。

首次64进程全量实际报告4987通过、188失败、2跳过，原5项collection错误仍在；
原5172节点与5项新增按明确预期顺序收集。该轮因合成fixture写坏真实config导致
`source_unchanged=False`，不得作为有效全量通过或来源冻结证明。owner和64个
worker均实际wait/reap；收据
`outputs/qa/repository_d3c_full_cpu_20261029/result.json`，SHA256
`50b5e322cfc3084752b72360e235b80a92589c546b0660ace3c7357e6bdf9cb2`。

原因是path-preparation测试替身仍登记旧`s2a.native_runtime_path_snapshot`，未接管
新canonical模块，导致合成fixture的CONFIG指向真实仓库。所有进程关闭后，从HEAD
精确恢复`configs/native_runtime_path_snapshot_20261029.json`，恢复SHA256为
`f41012908ebaf24f67763900b71f6f818c97cddf398ff67f82323d405d243aac`，与原catalog
相同。损坏字节及恢复收据独立保存于
`outputs/repository_d3c_full_source_recovery_20261029/`。原始登记及失败状态没有改写；
完整3280项与本轮3522项来源检查只发现该config，恢复后无其他漂移。替身已改为
准确canonical身份，文件构造前显式核临时根边界，不能靠重新登记合成SHA通过。

随后修复当前loader/历史gold来源证明并分别核验数值环境差异，收尾见下文；
raw配置/计划/报告尚未移动，最终D5全量CPU与真实dry仍待执行。

四项streaming数值失败另有迁移前原字节的独立环境诊断：原test和两原数学库在
outputs隔离目录执行，native3.11实际22通过、4失败，失败节点及最大误差逐项与
full64相同；当前开发3.12环境的相同26项已观察全部通过。原数学AST仅import
namespace不同，readout模块原字节保持；不修改算法、原断言或容差。两解释器的
Python/NumPy/OpenBLAS版本均不同，不能只归因NumPy或单个组件。隔离诊断的
owner实际exit0，测试child实际exit1，全部回收、来源前后相同；scope为既有合成
数值差异归因，不是把原失败改成成功。收据为
`outputs/wam_clean_streaming_baseline_native_diagnostic_20261029/result.json`，SHA256
`7a9b163928d3240279aba41ef314e4681ad0eb475b8252583c8a5620f1be20be`。

D3c修复后的64进程全量5177项，5169通过、6失败、2跳过；节点/参数/顺序和分片
集合exact，原5项Arrow collection错误保持，源码/文档前后一致。6项失败中4项为
上述原数值差异，另2项为GN审核的namespace断言和readout启动观察测试的调度
竞态。只修改最后两个测试：前者通过明确old identity定位canonical module，后者
使用真实JSONDecodeError ACK与真实identity文件握手，保留同一child及原退出/
回收断言，不增加固定sleep窗口或mock生命周期。两个完整模块51项32并发全部
通过；没有在最终版本重跑全部5177项。兼容3.12环境另16并发实际40项全部通过，
包括原5个Arrow模块14项及streaming26项；该证据与数值环境限制分别保存。

| 范围 | 收据 | SHA256 |
| --- | --- | --- |
| 修复后全量5177项 | `outputs/qa/repository_d3c_full_cpu_v2_20261029/result.json` | `f1514d09e53538a4ec50e368752b13da2e5b91fedc5a508f82c373c6c7911bef` |
| 兼容环境40项 | `outputs/qa/repository_d3c_compatible_environment_v2_20261029/result.json` | `459485504a0652379682a22dd87ee5181b912cd864f68314fb5c9121e6e48828` |
| 最后两模块51项 | `outputs/qa/repository_d3c_final_two_modules_v2_20261029/result.json` | `4a4c4a754659b67022bb08697b2a963f5a75276ff403d1c836aa0acfabfcbd6a` |

所有owner及worker实际wait/reap，未自动重试已执行测试。一次末轮nodefile误用
metadata对象在启动pytest前被拒，零child/零case，失败说明和正确list的新独占
记录分开保留，不伪称首轮已执行或成功。当前累计清单SHA256为
`e3828c550f2d3824d47d78705db78cdc7902fb99da15d1ef2c87017fc912ca02`。

历史来源修复保留原SHA与完整AST：原生bootstrap证明、readout helper链及long
qualification原CPU_REVIEW/Git来源均分开核原字节和当前批准源码。新增代码只处理
明确namespace/存储及身份元数据，原数学、证据门、常量、预算和失败状态保持。
旧Cosmos serving源码的静态locals测试使用011原登记绑定的精确归档；原prepared、
原declared source和snapshot SHA均核验，不用已变化的当前Cosmos树冒充历史版本，
不访问旧仓库或授予当前模型运行资格。

当前全仓2559个Python文件Ruff/format通过；静态依赖10970条边检查通过。
结构检查3280个历史身份、2366个实际移动通过，原节点全部保留；测试根3项。
此处结构收据在最后两测试修复前为
`outputs/repository_d3c_final_v2_structure_20261029/result.json`，SHA256
`8e2aa0dd30ed4a19d181a3798fdc1153febadcff6d175a3a3f73462b50f6741d`；最后两修复后
累计清单及51项来源冻结独立绑定当前字节，不冒称该旧收据检查过新版本。
`full_restructure_complete=False`；898份raw文件、导航及最终D5尚未执行。
