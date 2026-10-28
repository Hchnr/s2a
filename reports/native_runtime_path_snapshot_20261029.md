# 原 task0 运行时文件快照完成记录（2026-10-29）

文件准备已正常关闭：复制1084个源文件、451330154字节，保存原manifest/config并生成仅修改路径的派生配置；45份已登记LIBERO源码及120份已登记robosuite源码均匹配历史SHA。**当前只取得文件快照验证，未加载运行时、构建bootstrap环境或执行物理周期。** 文件准备时仍在运行的[固定前缀优化余量完整独立复算](prefix_optimization_reserve_20261029.md)现已正常关闭，确认完整H12 witness为0；实际仍cursor8、task/grasp0，GPU继续禁用，原总目标及18/12/24门槛保持。

## 执行范围

依据[执行前计划](../plans/v0.0.1/native_runtime_path_snapshot_20261029.md#L1)及[固定配置](../configs/native_runtime_path_snapshot_20261029.json#L1)，从安全只读根复制`LIBERO/libero`完整非缓存树、LICENSE和单个OSMesa库至`outputs/native_runtime_path_snapshot_20261029`。其中包文件1082个、LICENSE与OSMesa各1个；保留原双层namespace布局及包内init_files，不读取外部训练dataset。

原manifest和config字节副本完整保留。派生manifest只替换`runtime_config`内`libero_root/libero_config/osmesa_library`三项；派生config只替换`benchmark_root/bddl_files/init_states/assets/datasets`五项。seed、任务、初态和其他参数不改，旧路径继续作为历史provenance保存，未读取、恢复或写入旧`/share/project`目录。datasets指向仓库快照内的空`data`目录，明确`populated=False/source_read=False`。

实现以源/目标SHA、复制大小、完整目录和文件inventory、精确路径映射及反向对象比较验证快照，拒绝附加文件、遗漏、symlink、越界、覆盖和预算超限。源文件与含元数据的完整输出分别限制为10000文件/1GiB。参见[源检查:289](../src/s2a/native_runtime_path_snapshot.py#L289)、[独占准备:411](../src/s2a/native_runtime_path_snapshot.py#L411)、[完整文件验证:436](../src/s2a/native_runtime_path_snapshot.py#L436)。

## 已完成验证与退出记录

正式Python3.11.15下105项合成文件/CLI测试通过（138.24秒），未加载仿真或模型。用例覆盖历史源码与当前依赖绑定、三加五路径修改、文件和元数据变更、重新计算SHA后的非法非路径修改、完整inventory、路径边界、源在复制中变化、预算、独占失败及QA来源。QA绑定最终六个源码/测试/计划/配置文件；[CLI准入:104](../scripts/prepare_native_runtime_path_snapshot.py#L104)要求实际wait0、空CUDA/LD、固定解释器与命令、日志和来源SHA一致。

随后Ruff check、Ruff format --check、实际CLI dry三条命令全部wait/reap0。唯一一次实际文件准备也wait/reap0；子进程PID3977931/start_ticks615593293，无signal或自动重试，来源闭包保持。所有命令从本仓库执行，CPU环境显式空CUDA/LD。实现已在`9af86d2`提交；执行后没有修改其六份冻结来源。

| 证据 | 路径 | SHA256 |
| --- | --- | --- |
| 105项native QA | `outputs/native_runtime_path_snapshot_qa_20261029_seubljbq/result.json` | `3fceccd0854cf9c72d07291d9d9a7b4e88900bf0a6be20f2abed5c3489d5a7e4` |
| Ruff/format/dry准入owner | `outputs/native_runtime_path_snapshot_admission_20261029/result.json` | `a393fc63c9796ffbd2a31440139473b964afa4cfa975a2c42a50b3fa345beeae` |
| 实际准备owner | `outputs/native_runtime_path_snapshot_execution_20261029/result.json` | `bd43f48eb6517238e96d1716eeebfe17ec3b916983168795f9477ce8f62603c2` |
| CLI准备结果 | `outputs/native_runtime_path_snapshot_preparation_20261029/result.json` | `2f616deadc0cc4d73f397b919cbb6eb813f3731bb286a6cc42d4c564cfb4882e` |
| 最终快照manifest | `outputs/native_runtime_path_snapshot_20261029/manifest.json` | `a9ffc0fc6b883f00676d8211958cf96b752b0f977d72a737fcca17fde6b742cd` |
| 捕获的源inventory | `outputs/native_runtime_path_snapshot_20261029/source_inventory.json` | `3ffd1678eefb3d58c37bcbe18772ece6fd39f95cdad823015bb73720c3aabd3a` |
| 派生runtime manifest | `outputs/native_runtime_path_snapshot_20261029/runtime_manifest.json` | `ec8c377eb9fe505d69745f55e33b4ccb6f352fdf6d2e49555d6146c39cb111f5` |
| 派生config | `outputs/native_runtime_path_snapshot_20261029/config/config.yaml` | `9473aa7ab8e2a487f5acd5965f5551ae3ab01555f22eb276b42f80795e618f19` |

执行owner的[exit](../outputs/native_runtime_path_snapshot_execution_20261029/00/exit.json#L1) SHA为`c7d849a7cc03fdcc943e035408c4c94d0f1a866278c5227571199a0bac93d3ab`，[release](../outputs/native_runtime_path_snapshot_execution_20261029/00/ownership/release.json#L1) SHA为`7e353a659261f6e873da2d8706af99a1896c05816c4a65a8703e6b136e10a1f4`。owner driver SHA为`680d451019638b83d80f90c1b850a298f514f8b1bc3dbd54ca294aefc499dd33`。

## 独立文件核验及保留的核验失败

另行使用纯stdlib、未导入生产模块的独立driver重新枚举源与输出，逐一比较文件SHA和大小，并严格反向核对三加五路径修改、原件字节及空dataset。隔离v2唯一执行已实际wait/reap0，无signal或自动重试；**完整输出1090文件、454957103字节、247目录（不含根）**，包含上述1084源文件及6份元数据（3626949字节），仍在原文件/字节预算内。2324份唯一引用在检查结束时重读不变，45份LIBERO与120份robosuite历史源码匹配，20个当前依赖的身份标签保持。快照及原输入全程只读，只在独立检查目录保存driver和收据。

首次独立检查实际wait/reap1，在逐文件循环前停止：检查器在缓存目录截断，漏记45个后代文件的排除条目。只读复核确认归档与当前排除记录均为56项（11目录、45文件），没有新增或缺失。失败结果`outputs/native_runtime_path_snapshot_crosscheck_20261029/result.json` SHA为`544ff16d57c3f5e6d812a42d93f106e8368ba3d26c7100b4fef494cf42f0141b`，退出记录SHA为`43a8fbb2c807e78bfebab6e0793d53f503d20d645c6c58aa62603a8018efdba4`，均保留。此失败来自独立检查器的清单粒度，不能据此称原快照损坏，也不能把首轮记为通过。

新目录`outputs/native_runtime_path_snapshot_crosscheck_v2_20261029`仅修复缓存遍历：继承排除标志，递归stat并记录后代，不读取缓存内容；四类工具缓存名称按原声明处理。AST差异仅为`enumerate_tree`。修复记录`repair.json` SHA为`9125feb09baafe65bee29c881aed7dd539da5647919c9706f936fc44fb7d19c3`，绑定旧失败与两份driver；生产代码、快照和旧检查记录未修改。

| 独立v2证据 | SHA256 |
| --- | --- |
| [result.json](../outputs/native_runtime_path_snapshot_crosscheck_v2_20261029/result.json#L1) | `abfce6a762de30268123ca92da909441f31e2767716a2ca69ba1e221ca545100` |
| [execution.json](../outputs/native_runtime_path_snapshot_crosscheck_v2_20261029/execution.json#L1) | `29f59a8fce9977af01fa5e80ea61f1eb0906e1f164a1919cb40d64de095ab650` |
| [driver.py](../outputs/native_runtime_path_snapshot_crosscheck_v2_20261029/driver.py#L1) | `52a3acd483bc495be5eb68cb1d3f14ad59bcd47977381f4464e182824e0ef40d` |

## 仍未完成的资格

所有结果仅标记`file_snapshot_verified=True`，`runtime_loaded/bootstrap_verified/control_qualified=False`。新增数值响应、物理周期、世界模型调用和fresh案例均为0。复制了文件，不代表后续Python导入、默认控制配置或动态库解析实际使用这些文件。

任务BDDL、默认OSC JSON、未登记包文件及18个系统库仍只有当前SHA；不能追认为原物理运行时的历史身份。OSC JSON保留原venv引用，18个系统库仅只读绑定，未复制、预加载或修改RPATH/LD。包中非本任务XML的旧绝对路径亦未改动，不能推广为其他任务或全部LIBERO的资产资格。详见[原只读调查](native_runtime_path_followup_20261029.md#L54)。

下一步仍需独立的实际加载/bootstrap验证方案；本次未执行。物理pilot还必须等待原actual368/H12完整模型witness、全部响应独立复算和owner正常关闭全部通过。当前无该witness，不启动物理pilot，不降低控制门槛或重启GPU任务。
