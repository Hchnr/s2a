# 历史求解器来源桥接（2026-11-16）

历史 staged / stopped / two-step / guarded / margin 与 task-2 shared-shift
证据的原登记保留原路径和 SHA。以前这些读取入口把原登记的源码引用直接与当前
文件逐项比较；后续修改源码后，旧输入虽然仍一致，旧结果因源码字节不同而拒绝
准入。这里恢复的是历史证据的可溯源读取，不执行历史源码，也不改变旧实验结果。

[显式版本清单](../../configs/tracking/joint_control/historical_solver_source_bridge_20261116.json)
SHA256 为 `76e99de6807e020a2bc8a8d68991b86e1358c91247dfee733dcee5c3dcaed8b5`。
清单逐项固定原路径、历史 SHA、包含原字节的 Git commit 与当前 SHA；旧 staged
求解器定位至 `1dbf7c2a56027b4a22ddcac695182cc2a5164392`，旧 margin 求解器及
测试定位至 `9110f91133a54a96deea87fb2348137f7b542afd`，本次修改前的来源
读取器定位至 `5ca691fb258e934114c54b6436d14181aa19604b`。每次核验均从本
仓库 Git 读取历史 blob 并重算 SHA，另外核对当前文件，未登记变化一律拒绝。

[桥接实现](../../src/s2a/infrastructure/historical_source_bridge.py)对原 source
refs 的路径、顺序、数量和所有未修改字节保持严格匹配；新增的桥接实现与清单
随新运行的 source refs 一起登记。嵌套引用可以重复完整的两文件组，次数必须
成对。历史登记的输入、源码引用和输出保持原值；新读取器另返回桥接证明。若未来
任一受桥接的当前文件改变，须更新并审核明确的当前 SHA 与固定清单，不可静默
复用本桥接。checker 不因桥接加载候选或仿真模块。

本记录只覆盖上述实验链中清单列出的身份差异；其他历史链仍须独立核验，不能
据此授予控制、物理执行或最终验收资格。原 QA 失败和修复后的验证收据分别保留。
