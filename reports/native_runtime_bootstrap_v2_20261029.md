# CPU bootstrap v2：资产路径兼容修复

2026-10-29：隔离实现已提交`18b8f98`。新增core与CLI组件分别通过16项和12项native CPU合成测试，Ruff检查通过；统一QA已在原会话56093实际exit0关闭，**195 passed / 1 skipped，150.89秒**，绑定23份来源与7个测试模块。实际准入原会话46348已exit0，3个子进程均正常wait/reap。唯一v2 bootstrap已在原会话16687以exit1关闭：**constructor与hard_reset完成，record XML前置守卫失败**。已成功构建2个模型，但原79维初态恢复、控制器及显式渲染验收均未到达；`runtime_loaded=True`，`bootstrap_verified/control_qualified=False`。

本轮遵循[新计划](../plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md)及[固定配置](../configs/native_runtime_bootstrap_v2_20261029.json)。[v1失败](native_runtime_bootstrap_20261029.md)保留原字节：合法资产引用`assets/scenes/../textures/light-gray-floor-tile.png`被旧守卫拒绝，configure为1/1、model_construct为1/0，其余实际调用均0；局部对象无sim导致的清理错误亦保留。v1的`runtime_loaded=True`不等于bootstrap成功。

## 有限变化与不变项

资产检查只在批准根内逐组件规约绝对路径：跨出根后返回仍拒绝，遇symlink、缺失组件或非目录中间组件，即使后续`..`会消去它也拒绝。原XML字节不改；每个资产另记声明路径、规范路径和SHA，输出目录及其他源码/输入路径仍使用旧严格守卫。见[路径规约](../src/s2a/native_runtime_bootstrap_v2.py#L25)与[独立资产函数globals](../src/s2a/native_runtime_bootstrap_v2.py#L58)。

执行函数在独立namespace中编译，不修改旧模块globals。旧源SHA绑定及逆AST比较只允许把原cleanup的close/成功赋值替换成新局部清理helper，其余运行流程保持；见[逆比较](../src/s2a/native_runtime_bootstrap_v2.py#L87)和[独立调用](../src/s2a/native_runtime_bootstrap_v2.py#L137)。CLI同样只隔离版本路径、core导入、修复证明及父失败准入；[父失败检查](../scripts/verify_native_runtime_bootstrap_v2.py#L237)要求旧QA、实际准入、失败receipt与owner退出证据完整绑定。

部分构造失败且对象未持有sim/viewer时，新helper记录`cleanup_outcome=no_resources_created`、`cleanup_close_called=False`、`cleanup_complete=False`，不调用不存在资源的close，不将原失败记为成功。此标签仅描述该对象可见的资源状态，不能证明native内部从未短暂分配资源。已有资源仍只尝试一次close，原异常保持。见[清理helper](../src/s2a/native_runtime_bootstrap_v2.py#L70)。

原17份QA来源与失败产物冻结；新增6份来源构成23份新QA名单。新配置除独占输出目录和`repair_parent`外，必须还原为旧配置。任务、seed42/trial0、79维初态、固定OSC、20Hz/2ms、预算和验收条件均不改变；3模型/3context、2次非积分step1、forward上限32、一次reset和一帧软件OSMesa渲染仍需实际记录。control step/mj_step/mj_step2预算为0，子进程限600秒，无覆盖或自动重试。

## 当前证据边界

组件测试覆盖原XML逐值保留、根内规约、跨根/symlink/无效组件拒绝、canonical资产SHA、旧globals和输出守卫保持、部分清理及原控制/渲染门。统一QA子进程PID1008791，`complete/passed/actual_wait/reaped=True`、`returncode=0`，无signal/retry；23份来源及driver保持原SHA，原17份来源与v1 QA一致。已通过新CLI的QA读取器重新核验这些绑定。测试不产生真实runtime/model响应。

唯一skip仍是旧`tests/test_native_environment.py`所需历史import-only probe fixture未安装；不能计为执行通过，本次真实cv2准入已另外完成并通过。

实际准入3个子进程PID1019890/1019989/1020085均`actual_wait/reaped=True`、`returncode=0`，无signal/retry；27份owner登记来源及3份日志SHA重核一致，绑定同一QA及执行driver。CLI dry核对2686个引用，未加载runtime，新增环境/模型/物理响应均0。

| 当前冻结来源 | SHA256 |
| --- | --- |
| [新core](../src/s2a/native_runtime_bootstrap_v2.py) | `5f1d7da14c592383eac171dd3079dabacef777b5f3cef42c36336636b7dfa125` |
| [新CLI](../scripts/verify_native_runtime_bootstrap_v2.py) | `c1e834876879a96442fd45de89948c9cfba0917ce12e27a01502ababc369c956` |
| [新计划](../plans/v0.0.1/native_runtime_bootstrap_v2_20261029.md) | `cfceec0fa44e13aa0a4c93087f28183b53ed184b1522a69faf54556f34fc1535` |
| [新配置](../configs/native_runtime_bootstrap_v2_20261029.json) | `b37fda536c0739aaa659a3392a3260c2d1a4949e62a6d7dc332d789c9b7b15ef` |
| [统一QA receipt](../outputs/native_runtime_bootstrap_v2_qa_20261029_jbrxlsyy/result.json) | `49808224572128be90e07317cba22620574b10efb1552a093ab0c2de55552916` |
| [QA日志](../outputs/native_runtime_bootstrap_v2_qa_20261029_jbrxlsyy/pytest.log) | `241b35e9b2ceb07ce458c538958f53ca678d0eade8850a3305d2481a3666392b` |
| [QA driver](../outputs/native_runtime_bootstrap_v2_qa_driver_20261029.py) | `ae4b6bface80167fdbde7100469d91b41e4623fa8fe7b82450cbd66e5431f0b4` |
| [准入/执行driver](../outputs/native_runtime_bootstrap_v2_execution_driver_20261029.py) | `a35294abb899644a1277f34c79e482f259b53fff9326bf568b029a631b6df717` |
| [实际准入receipt](../outputs/native_runtime_bootstrap_v2_admission_20261029/result.json) | `7b044b5ba9f0b155d7b81406745eee1c8240919fb21df674e19d7122cc67faf4` |
| [v2失败receipt](../outputs/native_runtime_bootstrap_v2_20261029/result.json) | `91265cadcbe8c1e9f9a1b4a44a16ace135ba91c938e36f8998e8b4e2d2211873` |
| [v2失败owner](../outputs/native_runtime_bootstrap_v2_execution_20261029/result.json) | `8e1e94c83b4427bf1feefbf9e5ccef1c93dc3ae89de68c380870df9f3d66207b` |
| [v2实际日志](../outputs/native_runtime_bootstrap_v2_execution_20261029/00.log) | `6e889ccc1ba292137796f8a4bc93e8ec0f99db51ba25fcbfaa5971e8df92f802` |

## 实际失败与计数

子进程PID1024309、父PID1024259，owner实际wait/reap1，无signal或自动retry；CLI和owner的`complete/passed=False`。失败阶段为`record_xml`，原错误`ValueError: invalid declared XML path`及`BootstrapFailure`链均保留。CLI登记2753个来源、`source_unchanged=True`；本次报告复核已逐一重新hash，全部匹配。

| 实际操作 | attempted / completed |
| --- | --- |
| configure | 1 / 1 |
| model_construct、context | 各2 / 2 |
| forward | 9 / 9 |
| step1 | 2 / 2 |
| mj_resetData、render、read_pixels | 全部0 / 0 |
| control_step、mj_step、mj_step2 | 全部0 / 0 |

constructor和一次底层`env.env.reset()`均完成；表中的`mj_resetData`是低层计数，不表示高层hard_reset未执行。9次forward与2次step1的11条时间观察全部为0→0，无积分物理步。两个已构建模型各登记99个file属性，保存XML的SHA均为`4d9058e4c989b9628bf34730284c1c9088ce9eec9d775a8ce93e48aace5b1674`。随后outer cleanup实际调用一次并成功关闭，`cleanup_outcome=closed`、`cleanup_complete=True`，没有cleanup_error。

原record的99个file属性中，有4处历史`/Users/.../assets/scenes/../textures/...`引用（3个唯一文件，其中木桌纹理重复）。v2修复了模型编译入口的资产检查，却按有限AST合同保留了另一处[record XML前置守卫](../src/s2a/native_runtime_bootstrap.py#L506)；它在历史路径重映射前无条件拒绝`..`。先前检查遗漏了这4处声明，本次失败由该守卫触发，第三个模型未开始构建，不能归因为MuJoCo编译失败。

后续v3已安排仅隔离修复这一处record路径前置检查，仍需新登记、QA和实际验证。本轮23份执行来源及v1/v2失败产物均冻结，不原地修复或自动重跑。当前仍没有bootstrap资格；即使后续通过，也只证明指定当前运行时的初态恢复与渲染，不证明历史二进制等价、H12轨迹控制或任务成功。GPU继续禁用，原控制门槛、物理pilot前置和总目标保持。
