# CPU bootstrap v2：隔离修复资产路径规范化

2026-10-29，承接[首轮实际失败](../../reports/native_runtime_bootstrap_20261029.md)。旧17份QA源、配置、计划、快照及失败记录保持原字节；不覆盖旧输出。原运行时导入成功，但首次构建前资产守卫拒绝 `assets/scenes/../textures/light-gray-floor-tile.png`，实际模型成功、context、forward、物理步、render均为0。部分环境尚无sim，旧close另报AttributeError；原始路径错误及清理错误均保留。

新[配置](../../configs/native_runtime_bootstrap_v2_20261029.json)绑定旧配置、QA、实际准入、失败CLI及owner。除新独占输出目录和repair_parent引用外，所有旧配置字段严格相同。先复核原失败来源、实际wait/reap1、无signal/重试及构建前计数；不得把失败改记为通过。新执行是明确修复后的单独验证，无自动重试。

## 有限修复

新core复用旧函数，在私有globals内仅替换资产记录检查和部分失败清理；用旧SHA和AST逆比较确认其它流程不变。不修改旧模块globals、快照或venv。新CLI同样只对版本路径、core导入及父失败准入作有限隔离，保存修复证明。

资产专用检查允许已登记根内的绝对路径包含正常 `..`：逐组件检查原声明路径，每次退回仍须在同一批准asset根内；若逃出后再进入也拒绝。拒绝symlink、相对路径和未知根。原完整XML字节不改，声明路径、规范路径和实际文件SHA均记录；输出目录、源码、其他输入路径的旧守卫保持严格。此项只适配实际冻结包生成的资产路径，不修改机器人、几何、相机、控制参数或允许根。

仅在构造失败且部分环境尚无sim/viewer时明确记录 `no_resources_created`、`cleanup_close_called=False`、`cleanup_complete=False`，不调用该对象的close，也不将其标记为成功关闭。该标签只表示对象未持有sim/viewer，不证明native内部从未短暂分配资源。若对象持有资源，沿用一次close及独立清理错误记录。原bootstrap异常继续保留。

## 原预算与验证边界

沿用[首轮协议](native_runtime_bootstrap_20261029.md)的固定任务/seed42/trial0、完整原初态、20Hz/2ms及固定OSC合同、3次模型/3次context、2次非积分step1、forward上限32、reset1和单帧256RGB软件OSMesa验证。model动作/物理积分/参考消费/world-model/fresh均为0；GPU禁用，CUDA和LD显式空，线程变量为1。600秒child上限，无覆盖或预算追加。

统一QA绑定原17份来源与新增core/CLI/两测试/本计划/配置共23份，执行原5加新2共7个测试模块；故障与路径测试必须验证真实边界，不增加占位测试。随后Ruff/format和实际CLI dry通过，才在 `outputs/native_runtime_bootstrap_v2_20261029` 执行一次。QA/owner记录实际退出、来源SHA和失败阶段；结果全部保留，成功须重新核对加载来源及资产。

即使v2加载与初态渲染成功，也只取得当前运行时bootstrap资格，不证明历史二进制等价或轨迹控制成功。原完整H12仍无witness，实际仍cursor8、task/grasp0；原18正确参考/12WAM开发/24最终留出门槛保持，物理pilot不由此启动。
