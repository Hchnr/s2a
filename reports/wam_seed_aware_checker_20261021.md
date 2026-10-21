# Seed-aware日志检查器回归 — 2026-10-21

**014检查器的旧证据回归通过。** [新模块](../scripts/wam_position_acceptance_checker.py)中的完整`verify_case`与`verify_query`由冻结006/140f7e8函数显式继承，参数化采样seed及首query来源回调；旧执行器、checker、010审计入口和全部既有输出均未修改。此工作不启动模型或仿真，不打开fresh性能，也不决定后续候选或预算。

新函数要求`sampling_seed`是整数且属于`config.sampling_seeds`，同一值贯穿case result、每次request与response/Action128身份验证。WAM首query必须提供明确来源回调；已有seed42回归沿用冻结006回调，不凭空为新初态或seed43构造旧raw身份。未来正式验收仍需单独冻结fixture、初始物理状态、anchor、当前输入及其回调。新模块没有fresh执行入口。

[独立回归入口](../scripts/verify_wam_acceptance_seed_regression.py)按[014计划](../plans/v0.0.1/014_position_acceptance_audit_20261021.md)读取已经存在的004九条oracle、006十二条WAM及010六条完整WAM日志，共27条、16200个旧端点。每条都重新执行完整机械/几何、标量夹爪、原生结果、raw/effective/admitted/nominal参考与时钟检查，并将新函数的全部返回字段逐值与旧审计结果比较，全部精确相等。已包含010 raw8/bowl的原生稳定成功、其他失败，以及oracle正控。缓存来源/profile等外层审计元数据不属于该函数的返回字段，不替代逐步检查。

最小源码diff和AST证明显示，除显式seed参数/注册及结果验证、向query验证贯穿该seed、首query回调参数与来源guard外，全部原AST节点不变；没有修改机械/几何/时间/成功容差或删除字段检查。旧函数对象与全局保持原样，不使用`exec`、编译文本生成函数或monkeypatch。真实42配置传入43/`True`及query43的负例均被拒绝；没有伪造或改写任何43日志。

回归对5053个输入文件进行SHA前后核验，CPU进程CUDA/LD均为空；新增模型请求和物理步均为0。Ruff及格式检查通过。原case函数源绑定`18a5a54`/SHA`35c7e332f1fdfc6ff2b5e95c0af3311bba67b2815450d75a572c1432526b1119`，query函数源绑定`140f7e8`/SHA`8ab8e29b90f6e0419171e563192c2009ce6220d8b5a9c642699798f6e2f19915`；新模块在`1444f33`提交，SHA为`7a90309b1b2ea2b75429f1f0cecb4e5036aab43281cb66c4214a57de4df2d85f`。

可复核产物：

- [完整27条回归](../outputs/wam_stability_20261021/seed_aware_checker/seed42_regression.json)：`b90d0915f3880fe71ac8d4f3b79afcae18720ca569d270849ab5158712d85f7e`。
- [最小源码diff/AST证明](../outputs/wam_stability_20261021/seed_aware_checker/literal_source_diff.json)：`60082d732c08a1f28b2b61d082c6612059beed931639c25a40c11f287ab65873`。

本份证明仅覆盖真实seed42旧日志上的兼容性。[015另行注册的真实43 oracle执行资格](wam_seed43_execution_qualification_20261021.md)独立记录；它不能验证43的WAM请求/响应分支，也不能计入WAM成功分母。新checker回归通过不替代010完整开发结果或未来真实多初态验收。
