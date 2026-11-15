# wine step104 身份诊断的输出路径失败

2026-11-16。按[诊断登记](../../../plans/v0.0.1/tracking/references/wine_capture_identity_diagnostic_20261116.md)，
producer 重算原动作，checker 仅重放原动作；两条都执行到 step104 图像采集，
但在保存完整 capture 元数据时失败。本报告不改变此前六例对照中 wine / seed42
的失败，也不把本次诊断记为通过。

新增入口把 `capture104.json` 写到 worker 目录，而受控仿真运行时只允许写入
其专属 `runtime/` 子目录。运行时目录保护正确拒绝了这次写入；这是诊断接线的
路径错误。两条路径均因此抛出 `PrefixReplayFailure: path outside approved roots`，
尚未调用 step104 的字段比较，也没有得到完整实际 capture 元数据。应将记录
放入受控 runtime 目录，并用单独登记的新输出和预算验证；不能放宽目录保护、
覆盖本次结果或自动重试。

两条各完成 104 控制周期 / 2,600 个 2 ms 内步，共 208 / 5,200；各构造 3 个
仿真模型及 3 个 OSMesa context，均清理完成。producer 新选动作 104 个，checker
新选 0 个；两条均未派发 step104 动作、未发出新 WAM 请求，也未使用 GPU 或
fresh 初态。两个 worker 均被实际 wait/reap。owner 正常收尾但 `passed=false`，
独立的完整轨迹对比和字段诊断未完成。

两条保存的 step104 PNG SHA256 均为
`a09617202d6000f7e263144b4b618e78caf935499b77d4b9d069122fcd042643`，
最后完成端点的状态 SHA256 均为
`107d4d878bab324af3912f173e24fdb1df2c28bb423181c5d1b47f20b539429a`，
与此前对应记录一致。这仍不能确定最初身份比较中实际不同的字段。

零物理预检和 64 并发相关 CPU 回归 134 项通过、0 skip，但此前单元测试只模拟
保存回调，没有在原运行时目录保护下执行真实持久化，因此未发现本次错误。
修正后须补充覆盖受控路径的测试，并使用新的完整回归与预检收据。

| 本仓库 `outputs/` 相对路径 | SHA256 |
| --- | --- |
| `wine_capture_identity_diagnostic_20261116/result.json` | `09f8cb77d3379fdfb0f94758f6dfa3e3b72c24ae8a055b80a635e021b057c915` |
| `wine_capture_identity_diagnostic_20261116/producer/worker/result.json` | `6668e1e6f1581cbf52e1a37c52904a2a0cbc3f8ffa3b281ca3503cfc117f8d88` |
| `wine_capture_identity_diagnostic_20261116/checker/worker/result.json` | `c18801f1de6e424b898d77aecdc54a73a4480c90bb231d558496164e80c20b66` |
| `qa/wine_capture_identity_final_20261116/result.json` | `a46384a9f361098d291be5a85e5a5096e4994c22fff21e1018674825b9eee4cb` |

输出不随 Git 提交；失败入口的原始源码可从提交 `04543af1` 恢复核验。
