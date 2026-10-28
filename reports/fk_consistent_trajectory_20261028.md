# FK一致预览的同状态比较（2026-10-28，CPU）

[登记候选](../plans/v0.0.1/fk_consistent_trajectory_20261028.md)保留原冻结joint dynamics、关节/历史/制动/terminal与每端点精度，只改成FK(q)/J(q)qvel及一致导数；同一真实368、同一已生成40点参考和原warm，无物体状态规划输入。

| 窗口 | 完整模型witness | 响应 | CPU求解(s) | FK查询/实际缓存未命中 |
| --- | --- | ---: | ---: | ---: |
| 8 | 有 | 71 | 4.5847 | 8521 / 8521 |
| 12 | 无 | 160 | 21.3612 | 28801 / 28560 |
| 16 | 无 | 114 | 42.2072 | 27361 / 26985 |

8点解的全部端点FK目标误差最大.00247193504m/.02400348397rad，最小收紧内环slack .05186445812rad；8点全部通过独立raw-state XML FK对pose、world SITE twist和TCP误差重算。相较旧“模型8/8、同q FK5/8”的反例，内部几何一致性已修正。它仍只是当前冻结free joint模型中的命令，不是实际未来状态或控制成功。

12点耗尽160响应，最好已检查线性可行尝试最大归一化pose超限约1.51150；16点迭代上限，最好尝试超限约4.71308。未找到witness不是全局不可行证明。所有候选共享原160响应/60迭代上限，没有因计算更贵就放宽原验收。墙钟4.6/21.4/42.2秒远超50ms，不取得实时控制资格，物理时间步.05s不能代替实际计算时延证据。

数值owner actual wait0/reap0，verified SHA `443e1661345459ecc4a91a468971e49ed332371c3b8e470d7e0ae98667326b84`。

## MuJoCo交叉核验

独立入口`check_tracking_v5_fk_consistent_trajectory.py`不导入优化器/新目标函数，不重新搜索；各窗口固定选模型witness，否则选已保存最好线性可行尝试。重建冻结joint rollout后用MuJoCo XML-only mj_forward/mj_jacSite核验36个预测构型的SITE pose/J与目标残差、原动作/内环/制动/terminal约束。最大XML/MuJoCo姿态差2.55351296e-15、Jacobian差1.77635684e-15；8点合规标签8/8一致，12/16仅6/12、8/16在管内，仍不接收。没有执行候选动作或推进物理time。

交叉核验owner actual wait0/reap0，verified SHA `679a0f119a8d05300442921f29caa4f2a42e24440f2d00d463ba4266cd581cb1`。候选5项CPU测试通过（1.25s），新增源码Ruff/format通过，未重跑全量。新增物理、模型调用、fresh均0，GPU继续禁用。

下一步按[原生反馈计划](../plans/v0.0.1/fk_feedback_native_20261028.md)从实际368重放前缀，最多16个真实前向控制周期，包含关键cursor9；每周期只用自身当前状态及已完成历史重建规划。必须记录实际state error与关节裕量，不能把本次模型/几何资格当作执行或任务成功。原18/完整12WAM开发/final24门槛不变，总目标未完成。
