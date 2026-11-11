# 原夹爪影子准入的根文件指纹修复

2026-11-11，原夹爪正确参考影子尚未执行。为记录用户恢复GPU的通知而修改的根
`README.md`和`AGENTS.md`属于已固定迁移清单的物理来源；首个CPU QA在测试
收集前以`Current physical source SHA drift`失败，0项测试、0物理、0模型调用。
失败收据`outputs/qa/defer_close_normal_correct_shadow_20261111/result.json`的
SHA256为`7f564fc1b066905ef637d959b443f6e9766da744c5e5abb9354400bed43ca1e3`。

恢复两根文件的原字节，用户已恢复GPU的事实保留在`docs/status.md`及本次会话；
GPU新UUID授权范围仍待确认。原已固定迁移清单、原影子计划、候选规则和18/4907
分母不变。改用独立的`outputs/qa/defer_close_normal_correct_shadow_v2_20261111`
再次做完整专项，只有其来源和零模型准入通过才执行一次原定影子；不覆盖失败
收据，不自动重试，不将失败改称已通过。
