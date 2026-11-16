# 保存状态接触重建的首次专项检查失败

2026-11-16。首次 32 并发专项检查收集 31 项，28 项通过、3 项失败，0 跳过；
所有 worker 已实际 wait/reap。此时没有启动登记的 18 例重建，也没有新
物理积分或 GPU/WAM 调用。

两项实际 MuJoCo 合成测试在导入时失败：执行环境设置 `MUJOCO_GL=osmesa`
但空 `LD_LIBRARY_PATH` 下未预加载仓库内的 OSMesa 库。失败发生于模型创建
前。第三项旧 bowl 诊断的历史来源准入检查因当前 `reset_feedback_policy.py`
与其固定历史 SHA 不一致而失败；该检查不属于本次两组已审计轨迹的来源准入。
历史来源校验没有被放宽。

本任务只做无渲染的静态查询，后续使用 `MUJOCO_GL=disable` 和相同仓库内
MuJoCo 2.3.7 二进制，继续空 CUDA/LD；新代码直接将当前二进制、原生谓词
来源、两组实际轨迹及 XML SHA 绑定，并用新的独占 QA / dry 输出验证。
新测试不将失败的旧 bowl 来源准入检查报告为通过。若正式查询发现来源或
原生标签不一致，仍须保留并限制解释。

原失败结果：`outputs/qa/seed_policy_contact_reconstruction_unit_20261116/result.json`，
SHA-256 `7f15ba03d09689855d61a17b7f2472ee4c2c1e644ace1a944d0f2c4ec4904efa`。
