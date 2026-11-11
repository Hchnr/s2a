# 多方向物理对照的首次专项失败

2026-11-11。首次 32 并发专项 `outputs/qa/box_face_feedback_20261111/result.json`
有 43 项通过、1 项失败，零跳过和收集错误；所有 worker 已实际 wait/reap。失败
发生在接触统计测试：合成测试样本写入 `slide_axis_change`，而实际测量记录和
统计函数使用 `slide_axis_change_tcp`，因此触发 `KeyError`。此次是测试数据字段
不匹配，不是物理结果；**物理仿真、WAM 推理和新增命令均未启动**。

测试样本已改用测量合同的实际字段；后续专项使用新的输出目录，原失败不覆盖。
原运行时 owner 和测试源的字节分别保存在
[owner 副本](../../../docs/history/experiments/run_box_face_feedback_before_qa_fix_20261111.py.txt)
和[测试副本](../../../docs/history/experiments/test_box_face_feedback_owner_before_qa_fix_20261111.py.txt)，
其 SHA256 分别为 `5081879148d1fd3b3685ba036ac1ec92fd31cdd7bdc39cf19cdc9601d4f5114b`、
`37bac8846cbd6dbd7dbd7d15b45ed44f6499684f7c5a96ff64f92d2720db3849`。失败结果
SHA256 为 `9e1b8d3ee078863b83e1daeb736c86ceef18106495890ac66864c8bfaf3c82ed`。
