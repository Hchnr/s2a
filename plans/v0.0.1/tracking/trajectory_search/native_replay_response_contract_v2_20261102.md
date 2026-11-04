# 完整响应对照 v2：比较器结构修复

2026-11-02（Asia/Shanghai）。沿用
`native_replay_response_contract_20261102.md` 的科学问题、固定命令与容差。

首轮因比较器不能遍历指标中的 `(slack_array, tolerance)` 元组而退出，原结果
`outputs/native_replay_response_contract_20261102/result.json` SHA256
`36ba28e6311f1fe7899a96e729736e181794a234c0706aefe59c8acc3f0d39c7`。
此为执行工具错误，不是算法或完整响应通过。原结果不覆盖。

v2 仅使比较器对 tuple 与 list 一样逐元素递归；实际执行前用独立结构预检验证
嵌套字典/tuple/数组/布尔及故意超差能正确记录。响应实现和 checker 不变。
先写新 registration，固定 [0,0,3,4]，最多8响应、2400内步、2498查询、2model。
首轮实际2响应、600内步、626查询、2model另计，两轮合计上限10响应、3000内步、
3124查询、4model。不追加已关闭原 search 的122响应预算。
独占新目录、无自动重试、无搜索/物理/GPU/WAM，失败仍保留 False。
