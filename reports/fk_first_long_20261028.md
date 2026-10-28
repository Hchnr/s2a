# 早期真实起点首周期FK长窗口结果（2026-10-28，CPU）

[登记](../plans/v0.0.1/fk_first_long_20261028.md)以同一真实endpoint368、同一已生成chunk46、相同控制器/初始history/previous/8点warm，固定比较H8/12/16；长warm只重复末命令扩展。每组新建FK缓存，首预测点精度硬约束、全窗口pose/twist成本与关节/制动/terminal政策一致，但约束范围及terminal随H变化。不按不同H总objective直接排名。

| H | 首周期witness | 响应数 | 求解秒 | 独立几何达标端点 |
| --- | --- | ---: | ---: | ---: |
| 8 | 有 | 69 | 4.622809 | 8/8 |
| 12 | 有 | 90 | 11.483232 | 9/12 |
| 16 | 有 | 122 | 27.597778 | 9/16 |

三个候选均有首周期数值解，共281响应。所有36个预测构型的MuJoCo FK/J与机器人XML实现一致，最大pose差2.554e-15、J差1.777e-15；完整重建joint rollout、动作box/rate、内环历史裕量、制动和terminal约束通过原容差。H16内环最小约束余量-3.928e-11rad、制动-4.357e-11rad在预先固定1e-10数值容差内，不能宣称严格正余量。H12/16未来分别3/7端点精度未达标，完整标签保留；首点witness不代表完整预测轨迹或真实控制成功。

producer和独立检查均actual wait0/reap0，新增物理/模型/fresh0。producer verified SHA `68348411dcb921b7dbaa1955a9b991a725cabc7ae36961ab3ba4f8a322536f11`；checker `3584e1bcf5f4499dbbbd1e102c514ed177ac977b7fbaad1dcbd8a93301787849`。相关4测试通过3.18s，新脚本Ruff/format通过；未重复全量回归。

[下一阶段](../plans/v0.0.1/fk_first_long_native_20261028.md)预先登记全部三组各最多16新周期的同起点实际反馈与独立重放；当前无实际比较结果。GPU禁用，原18/完整12WAM/final24资格不变，总目标未完成。
