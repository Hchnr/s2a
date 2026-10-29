# 原H12时序的有限信赖域对照（执行前登记，CPU，待执行）

总目标与原18正确参考、12WAM开发及最终24留出任务门槛保持。GPU继续禁用，实际仍仅到原cursor8、task/grasp0。本轮尝试在完整安全响应验收后重新选择优化中心，争取跨过cursor9并继续达到全部H12目标；不把局部下降改称控制成功。

## 来源、假设与相同起点

[固定回缩检验](../../reports/saved_direction_backtracking_20261028.md)及完整独立v2 checker均关闭。producer SHA`0957f81930cd65d08761220c76000374e1b7c9826872145319e311a42e26028e`；checker SHA`f41948e32a766d2ad3b29528f247527c6286ec37230f8f4fc575f59ac7d58136`、checker owner SHA`c19227f75375c94fae8d20319e3cee68f9856674e4fb73ab6e6acfae7f3d6600`。启动必须重新绑定实际成功wait/reap、全部预算及来源未变，不能只判断文件存在。第一版checker的元数据失败和300内步/313查询保留，不重复执行producer。

14/16方向有较短可行下降，但0/96达到target、完整witness0。这个结果仅支持候选逐条完整响应验收；旧SLSQP本身已有线搜索，不能声称原求解器没有回缩，保存cache trace也不是完整accepted-iterate历史。新实验检验显式有限半径与即使prefix未增加也能重新定中心的规则，而非继续加密旧固定方向。

只用原时序H12，不新增H13或改参考。两臂同一warm：上轮producer `cases.prefix.entries[55].request.commands`，base3246→trial3319、alpha=.0625，完整bytes冻结。此点joint合格、prefix9、非线性minimum`7.5647246729e-6`、target max`1.3249481018`；相对旧selected`1.32624757`改善有限。不能混用不同初态或为每臂选择不同warm。每臂从同actual368 q/v、completed367 command/latch和相同固定手指状态重新计算warm并计费。

世界坐标/TCP、原chunk46 targets0..11、50ms/25×2ms、原生goal递推、逐内步更新M/J、implicit damping Euler、扭矩裁剪、关节margin .05、历史零profile、制动.1s及terminal q7速度约束全部不变；不能读取actual369+。原position .01m/rotation .05rad、box/rate1e-12与动态1e-10审计保持。

## 唯一A/B及有限预算

A调用冻结`solve_prefix_preserving_trajectory_response`，仅替换为两臂共同新warm。B使用新源码副本保留相同目标、完整约束/Jacobian和精确bytes响应缓存，改变下列控制流。此A/B比较整套半径、短阶段、验收与重新定中心机制，不能将差异单独归因于radius。

1. 每短阶段以最近已接受命令为中心，原box/首命令rate bounds再与`anchor±radius`相交。全部H12×6动作仍可联合调整；原跨周期rate线性约束保留。初始radius=.01。
2. 每阶段最多5次SLSQP迭代（剩余总迭代不足时取剩余），相同epigraph target目标、`1e-4`全轨迹loss权重、有限差分步1e-4、ftol1e-12。有限差分选向遵守本阶段bounds，不裁剪真实候选来伪造已评价命令。响应缓存可跨阶段复用；导数缓存每阶段清空，不能仅凭x bytes复用不同bounds下的旧差分方向。
3. 每个被评价命令必须执行完整12周期响应、九组审计与全部TCP误差；保存违规探针。普通新anchor必须保住原hard前缀、完整joint/command合格、非线性minimum≥1e-8，并且prefix增长或当前target max较旧anchor下降>1e-12。只比较实际非线性结果，不以线性预测或optimizer success代替。
4. 每阶段结束从本阶段实际评价/命中缓存的合格命令中按最大prefix、候选首个失败cursor的target max、全轨迹loss选择（同冻结prefix_rank），稳定tie按首次评价顺序。第三条验收的下降仍比较该阶段原target，不能偷换成新cursor。完整H witness优先，按原最大归一化误差/全轨迹loss排序，并沿用原定义：full audit和全部caps通过，不强加优化器额外buffer。
5. 普通下降获接纳后，即使prefix仍9也更新anchor、重新线性化，半径回到.01；若本阶段没有可接纳下降，anchor不变，按固定`.01→.005→.0025`依次缩小。三半径都失败则停止。不扩半径、不换warm、不加入结果驱动的额外设置。prefix增长后继续cursor10/11，不能在cursor9通过即宣称完成。

每臂共享最多4000唯一完整响应、60累计SLSQP迭代、60阶段。warm、所有差分/目标/约束/最终检查均共用同一精确bytes缓存和预算；命中旧缓存不得伪记新响应，任何新响应都不能免费。B记录每次请求（包括缓存命中）的顺序、stage、role、command key、真实新响应数，以及被接受anchor/拒绝原因/半径和完整约束。A保存冻结函数实际提供的阶段与唯一响应trace、由外部predict计完整响应，不声称恢复其未暴露的缓存调用顺序或accepted-iterate历史。迭代按实际callback/nit记录，异常退出仍计已发生工作；不得因阶段重启重置总预算。时限每臂1200s，无自动重试。

同一数学响应计每条300数值内步、312查询，各臂另一个current query；若用满4000，producer每臂1200000内步/1248001查询。独立checker逐条重建该臂全部实际唯一响应，另计其查询/内步，不免费复用producer。上文1200s是每臂producer时限；每臂完整checker单独设5400s，超时仍保留失败，不缩减应核trace或获得资格。阶段/迭代没有用满如实报告，不追加其他case补预算。

## 保存、验证与后续门槛

新增solver、producer、独立checker、owner及必要合成测试，旧执行文件全部冻结。先验证“prefix未增长的真实下降可更新anchor”、违规/丢前缀候选不可更新、radius/所有预算不跨阶段重置、缓存命中与真实调用可区分、H12全部caps才能witness，以及实际状态/goal重置，再原生CPU QA、CLI dry和提交。入口/输出分别使用`tracking_v5_prefix_trust_baseline_20261028`和`tracking_v5_prefix_trust_candidate_20261028`，独立`_check_`及`_execution_`目录，拒绝覆盖。

两臂必须同版本/同来源QA；A、B完整执行与检查分别记录actual wait/reap及source/data/seed42指纹。报告全部失败和完整trace，比较cursor9是否真正达标、最终原节点prefix、完整H witness、目标误差及达到相同资格消耗的响应/迭代；单独列每次接受anchor的改善和活动约束。0witness不得改门槛或宣称无全局解。

本轮无新增物理/模型/fresh。只有完整局部模型witness经独立核验后，才另登记同初态物理pilot与完整重放；原实际控制及留出任务资格仍需真实证据，不能用本地求解器成功代替。
