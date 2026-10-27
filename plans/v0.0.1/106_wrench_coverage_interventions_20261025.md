# 106 按fit当前wrench覆盖选择独立追加干预

105证实group4自然状态在fit中，但wrench方向/moment超出现有干预条件范围；不把诊断阈值或开发标签用于选点。仅使用097原21条独立fit profile（排除086三profile）和103当前实测wrench输入，在每条profile选择一个新增anchor，不读holdout或任何y/probe目标/性能标签。

选择规则在采集前固定：用104原24fit统计对6维world-at-control-SITE wrench标准化；初始支持集为086原3训练anchor+097原21训练anchor。按097 profile原顺序逐条处理，候选为已有前一命令、尚未进入80步绝对goal override的native周期，排除该profile原first_close采集点。选择到当前支持集最小6维均方距离最大的候选，tie取最早，选后加入支持集。每profile恰一新增点，平衡每task7条，不按测试力大小/方向阈值或标签筛选；距离不是新验收门槛。

21新增anchor各baseline+x/y/z±0.1裁剪，旋转/夹爪与完整前缀命令不变。明确仅fit数据，原6holdout及086开发anchor身份保持，不把开发状态称为新初态。所需显式物理预算由确定性名单精确计算并写配置；完整数据资格/294阶段+owner实际wait/reap前不得fit。新namespace outputs/wrench_coverage_interventions_20261025，不覆盖097。

沿用086/097采集/独立XML FK/数值twist、机械、图像、标签和基线/前缀一致性审查，基线合格才采本组探针。CPU/OSMesa空CUDA/LD；初始化内部步骤另列。世界模型/fresh/fit0，不改变模型或控制器，原18/18及WAM开发/最终24条门槛保持。当前只准备与登记，尚未物理采集。
