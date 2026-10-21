# 022 启动前条件隔离补充

本补充登记时022模型请求、模型服务启动及物理步均为0。原prepared.json（44cfd7…）及prepared_v2.json（429213…）只做过CPU准备，保留原件，不将准备元数据更新算性能重试。

独立source peer在启动前指出：当前服务的task-only renderer在seed42/43均产生空`task_plan: []`，而两字段JSON输入会缺少这个空键。直接使用先前JSON将违反“实际条件只差command”的登记目的。CPU直接复算当前renderer已确认两个seed都有此空占位；这不是已证明的性能失败原因。

expanded请求静态JSON保持原两字段，sample manifest不改写；服务在实际stock准备后、模型调用前显式补`task_plan: []`，仅保留同baseline的空schema占位，不携任何示范计划、current_subtask、阶段、未来状态或任务进度。baseline必须本来就具有同一空占位，原batch保持；expanded若出现非空计划或current_subtask直接拒绝，不能清空以掩盖输入。task_title和长command选择规则不变，同一task全部样本/seed同一静态JSON。原“移除task_plan”具体指移除含示范信息的计划，空占位按本显式补充处理。所有门槛、输入图像、采样参数、模型与96请求/0物理预算保持。

在模型启动前，用实际外部只读`prepare_wam_request`及同一新增空占位处理，对全部48对请求运行CPU准备，逐对证明实际batch caption字典仅command不同、其余键和值完全相同，实际首帧像素与源图及配对均精确、mask/domain/fps/shape相同。不导入训练dataloader、不加载模型权重或进行生成。若任何一对不符，保留错误、停止模型阶段，不通过修改其它caption键掩盖。新prepared_v3只允许在没有任何模型/查询/编排产物的目录中建立，现有sample manifest只读逐值相同后才使用新文件，不覆盖旧prepared或样本。
