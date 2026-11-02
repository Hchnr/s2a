# 160 数值接口修复后原18条控制回归

绑定159真实164步goal解码验证，新版本只将模型旋转goal输入改为实际固定控制器数值路径。原146参数、18profiles/4907步/seed42、083旋转/081夹爪和149有界平移保持。旧157失败和冻结源码不变，输出新namespace。重新CPU/18实际CLI/checker导入准入，冻结后持久supervisor/owner依次采集及完整独立检查，不自动重试。

严格goal检查不放宽；18/18稳定、位置RMSE<=.01m、SO(3)RMSE<=.05rad、36stage及owner退出/来源通过前WAM/fresh关闭。CPU/OSMesa空CUDA/LD，无GPU/全局fit/世界模型/fresh。
