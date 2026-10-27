# 四步预览的 WAM 响应身份校验准备

本次补齐逐响应的开发案例、raw/terminal058 分支、读出参数及当前请求绑定。尚未接入物理执行入口，不启动模型、不消费 fresh；原18/18正确参考控制门槛保持。

## 实现与验证

`src/s2a/coupled_preview_wam_protocol.py`接受实际保存的当前图像请求、登记的开发案例、请求序号、checkpoint与固定058参数引用。先核对全部请求字段与响应args一致，再核对12案例/每案例75请求顺序、seed、分支及参数身份；最后沿用独立Action128合同检查，返回原生成命令而不修改。固定058 SHA为`30210a2bda69962bbd24f8e86849f58c4516d2209bfb3d8a5903fc45ae543b4f`。

测试使用旧061真实raw/terminal响应的字段投影fixture，保留原响应路径/SHA，明确删除仅用于诊断的normalized数组。13项CPU测试通过，覆盖原值保留、分支/案例/请求错位、错误参数身份、当前图像/任务/名称不一致、布尔索引和非零无效槽。三个新增Python文件Ruff与格式检查通过。

`scripts/audit_coupled_preview_wam_protocol.py`已实际退出0，完整读取旧061的12案例、900对请求/响应、36000个参考端点，逐项通过新校验并复查输入SHA；输出`outputs/coupled_preview_wam_protocol_20261026.json`。36000是40端点×900请求，不是35个去噪节点×900请求；没有新增推理、物理或fresh结果。

## 证据边界与下一步

此回放核对历史JSON内部的请求/案例身份及每案例checkpoint一致性。回放脚本的checkpoint基准来自每案例首响应，不能替代新服务的独立checkpoint来源核验；在线调用者必须从登记服务配置传入checkpoint，不能信任响应自报身份。

该函数不读取专家轨迹或未来状态，但单靠字符串一致不能证明输入PNG确为当前图像；还需现有服务ledger的图像SHA、请求预算、同首图检查和物理时钟证据。它也不核验模型GPU UUID、服务实际来源、18条控制资格或模型退出，不能单独授权WAM执行。

按237接入仍需：派生独立WAM物理入口（当前preview入口明确只允许oracle）、绑定注册checkpoint及允许UUID、接入本函数与原生目标/SOURCE提交/连续双历史、准备新服务与owner的冻结来源和真实CLI准入，最后执行raw/terminal固定12条开发对照。当前preview回归已见前8条独立核验全部通过，owner与closure仍经/proc确认存活，未获得完整18/18资格。
