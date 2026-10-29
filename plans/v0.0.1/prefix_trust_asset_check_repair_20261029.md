# H12独立检查的完整资产路径迁移（执行前登记，CPU）

原总目标、18正确参考/12WAM开发/24最终留出门槛保持。GPU禁用。本登记只修复已完成H12两臂数值搜索的独立检查加载路径，不重跑producer，不改动作、参考、约束、容差或数学，不进行物理pilot。

## 已关闭的生产与失败证据

- baseline producer SHA`c3814b6d0bf3db48d85865ee23c4104d2cc1fb0cde90f06d37b0d4c3c8c5c841`，owner SHA`11198fad76716be971aa3b4bb48e62a1cc9af94b0efd75f78938bd28b868cb23`；4000响应/53迭代/1200000数值内步/1248001查询，prefix9、完整witness0。
- candidate producer SHA`7338118bcad2cc80644555fcee42f9257b411cf9fa6234444fe1790601386d34`，owner SHA`1c103741c1da5f97450dfb25a9f2851083cef3bc751070b3390bb30d71264f28`；3148响应/45迭代/9阶段/944400内步/982177查询，三半径停滞、prefix9、完整witness0。
- candidate第一次checker SHA`cc4993e6e0e2772ad4e2172910cf7f93b634b1cb94f61753ae18499f588cea47`，owner SHA`335ea5abf56ea58b41d653b8d5ba7953ef22fc20f2e55ec985c433a6ccdf9a84`。完整XML加载时报旧`/share/project/.../wooden_cabinet_top_vis.msh`不存在；实际wait1/reap1、无signal、无retry，0响应/0查询/0数值内步。baseline checker仅metadata dry，未启动实际重构。

原producer和checker都加载完整XML；并无robot-only裁剪。两producer此前成功加载，而checker后来加载失败。当前旧外部路径有16个唯一路径缺失（19个引用），但不能据此确定删除时刻。1969项来源未变不覆盖这些间接资产的运行时字节，必须明确此缺口。

## 精确资产闭包

原XML SHA`50d3ece2777882c860513208d8c5ac75a31f49ff9db31d034211b5879c98f0e6`不变，共99个file声明、96个唯一路径、无include。历史记录`outputs/terminal_development_20261025/historical_oracle_xml_migration.json` SHA`a4a423a077c2a99ec9817159c48d89c6d6e01a7f816cd4cb13bcca84d531d186`绑定该XML SHA并保存全部96资产SHA。只读核验显示当前安全副本96/96匹配：32个LIBERO资产由旧`/share/project/`前缀明确迁移到`/share-mixed/project/`，64个robosuite资产已在当前仓库环境内。历史记录的`historical_hash_recorded=False`指更早015阶段未存hash，不能混同于本20261025已保存的资产指纹。

新`outputs/prefix_trust_asset_snapshot_20261029`必须独占创建。逐项读取安全副本、验证历史SHA后复制到该仓库内assets，按摘要/序号保留文件扩展名、防同名覆盖。保存原XML和新XML；只替换每个file属性的路径字节，逆映射须逐字恢复原XML。不能删body/geom/mesh、改density/viscosity、惯性、关节、执行器、选项或任何其他内容。保存99引用/96资产映射、历史/安全来源/本地副本SHA；执行前后全部重哈希。原旧路径不写入、不恢复，外部安全源只读。

先运行纯合成测试，再一次完整资产准入和native MuJoCo2.3.7编译检查。编译仅创建完整MjModel，记录模型尺寸/运行时和SHA，不创建预测响应、不执行mj_forward/mj_step；新响应/查询/数值内步/物理步/模型调用均为0。复制和编译失败均保留，不覆盖重试。

## 隔离checker与完整预算

旧source/tests/plan/result冻结。新增asset helper、repair metadata、snapshot准备入口、v2 checker/owner和必要测试。v2从冻结checker的main AST派生，只允许两处转换：准入调用换为扩展repair准入、独立结果目录换为v2；逆转换AST必须与原main完全一致。其余独立数值函数、严格分类、排序、计账及九组审计保持。case_engine薄包装仅复制source_reg并把xml_ref指向已验证的新XML；原登记引用保留，传给原case_engine的其他参数和执行阶段不变。不以AST一致代替资产字节或模型完整性证明。

两臂使用相同修复源码/QA和同一96资产快照，绑定原两producer成功wait/reap与原9source SHA不变；候选失败闭合也必须匹配。每臂新路径`outputs/tracking_v5_prefix_trust_{baseline,candidate}_check_v2_20261029`及相应`_execution_20261029`，独立owner实际wait/reap，无自动重试，时限各5400s。

必须重建全部保存trace：baseline4000、candidate3148；分别1200000/944400数值内步和1248001/982177查询。合计7148响应、2144400内步、2230178查询；旧checker失败0不得漏报也不得重复算成一次响应。只initial/selected/closest保存完整rollout，其他trace比较全部误差/严格标签；候选所有接受anchor另比较完整审计，逐请求/缓存/阶段/半径/预算核对保持。

新旧共6组组件测试和所有source/plan绑定同一native QA，真实CLI dry通过并提交后启动v2。原数学结果须全部严格一致，不能为了运行通过放宽阈值。这里只能证明安全副本匹配20261025资产快照及新完整复算，不能补称两producer加载当时已逐资产核hash。

完整检查未关闭前不授予本轮数值资格。即便检查通过，两臂均无完整witness，当前不进入物理pilot；实际仍cursor8、task/grasp0，总目标未完成。
