# 结构迁移与来源身份

[全局设计](../architecture/repository_design.md)规定顺序及验收。
[基线清单](baseline_20261029.json)只记录迁移前的3276项身份，不预判新路径；
基线commit为`7b2712850dea54090d61e564380f88825ad8b8f6`，清单SHA256为
`1a7d3737696e5c6cb1c4d22bc572257dd5994493a01bcd991228df9f9eb23897`。

## D1：路径及历史字节定位

`s2a.infrastructure.repository_paths.repository_root()`从明确anchor查找项目标志，
不依赖固定目录深度；`repo_path()`校验仓库相对路径。

`s2a.infrastructure.source_identity.SourceIdentityLocator`接收明确root、baseline
commit、catalog_ref、migration_refs及outputs内cache目录。`resolve(declared_ref)`
首先确认原路径/SHA已登记，再返回declared_ref、storage_ref、migration_ref、
catalog_ref、原Git来源及resolution_kind。调用方最后执行`finish()`核验整批。

后续迁移清单绑定相同catalog_ref，moves每项包含：

```json
{
  "old_path": "src/s2a/example.py",
  "original_sha256": "<baseline SHA256>",
  "new_path": "src/s2a/research/example.py",
  "current_sha256": "<current storage SHA256>"
}
```

路径在迁移后改字节时，原身份定位原Git blob，当前源码另有指纹。已登记current
目标漂移会失败，不通过fallback掩盖。未知版本、错误blob、缓存损坏、路径越界
及旧根访问均拒绝；不会自动导入找到的源码。结果scope固定为`byte_identity_only`，
不授予历史代码执行、runtime或控制资格。旧JSON与旧helper目前保持原样。

D1专项36项CPU测试通过（最终session28118实际exit0，35.12s），新文件Ruff及
format检查通过。真实3276项catalog接线、active源码定位及本地Git收尾通过。

## 测试发现基线

`outputs/repository_baseline_collection_20261029/`保存迁移前实际collect-only收据、
日志与nodeids。session17648/child3375668实际wait/reap；收集到4976节点，5项既有
collection错误，退出码1。错误均为CPU环境PyArrow要求NumPy≥2.0，而当前用于
数值回归的解释器为NumPy1.26.4。没有执行测试用例；不能将此记录称为全仓通过。

后续按旧→新文件映射比较测试node集合，并分别报告新增测试和这些既有错误。
收集器中的模型/任务零计数字段没有独立运行时探针；本基线只用作测试发现与
源码未变的证据，不据此授予任何模型或仿真执行资格。

## D2：首批完整领域

[首批清单](trajectory_search_20261029.json)绑定85个实际移动文件和81个原位置修改。
8个研究实现、23个入口/独立审核、24个测试、7份配置、9份计划、14份报告分别位于
各自的 `tracking/trajectory_search` 或 `research/trajectory_search` 目录。配置、计划、
报告30份原字节保持；历史Markdown仍按旧声明位置解释相对链接。

[根文件补充清单](root_files_20261029.json)只登记原 `.gitignore`、`AGENTS.md`、
`README.md`、`pyproject.toml` 四项。它与3276项主清单采用同一基线commit，不扩大
到用户未跟踪文件。当前Python包导入及pytest路径另行记录新源码身份。

最新两个入口使用 `s2a.research.trajectory_search.inputs.SearchInputContext`，分开
历史源码、当前源码/输入和已固定的外部仿真资产。历史源码只通过catalog/locator
证明字节；32个外部mesh/纹理依据已绑定bundle的exact path/SHA只读核验，不作为
可执行外部代码。`finish()`重验已消费来源，未知版本、漂移及旧清理根均拒绝。
迁移未重跑生产或完整checker，也不升级原failed checker的资格。

当前命令从仓库执行，例如：

```bash
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH='' PYTHONPATH=src:scripts:. \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1 \
outputs/libero_env311/bin/python -m \
  scripts.tracking.trajectory_search.probe_tracking_v5_state_vector_dual_recovery --dry
```

两个真实入口dry均通过，QP/响应/查询各0。首次dry对32项外部资产的分类拒绝记录
保留在 `outputs/trajectory_search_migrated_dual_dry_20261029/`；修复后的独占v2记录
在 `outputs/trajectory_search_migrated_dual_dry_v2_20261029/`，owner和两child实际
wait/reap、exit0，175项来源前后一致。其适用范围是当前入口准入，不是重新执行
历史搜索或控制。

全量collect-only得到5096节点：旧4976节点按文件路径映射后完整保留（参数及顺序
保持），新增120；仅保留原5项PyArrow/NumPy错误，没有新增collection错误。
原退出码1与比较通过分开记录在 `outputs/repository_restructure_collection_20261029/`。

统一302项CPU QA最终通过；v2独占收据位于
`outputs/repository_restructure_unified_qa_v2_20261029/result.json`，SHA256
`fabd2a958a242ab5c303e0a9c5b9a2a9c9256dbdf094daf784889ef9759faf7c`。
owner及child实际wait/reap exit0，3317项来源/文档在本轮前后一致。首次统一QA的
302测试通过，但因进行中编辑了本说明导致owner的冻结检查False，原失败收据
保留；v2显式绑定该失败及唯一文档差异，没有修改测试门槛。153个修改/新增Python
文件的Ruff/format均通过。

最终结构及节点检查收据为
`outputs/trajectory_search_d2_final_structure_20261029/result.json`，SHA256
`33e568d2c423f4dc01140bb8ae8e8e6467d7c6a149901602c258cf14fa9f6e26`。
此时根平铺仍为src378/scripts1467/tests442/configs192/plans341/reports365，
`full_restructure_complete=False`。

维护检查入口为 `python -m scripts.maintenance.check_repository_structure`；
`--manifest`显式传迁移清单，`--additional-catalog`显式传根文件补充清单。
它核当前存储SHA、映射后的pytest节点及根平铺数量；首批结构通过不表示全仓整理
完成，更不表示控制或运行时资格通过。
