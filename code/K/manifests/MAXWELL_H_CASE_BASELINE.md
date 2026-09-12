# Maxwell 基线：723 H 案例

本文件记录 Configuration K 的 Maxwell 运行模板来源。它只证明 H 案例曾在该运行环境成功执行，不能替代 K 参考物理模拟的无 Pc provenance。

## 证据文件

- H Slurm 模板：`PINN12_723.pt/run_pinn_v100_3.slurm`
  - SHA-256：`064e4c7ac1460cd9c315a634f8f8375df7b16dea0b027c1e65ea879e3f5afe85`
- 代表性成功日志：`PINN12_723.pt/logs/pinn_M7_BASE_FRONT_PLUME_seed42_v100_tables_cache_0_723_step1_3379810.out`
  - SHA-256：`1892086e4bf9653c04664996bea2a0b72c7262977e63056f57a2ecb8c28b1eb2`
- 对应 stderr：`PINN12_723.pt/logs/pinn_M7_BASE_FRONT_PLUME_seed42_v100_tables_cache_0_723_step1_3379810.err`
  - SHA-256：`2e11e21b353f61288bb71bee51c2a00319936b457d3862da03f1d9bc59b5d76e`

上述日志记录 Job `3379810` 完成 20,000 iterations、保存 final checkpoint 并正常打印 `End`。

## 已由实际日志证明

- host：`egpu001.int.maxwell.abdn.ac.uk`
- GPU：`Tesla V100-PCIE-32GB`，32768 MiB
- Python：`Python 3.11.7`，路径族为 `$HOME/conda/envs/py311/bin/python`
- 运行设备与精度：PyTorch CUDA、`torch.float64`
- CPU threads：4
- GPU compute capability：7.0，80 SM
- `nvidia-smi` 驱动：550.144.03；驱动报告 CUDA 12.4
- PyTorch build 报告：`torch.version.cuda=12.8`
- 该代表任务峰值显存：18.57 GiB allocated、18.65 GiB reserved
- 实际运行时间：2026-06-11 19:27:03 至 22:44:31 BST，约 3 小时 17 分

对当前保留的 124 个 H `.out` 日志进行一致口径复核，124 个均有正常 `End`；其峰值 reserved GPU memory 范围为 12.78–28.57 GiB，中位数为 18.44 GiB。因此 K 不能把任意 `gpu:1` 视为等价于 H 的 32 GB V100。

驱动报告的 CUDA 版本与 PyTorch build CUDA 版本含义不同，归档时不得合并成一个字段。现有 H 材料没有记录精确 PyTorch package 版本，因此 K 必须在每个 `resolved_config.json` 中重新记录实际版本。

## 继承到 K 的 Maxwell 模板

- GPU 训练：`spot-gpu`、1 node、1 task、4 CPUs、80 GiB RAM、1 GPU；正式训练 walltime 7 天。
- GPU 评价：`spot-gpu`、1 GPU；每检查点独立评价，walltime 2 天。
- CPU 聚合：H 评价工作流使用的 `uoa-compute`。
- 初始化：`source /etc/profile`、`module purge`、`module load slurm`，随后才启用 shell nounset。
- Python 默认：`$HOME/conda/envs/py311/bin/python`。
- 启动：由 Slurm 分配资源后使用 `srun` 调用该 Python。
- K 的 GPU preflight 要求 V100 且总显存至少 30 GiB，以保持 H 的已验证 V100 runtime profile。

K 脚本没有硬编码 `egpu001`。`spot-gpu` 在 H 日志中实际落到该节点，但固定节点并非科学合同，并会降低另一账号下的可调度性；GPU preflight 会对实际分配的设备 fail closed。若管理员要求固定节点，应先用 `scontrol show node egpu001` 核实后再修改脚本与代码清单。

## 明确不继承

- H 的旧工作目录 `$HOME/PINNnew2/PINN12`
- H 数据 `tables_cache_0_723_step1.pt`
- H 的 `Results1`、日志和 checkpoint 命名空间
- H 的完整 M0–M7、完整 leave-one-out、训练策略和 holdout 提交矩阵
- H 的 `USE_DATA_MAX_TIME_AS_T_REF=True`
- 已弃用的 `PYTORCH_CUDA_ALLOC_CONF`

K 继续使用新工程根目录的动态路径、`data/tables_cache_0_775_step1.pt`、固定 H 时间尺度和 5×4 任务矩阵。

## 证据边界

旧日志证明的是历史运行，不保证提交当天 partition、QOS 或节点状态不变。正式入口仍先执行资产检查和短 smoke；实际 job ID、host、GPU、Python/PyTorch/CUDA、walltime 与峰值显存由 K 的新产物记录。若 `spot-gpu` 或 `uoa-compute` 已变化，只能依据 Maxwell 当前只读查询结果修改，不能猜测名称。
