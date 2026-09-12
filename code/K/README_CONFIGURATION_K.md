# Configuration K：775 异质数据无 Pc 工作流

本目录是本会话唯一的正式工程根目录。其他 PINN 目录只供参考，不得从中读取正式数据、
checkpoint 或结果。

## 已冻结的数据合同

- 训练/评价数据：`data/tables_cache_0_775_step1.pt`
- SHA-256：`2132a9be710e2f257ea397cd080654e689093d9c561be3edb5be1e471283e761`
- 173,688,979 行，776 个时间层，时间范围 `0`--`30986.943359375 s`
- 包含 `permeability`；全部有限、严格为正，范围
  `7.408182343938106e-15`--`1.3498588097987652e-14 m^2`
- 固定 `T_ref=31544.99609375 s` 和
  `analysis_t_max=31544.99609375 s`；775 数据全部保留
- 无 Pc：训练入口固定 `PINN_PC_ENABLE=0`、`PINN_PC_ENTRY=0`
- RAR：start 3500、interval 100、candidate 30000、residual 22500、global 0

训练和评价共同读取
`permeability/K_field_unique_grid.npz`。该 128×128 场由 775 缓存最早层的
唯一空间坐标确定，NPZ SHA-256 为
`fd7c4ef13b7d0cd3d30d32b3f24fe6819f91e62d4657fe01769d3830a75f0a47`。
完整审计见 `manifests/LOCAL_ASSET_AUDIT.md` 和
`manifests/REFERENCE_SOURCE_TRACE.md`。

Maxwell 资源设置继承已成功运行的 H 案例，证据见
`manifests/MAXWELL_H_CASE_BASELINE.md`；没有硬编码节点。

## 正式任务矩阵

正式训练为 exactly 20 个任务：5 个角色 × seeds `0,1,2,42`。

| role | EXP | STRATEGY | LEAVE_OUT |
|---|---|---|---|
| baseline | M0 | BASE | NONE |
| complete | M7 | BASE | NONE |
| no_representation | M7 | BASE | PLAIN_TWONET |
| no_front_supervision | M7 | BASE | FRONT_PLUME |
| no_local_fv | M7 | BASE | FV |

`configuration_k.py` 是训练、评价和 Slurm 共用的唯一合同源。

## 本地或登录节点预检

在工程根目录执行：

```bash
cd <LOCAL_PATH>storeplace1/PINNnew2/PINN12_723_clean
export PY="$HOME/conda/envs/py311/bin/python"
sha256sum data/tables_cache_0_775_step1.pt
sha256sum permeability/K_field_unique_grid.npz
sha256sum -c manifests/CODE_SHA256SUMS
"$PY" tools/audit_project_consistency.py --root .
"$PY" tools/validate_k_assets.py \
  --dataset data/tables_cache_0_775_step1.pt \
  --field permeability/K_field_unique_grid.npz \
  --field-metadata permeability/K_field_unique_grid.json \
  --reference-provenance manifests/k_reference_provenance.json \
  --run-kind smoke
```

若需要从冻结数据重新生成场（通常不需要）：

```bash
"$PY" tools/extract_k_field.py \
  --dataset data/tables_cache_0_775_step1.pt \
  --field permeability/K_field_unique_grid.npz \
  --metadata permeability/K_field_unique_grid.json \
  --grid-size 128 \
  --trust-local-pickle
```

`.pt` 是 pickle 容器；只对已核对完整 SHA-256 的本地可信文件使用
`--trust-local-pickle`。

## Maxwell 上传后检查

进入上传后的工程根目录，再执行：

```bash
pwd
sinfo
scontrol show partition spot-gpu
scontrol show partition uoa-compute
scontrol show node egpu001
test -x "$HOME/conda/envs/py311/bin/python"
"$HOME/conda/envs/py311/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
sha256sum -c manifests/CODE_SHA256SUMS
sha256sum data/tables_cache_0_775_step1.pt
```

若分区、账户权限或 Python 路径与 H 案例证据不一致，先修改并重新生成代码清单；
不要猜测 Maxwell 分区名。

## 先跑 smoke

```bash
sbatch --chdir="$PWD" slurm/run_pinn_K_smoke.slurm
```

等待结束后检查：

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
test -f smoke/complete_seed0/resolved_config.json
test -f smoke/complete_seed0/training_summary.json
test -f smoke/complete_seed0/final_checkpoint.pt
```

smoke 必须为 M7/BASE/NONE/seed0、50--200 iterations，并确认 Pc、时间、RAR、
数据哈希和场哈希均符合合同。smoke 不进入 20 行正式结果。

## 正式提交前的强制门槛

`manifests/k_reference_provenance.json` 当前保持 `verified=false`，因为仅凭缓存不能
独立证明原模拟 Pc 设置。用户已说明该数据无 Pc，但正式可复现实验仍需把原模拟输入或
日志复制到 `manifests/reference_evidence/`，记录 SHA-256，并完成 provenance 字段。

在该证据补齐前，do not submit 正式 20 个任务；正式预检会按设计失败。可以运行 smoke。

补齐证据后执行：

```bash
"$PY" tools/validate_k_assets.py \
  --dataset data/tables_cache_0_775_step1.pt \
  --field permeability/K_field_unique_grid.npz \
  --field-metadata permeability/K_field_unique_grid.json \
  --reference-provenance manifests/k_reference_provenance.json \
  --run-kind formal
```

## 一次提交完整正式链

只有 smoke 完成且 formal preflight 通过后：

```bash
bash slurm/submit_configuration_k.sh
```

该入口原子锁定命名空间并提交三个依赖阶段：

1. 20 个训练 array tasks；
2. 20 个逐 checkpoint GPU 评价 tasks（`afterok`）；
3. CPU 聚合与验证（`afterok`）。

不要手工重复提交同一正式链。提交器会拒绝已有正式结果或已有
`manifests/formal_submission.lock` 的目录。恢复前先用 `sacct` 确认全部已记录任务终止，
并归档现有产物。

最终应产生恰好 20 行
`evaluation/heterogeneous_metrics_by_seed.csv`，每行含五个固定指标：
`saturation_rel_l2`、`front_band_rmse`、`contour_chamfer_s0175`、
`local_fv_co2_rmse`、`co2_mass_rel_error`。

更详细的上传和命令说明见 `MAXWELL_UPLOAD_AND_RUN.md`。
