# Front-aware PINNs for CO2--brine displacement fronts

Repository: <https://github.com/ZhaiHenglai/front-aware-pinn-co2-brine>

This repository contains the manuscript-specific H/K training and evaluation source, figure-generation source, 20 sanitized K run configurations, portable Slurm templates, tests, environment information and external-asset metadata for the EMS manuscript. It excludes all large tensors, checkpoints and native IC-FERST output. Processed and figure-source datasets are deposited in the linked Zenodo records.

## Verification boundary

Run `PYTHONDONTWRITEBYTECODE=1 python3 tools/smoke_code_only.py` for the no-large-assets smoke check. The code-only checks validate source integrity, configuration structure, syntax, imports and tests that do not require large assets. They do not retrain the models, reevaluate checkpoints or rerun IC-FERST. The public Zenodo records containing the external assets are listed in `ZENODO_RECORDS.md`; each deposit includes its own manifest and SHA-256 checksums.

## Licensing and release status

See `LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `RELEASE_STATUS.md` and `ZENODO_RECORDS.md` for the licence scope, third-party notices, release state and external research objects.
