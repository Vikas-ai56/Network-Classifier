# Architecture Change Checklist

Use this checklist before and after making structural changes.

## 1. Scope and Pipeline Selection

- [ ] Confirm target pipeline (`main.py` dual-branch vs `src/*` image NetMamba).
- [ ] Confirm expected dataset format (JSON sequence or image folders).
- [ ] Confirm target metrics (SupCon/ProtoNet accuracy, top-1/f1, speed).

## 2. Contract Updates

- [ ] Input contract updated (feature names, lengths, dimensions).
- [ ] Model signature updated to match data contract.
- [ ] Training/evaluation logic updated to match model outputs.
- [ ] Checkpoint load/save compatibility considered.

## 3. Files to Review

- [ ] Entrypoints: `main.py`, `src/pre-train.py`, `src/fine-tune.py`, `src/eval.py`
- [ ] Model files: `src/models_dual_branch.py`, `src/models_net_mamba.py`, `src/models_mamba.py`
- [ ] Data files: `src/dataset_netmamba.py`, `dataset/dataset_all.py`, `dataset/dataset_common.py`
- [ ] Runtime loops and utilities: `src/engine.py`, `src/util/*`

## 4. Validation Expectations

- [ ] Run the relevant training/evaluation command for the edited pipeline.
- [ ] Verify shape compatibility and no runtime errors on one full batch.
- [ ] Verify checkpoint artifacts are saved at expected location.
- [ ] Verify metrics logging paths and output JSON structure.

## 5. Portability/Operational Readiness

- [ ] Replace hardcoded absolute paths if introducing new data sources.
- [ ] Confirm dependency requirements for added libraries.
- [ ] Document new CLI flags and defaults.

