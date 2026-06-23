# Models & Datasets

## Models Used (open-weight only)

**No pretrained open-weight model is used in the shipped inference path.** The
`DualBranchEncoder` is trained **from scratch** on CESNET-QUIC22. This is intentional —
the network-traffic domain has no standard pretrained backbone, and training from scratch
on flow features avoids importing irrelevant priors.

The architecture builds on open-source *architectures/components* (not pretrained
weights):

| Component | Type | License | Link |
|---|---|---|---|
| Mamba SSM | Architecture / library (`mamba-ssm`) | Apache-2.0 | https://github.com/state-spaces/mamba |
| NetMamba | Reference architecture for traffic encoding | see repo | https://github.com/wangtz19/NetMamba |
| Supervised Contrastive Learning | Loss methodology | BSD-2-Clause | https://github.com/HobbitLong/SupContrast |

> If a future variant loads pretrained open-weight encoders, list their Hugging Face
> links here.

## Models Published

We developed the **DualBranchEncoder** as our solution and **published** the trained
checkpoint (~24 MB, ~1.98 M params) to Hugging Face under the **Apache-2.0** license.

- 🔗 **Hugging Face model:** https://huggingface.co/dhruvsinghal1387/dualbranch-quic-encoder

Published files: `best_model.pth` (encoder, epoch 28, val acc 0.927), `prototypes.pth`
(class-prototype gallery for nearest-prototype classification), and a model card (`README.md`)
covering architecture, the `(30,3)` + `16`-feature input contract, the 5-tuple-purge
rationale, KPI results, and a usage snippet.

Load it in a few lines:
```python
import torch
from huggingface_hub import hf_hub_download
from src.models_dual_branch import DualBranchEncoder

repo = "dhruvsinghal1387/dualbranch-quic-encoder"
model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
ckpt = torch.load(hf_hub_download(repo, "best_model.pth"), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

How it was published (run locally where the checkpoint lives):
```python
from huggingface_hub import HfApi, create_repo
repo = "dhruvsinghal1387/dualbranch-quic-encoder"
create_repo(repo, repo_type="model", exist_ok=True)
api = HfApi()
for f in ("best_model.pth", "prototypes.pth"):
    api.upload_file(path_or_fileobj=f"model/{f}", path_in_repo=f, repo_id=repo, repo_type="model")
```

## Datasets Used

| Dataset | Description | License | Link |
|---|---|---|---|
| **CESNET-QUIC22** | ~10M+ real QUIC flows from the CESNET ISP backbone; per-packet info (PPI), flow stats, app categories. **Primary training & evaluation dataset.** | Open (Creative Commons; see dataset page) | https://github.com/CESNET/cesnet-datazoo · dataset article: https://doi.org/10.1016/j.dib.2023.108888 |
| **ISCXVPN2016** | VPN / non-VPN encrypted traffic captures (reference / cross-dataset checks only) | Academic/research use (UNB) | https://www.unb.ca/cic/datasets/vpn.html |

CESNET-QUIC22 is accessed programmatically via the `cesnet-datazoo` package, which streams
the data — no manual download is required for training.

## Datasets Published

No new dataset was created or published for this project — we trained on the publicly
available CESNET-QUIC22. The only derived artifacts are **transient feature tensors**
computed on the fly by `src/feature_engineering.py`; these are not stored or redistributed.

- 🔗 **Published dataset:** _N/A (no dataset published)_

> If you later export a curated/derived sample (e.g. a small benchmark split of
> processed embeddings), publish it on Hugging Face under Creative Commons / Open Data
> Commons and add the link here. A helper exists at `scripts/export_cesnet_sample.py`.

## Licensing summary

- **Our code:** open source (repository under its stated license).
- **Our model:** to be published open-weight (Apache-2.0 / MIT recommended).
- **Training data:** CESNET-QUIC22 under its open Creative-Commons-style license; usage
  complies with the dataset's terms.
- **Dependencies:** all OSS — see [OSS_LIBRARIES.md](OSS_LIBRARIES.md) for per-library licenses.
