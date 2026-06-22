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

We developed the **DualBranchEncoder** as our solution. Publish the trained checkpoint
(`best_model_v1_3of5.pth`, ~8 MB, ~1.98 M params) to Hugging Face under an open license
(recommended: **Apache-2.0** or **MIT**).

- 🔗 **Hugging Face model:** _<add link after upload, e.g. https://huggingface.co/<user>/dualbranch-quic-encoder>_

How to publish (run locally where the checkpoint is):
```bash
pip install huggingface_hub
huggingface-cli login
python - <<'PY'
from huggingface_hub import HfApi, create_repo
repo = "<user>/dualbranch-quic-encoder"
create_repo(repo, repo_type="model", exist_ok=True)
HfApi().upload_file(
    path_or_fileobj="best_model_v1_3of5.pth",
    path_in_repo="best_model.pth",
    repo_id=repo, repo_type="model",
)
print("uploaded to", repo)
PY
```
Include a model card stating: architecture (DualBranchEncoder, 256-d L2-normalized
embedding), training data (CESNET-QUIC22 XS), intended use (encrypted traffic
classification), KPIs (see [../KPI_RESULTS.md](../KPI_RESULTS.md)), and license.

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
