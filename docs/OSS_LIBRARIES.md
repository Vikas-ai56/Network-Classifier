# Open-Source Libraries & Projects Used

All components of this project are open source. Below is every library/project we depend
on, what we use it for, its license, and a link to its source.

## Deep learning & modelling

| Library | Purpose in this project | License | Link |
|---|---|---|---|
| **PyTorch** | Core DL framework — model, autograd, AMP training, inference | BSD-3-Clause | https://github.com/pytorch/pytorch |
| **torchvision / torchaudio** | PyTorch companion packages (tensor utils, transforms) | BSD-3-Clause | https://github.com/pytorch/vision · https://github.com/pytorch/audio |
| **mamba-ssm** | Mamba State-Space Model sequence encoder (Branch A, primary path) | Apache-2.0 | https://github.com/state-spaces/mamba |
| **causal-conv1d** | CUDA causal conv kernels required by Mamba | BSD-3-Clause | https://github.com/Dao-AILab/causal-conv1d |
| **einops** | Tensor rearrangement in model code | MIT | https://github.com/arogozhnikov/einops |
| **timm** | Layer utilities / scheduling helpers (NetMamba lineage) | Apache-2.0 | https://github.com/huggingface/pytorch-image-models |
| **transformers** | Tokenizer/utility deps inherited from upstream NetMamba code | Apache-2.0 | https://github.com/huggingface/transformers |

## Retrieval / classification / metrics

| Library | Purpose | License | Link |
|---|---|---|---|
| **FAISS** | Exact inner-product (= cosine) k-NN over embeddings, per-epoch eval | MIT | https://github.com/facebookresearch/faiss |
| **scikit-learn** | k-NN, Logistic Regression, SVM, classification reports, splits | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn |
| **SciPy** | Numerical routines used by scikit-learn | BSD-3-Clause | https://github.com/scipy/scipy |

## Data pipeline

| Library | Purpose | License | Link |
|---|---|---|---|
| **cesnet-datazoo** | Streams CESNET-QUIC22 flows chunk-by-chunk | BSD-3-Clause | https://github.com/CESNET/cesnet-datazoo |
| **NumPy** | All array math and feature vectors | BSD-3-Clause | https://github.com/numpy/numpy |
| **pandas** | DataFrame handling of flow records | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| **PyArrow** | Reading CESNET parquet partitions | Apache-2.0 | https://github.com/apache/arrow |
| **nfstream** | PCAP → flow feature extraction (`src/nfstream_extractor.py`, live-capture path) | LGPL-3.0 | https://github.com/nfstream/nfstream |
| **scapy** | Low-level packet parsing utilities | GPL-2.0 | https://github.com/secdev/scapy |

## Training infrastructure / tooling

| Library | Purpose | License | Link |
|---|---|---|---|
| **Weights & Biases** | Experiment logging (run in offline mode) | MIT | https://github.com/wandb/wandb |
| **MLflow** | Optional experiment tracking (inherited dependency) | Apache-2.0 | https://github.com/mlflow/mlflow |
| **tqdm** | Progress bars | MPL-2.0 / MIT | https://github.com/tqdm/tqdm |
| **matplotlib** | Plots / visualizations (e.g. `phist_visualization.png`) | PSF/BSD-style | https://github.com/matplotlib/matplotlib |
| **tensorboard** | Optional metric visualization | Apache-2.0 | https://github.com/tensorflow/tensorboard |

## Reference projects / prior art we built on

| Project | How we used it | Link |
|---|---|---|
| **NetMamba** | Architectural inspiration for the Mamba-based traffic encoder; some scaffolding in `src/models_net_mamba.py`, `src/models_mamba.py`, and `src/util/` derives from this lineage | https://github.com/wangtz19/NetMamba |
| **Mamba (state-spaces)** | The State-Space Model that powers the primary sequence branch | https://github.com/state-spaces/mamba |
| **Supervised Contrastive Learning** | Loss-function basis adapted into our `MarginBasedSupConLoss` | https://github.com/HobbitLong/SupContrast |

> A complete pinned dependency list is in [`../requirements.txt`](../requirements.txt).
> Note: `cesnet-datazoo` and `faiss-gpu` are installed separately on the GPU instance
> (see [INSTALLATION.md](INSTALLATION.md)); they are not in the inherited pinned list.
