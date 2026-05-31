# Next-Generation Encrypted Network Traffic Classifier

## 📘 Full Codebase Documentation

For end-to-end repository architecture, entrypoints, folder responsibilities, flow mapping, and architecture change guidance, see:

- [docs/README.md](docs/README.md)
- [docs/ARCHITECTURE_CHANGE_CHECKLIST.md](docs/ARCHITECTURE_CHANGE_CHECKLIST.md)

Welcome to the core repository for our Hackathon Project! This repository contains a state-of-the-art deep learning architecture designed to classify encrypted network traffic in real-time. It focuses specifically on **Zero-Day Generalization**—the ability to accurately identify traffic types the model has never been explicitly trained on.

---

## 🚀 The Architecture in Detail

Traditional Deep Packet Inspection (DPI) fails entirely on encrypted traffic (like HTTPS or VPNs), and standard sequence models (like Transformers) suffer from quadratic complexity, making them too slow for real-time, packet-by-packet network analysis. 

Our solution is a **Hybrid Dual-Branch Encoder**:

### 1. Branch A: The Temporal Sequence Branch
Network traffic is a conversation over time. This branch processes the chronological sequence of the first 128 packets in a flow. 
* **Input Features:** `[Packet Size, Inter-Arrival Time (IAT)]`. We explicitly exclude IP addresses (the "5-tuple trap") so the model learns behavioral fingerprints, not routing topologies.
* **The Engine:** We utilize a **Mamba State Space Model (SSM)**. Mamba processes sequences in linear time $O(N)$, making it blazingly fast compared to Transformers. 
* *Fallback Strategy:* If deployed on a machine without CUDA/Mamba support, the architecture automatically falls back to a 2-layer Bidirectional LSTM.

### 2. Branch B: The Statistical Branch
Static metadata cannot be fed into a sequence model without causing gradient distortion.
* **Input Features:** Flow-level QoS scalars extracted at the end of the flow, such as `Total Bytes`, `Packet Rate`, `TCP RTT`, and `UDP Jitter`.
* **The Engine:** A standard Multi-Layer Perceptron (MLP) with ReLU activations and Dropout layers.

### 3. The Brain (Fusion & Projection)
The high-dimensional outputs from Branch A and Branch B are concatenated. They are then passed through a final MLP Projection Head to generate one unified, 128-dimensional **"Context-Aware Embedding"**. This single vector represents the entire "personality" of the network flow.

---

## 🧠 Training Strategy: SupCon + ProtoNet

We do not use standard Softmax classification, because it fails on zero-day traffic. Instead, we use:

* **Phase 1: Supervised Contrastive Learning (SupCon)**
  During pre-training, the model acts as a "Teacher." It uses `SupConLoss` to mathematically pull embeddings of the same traffic application (e.g., two different Skype calls) close together in the latent space, while forcefully pushing different applications (e.g., Skype vs. FTP) far apart.
  
* **Phase 2: Prototypical Networks (Zero-Day Evaluation)**
  During evaluation, we use **Episodic Sampling (N-way, K-shot)**. We pick 5 random classes and give the model 5 examples of each to compute a "Prototype" (the geometric center point of that class). We then give it 15 unseen flows and ask it to classify them based strictly on Euclidean distance to the prototypes. This is what enables zero-day detection!

---

## 📂 File-by-File Breakdown

### `main.py` (The Orchestrator)
This is the entry point of the application. When you run `python3 main.py`, it performs the following:
1. Initializes the `NetMambaDataset` to load the offline traffic data.
2. Initializes the `DualBranchEncoder` model and the `AdamW` optimizer.
3. Runs the **SupCon Pre-training Loop** (updating model weights).
4. Pauses at the end of every epoch to run the **Prototypical Zero-Day Evaluation**.
5. Automatically saves the weights to `model/best_model.pth` whenever it hits a new high score in zero-day accuracy.

### `src/models_dual_branch.py` (The Architecture)
Contains the PyTorch classes that build the neural network:
* `SequenceBranch`: Implements the Mamba (or fallback LSTM) logic. It takes the `(Batch, 128, 2)` sequence tensor and applies Global Average Pooling.
* `StatBranch`: Implements the MLP for the `(Batch, 4)` static scalar tensor.
* `DualBranchEncoder`: The parent class that initializes both branches, merges their outputs, and applies the L2-normalized Projection Head.

### `src/train_supcon.py` (The Math & Logic)
Contains the custom loss functions and samplers crucial for this project:
* `SupConLoss`: Calculates the contrastive loss using temperature-scaled dot products. 
* `EpisodicSampler`: Groups data into N-way, K-shot episodes instead of standard randomized batches.
* `compute_prototypes` & `prototypical_loss`: Calculates the geometric centers of traffic classes and classifies new traffic based on distance.

### `src/dataset_netmamba.py` (The Data Loader)
A custom PyTorch `Dataset` class designed to parse the offline ISCXVPN2016 dataset.
* It crawls the directory structure to find `.json` files, extracts lengths and intervals, and pads/truncates to exactly 128 packets.

### `src/nfstream_extractor.py` (The Real-World Extractor)
While `dataset_netmamba.py` is for offline JSONs, this script is for **production deployment**.
* Uses the `nfstream` library to ingest raw `.pcap` files or live network interfaces.
* Applies the critical guardrail of setting `splt_analysis=128` to extract the first 128 packets without IP addresses.

### `assets/` (Visual Documentation)
This folder contains architectural diagrams and visual aids (like `NetMamba.png`). These are useful for understanding the flow of the Mamba blocks and should be used when presenting the project.

### `requirements.txt` (Environment Dependencies)
This file lists all the Python dependencies required to run the original Mamba codebase. 
* **Key Dependencies:** `torch`, `torchvision`, `scikit-learn`, `pandas`.
* **Important Note:** You will also need to manually `pip install nfstream` to use the `nfstream_extractor.py` for live traffic.

---

## 🏆 Project Status: Achieved vs. Not Achieved

### ✅ What We Have Achieved So Far
1. **Built the Custom Architecture:** The entire Dual-Branch Mamba + MLP model is fully coded and functional with dynamic fallback support.
2. **Built the Data Pipeline:** The extraction logic successfully converts traffic into behavior sequences without data leakage (no IP addresses).
3. **Implemented Zero-Day Logic:** SupCon Loss and ProtoNet Evaluation are fully integrated.
4. **Successful Proof-of-Concept:** We trained the model on the ISCXVPN2016 benchmark dataset for 5 epochs. **Result:** It achieved **43.8% Zero-Day (Few-Shot) Accuracy**. Since random guessing for 5 classes is 20%, we have proven the model successfully learns behavioral fingerprints and is over 2x better than random on unseen data!

### 🚧 What is NOT Achieved (Next Steps for the Team)
**If you are picking up this project, here is what you need to work on next:**

1. **Activate the Mamba Engine (Hardware Requirement):** 
   * *Status:* The model is currently running on the **LSTM Fallback** because it was developed on a Mac. 
   * *Task:* Move this codebase to a Linux machine with an NVIDIA GPU. Run `pip install mamba-ssm`. The code will automatically detect it and switch to the ultra-fast linear Mamba engine.
2. **CESNET-QUIC22 Generalization Test:**
   * *Status:* We proved the concept on ISCXVPN2016, but the hackathon requires testing on CESNET-QUIC22 (a massive 153M flow dataset).
   * *Task:* Use the `cesnet-datazoo` library to download the QUIC dataset, train the model on it, and hit the >85% accuracy KPI.
3. **Build `inference.py` (The Production App):**
   * *Status:* We have the extractor (`nfstream_extractor.py`) and the trained model (`model/best_model.pth`), but they aren't tied together for a live demo.
   * *Task:* Write a script that takes a live `.pcap` capture, runs it through `NFStreamExtractor`, feeds it to the `DualBranchEncoder`, and prints the traffic classification to the terminal in real-time.

---

## 💻 How to Run

1. Ensure your offline dataset is unzipped into the `dataset/netmamba/ISCXVPN2016/images_sampled_new` directory. *(Note: The `dataset/` and `model/` folders are ignored in `.gitignore` due to their massive size).*
2. Run the main engine from the root of the project:

```bash
python3 main.py
```