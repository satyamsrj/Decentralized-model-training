# 🚀 How to Use the Decentralized P2P Machine Learning Swarm

This guide provides end-to-end instructions for configuring, launching, and managing distributed model training across heterogeneous, decentralized nodes (laptops, workstations, cloud VMs).

---

## 📋 Table of Contents
1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Why & How Multi-Laptop Training Works](#2-why--how-multi-laptop-training-works)
3. [Multi-Laptop Real-World Walkthrough (3 Laptops on Different Networks)](#3-multi-laptop-real-world-walkthrough-3-laptops-on-different-networks)
4. [Quickstart (Single-Machine Simulation)](#4-quickstart-single-machine-simulation)
5. [Cluster Setup Workflow](#5-cluster-setup-workflow)
   - [Step A: Configure the Swarm (`swarm_config.yaml`)](#step-a-configure-the-swarm)
   - [Step B: Launch the Matchmaker Bootnode](#step-b-launch-the-matchmaker-bootnode)
   - [Step C: Prepare & Shard the Dataset](#step-c-prepare--shard-the-dataset)
   - [Step D: Launch Swarm Worker Nodes](#step-d-launch-swarm-worker-nodes)
6. [Training Custom Models](#6-training-custom-models)
7. [Monitoring & Metrics](#7-monitoring--metrics)
8. [Checkpointing & Resumption](#8-checkpointing--resumption)
9. [Production Deployment (Docker & Systemd)](#9-production-deployment-docker--systemd)
10. [Troubleshooting & Verification](#10-troubleshooting--verification)

---

## 1. Prerequisites & Installation

### Requirements
- **Python**: 3.10, 3.11, or 3.12
- **Operating System**: Linux, macOS, or Windows
- **Network**: Internet access (or local network for testing); NAT hole-punching works across standard residential routers and hotspots.

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/satyamsrj/Decentralized-model-training.git
cd Decentralized-model-training

# Create virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies in editable mode
pip install -e .

# (Optional) For 8-bit gradient compression on CUDA-capable GPUs:
pip install bitsandbytes
```

---

## 2. Why & How Multi-Laptop Training Works

### The Challenge of Training on a Single Laptop
- **Extreme Heat & Thermal Throttling**: Training complex neural networks on a single laptop pushes the CPU/GPU to 100% for hours, causing severe thermal throttling, slow fan speeds, or system crashes.
- **Limited VRAM / RAM**: Consumer laptops rarely have more than 6GB–16GB of VRAM. Large batch sizes or large models cause Out-Of-Memory (OOM) errors.
- **Time Inefficiency**: What takes 24 hours on one laptop can be completed in hours when distributed across multiple machines.

### How This Swarm Makes Multi-Laptop Training Seamless

```text
 ┌────────────────┐          ┌────────────────┐          ┌────────────────┐
 │    LAPTOP A    │          │    LAPTOP B    │          │    LAPTOP C    │
 │  (Home Wi-Fi)  │          │  (5G Hotspot)  │          │  (Cafe Wi-Fi)  │
 │  NVIDIA RTX GPU│          │   MacBook M-chip│         │ Intel Core CPU │
 └───────┬────────┘          └───────┬────────┘          └───────┬────────┘
         │                           │                           │
         │  1. Check out Shard #1    │  1. Check out Shard #2    │  1. Check out Shard #3
         ▼                           ▼                           ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │           Distributed Hash Table (DHT) & Cloud Shard Storage           │
 └────────────────────────────────┬───────────────────────────────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │ 2. Compute local forward & backward gradients   │
         │ 3. Quantize gradients to 8-bit (4x bandwidth ↓) │
         │ 4. P2P Gossip All-Reduce directly between nodes │
         ▼                                                 ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │           All Laptops Converge on the Same Global Model State           │
 └────────────────────────────────────────────────────────────────────────┘
```

1. **Combines Idle Compute Without Buying Servers**:
   - You, your teammates, or your friends can combine consumer laptops (a gaming laptop with an RTX GPU, a MacBook Air, and a work laptop with an Intel CPU) into a single virtual supercomputer.
2. **Heterogeneous Hardware Friendly (No Deadlocks)**:
   - Traditional PyTorch Distributed (DDP) forces fast nodes to freeze and wait for the slowest node on every single step.
   - In this swarm, work is divided into independent **shards**. The fast gaming laptop processes 5 shards while the slower laptop processes 1 shard. Both contribute proportionally to convergence without holding each other back!
3. **Fault-Tolerant (Laptops Can Disconnect Any Time)**:
   - If a laptop loses Wi-Fi connection, runs out of battery, or closes its lid, **the training job does NOT crash**.
   - Shards are leased with a timeout (default 300s). If a disconnected laptop fails to report a shard within the timeout, the DHT automatically returns the shard to the pool so another laptop can pick it up.
4. **Zero Port-Forwarding / NAT Hole-Punching**:
   - Residential routers and mobile hotspots normally block incoming connections.
   - This swarm uses `libp2p` and a lightweight bootnode to punch through home NAT firewalls and establish direct peer-to-peer tunnels between laptops across the internet.
5. **Bandwidth-Optimized (8-Bit Compression)**:
   - Gradients are quantized into 8-bit integers before being broadcast over the network. This reduces network payload by **75%**, allowing training over typical home Wi-Fi and 5G mobile connections without lag.

---

## 3. Multi-Laptop Real-World Walkthrough (3 Laptops on Different Networks)

Here is exactly how three people in different locations run training together:

### Setup Overview
- **Bootnode VM**: A cheap $5/month cloud VPS (e.g. DigitalOcean, Hetzner, AWS) running the bootnode on public IP `203.0.113.42`.
- **Laptop 1 (Alice at Home on Wi-Fi)**: Windows laptop with NVIDIA RTX 3060.
- **Laptop 2 (Bob on 5G Mobile Hotspot)**: MacBook Air with Apple Silicon.
- **Laptop 3 (Charlie at a Coffee Shop)**: Linux laptop with Intel CPU.

---

### Step 1: Start the Rendezvous Bootnode (On Cloud VM)
Run on the cloud VM once:
```bash
python -m p2p_ml_swarm.bootnode --port 38000 --host 0.0.0.0
```
Note the multiaddress printed in the console:
```text
/ip4/203.0.113.42/tcp/38000/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N
```

---

### Step 2: Share the `swarm_config.yaml`
Every laptop uses the exact same `config/swarm_config.yaml` pointing to the bootnode and cloud storage:

```yaml
swarm:
  bootnodes:
    - "/ip4/203.0.113.42/tcp/38000/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N"
  dht_prefix: "team_project_alpha"
  lease_timeout_seconds: 300
  target_batch_size: 1024

storage:
  provider: "r2"
  bucket_name: "team-ml-dataset"
  endpoint_url: "https://<account-id>.r2.cloudflarestorage.com"
  region_name: "auto"
  access_key_id: "TEAM_ACCESS_KEY"
  secret_access_key: "TEAM_SECRET_KEY"

training:
  learning_rate: 0.001
  batch_size: 64
  use_8bit_optimizer: true
```

---

### Step 3: Launch Training on Each Laptop

#### On Laptop 1 (Alice - Home Wi-Fi, GPU):
```bash
python examples/mnist/train.py --config config/swarm_config.yaml --device cuda
```

#### On Laptop 2 (Bob - 5G Hotspot, Mac):
```bash
python examples/mnist/train.py --config config/swarm_config.yaml --device cpu
```

#### On Laptop 3 (Charlie - Coffee Shop Wi-Fi, Linux):
```bash
python examples/mnist/train.py --config config/swarm_config.yaml --device cpu
```

---

### What You See on Your Screen
Once launched, the Rich terminal dashboard on every laptop lights up:
```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                 🌐 P2P ML SWARM ACTIVE STATUS                       ┃
┣━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Swarm Peers   ┃ 3 Active Nodes Connected in Mesh                    ┃
┃ Current Epoch ┃ 1 / 5                                               ┃
┃ Shard Leased  ┃ Shard #004 (Streaming from Cloudflare R2...)        ┃
┃ Rolling Loss  ┃ 0.2841                                              ┃
┃ Throughput    ┃ 412.5 samples/sec (3 nodes combined)                ┃
┗━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```
- Alice's laptop rapidly trains through shards #1, #4, #7 on GPU.
- Bob's laptop trains through shard #2 on Mac CPU.
- Charlie's laptop trains through shard #3 on Linux.
- After each batch, all three laptops exchange 8-bit compressed gradients over the P2P mesh network via gossip averaging.
- If Bob closes his MacBook lid, Alice and Charlie automatically detect the expired lease and finish the remaining shards.
- The resulting model weights are saved to `./checkpoints/latest_checkpoint.pt`.


---

## 4. Quickstart (Single-Machine Simulation)

You can test the entire pipeline on a single machine across separate terminal tabs using the included MNIST example:

### Terminal 1: Start the DHT Bootnode
```bash
python -m p2p_ml_swarm.bootnode --port 38000
```
*Output will display the local multiaddress (e.g. `/ip4/127.0.0.1/tcp/38000/p2p/Qm...`).*

### Terminal 2: Download & Shard MNIST
```bash
python examples/mnist/prepare_data.py --output-dir ./mnist_shards --shard-size-mb 10
```

### Terminal 3: Launch Worker 1
```bash
python examples/mnist/train.py --config config/swarm_config.yaml --epochs 5 --device cpu
```

### Terminal 4: Launch Worker 2 (Peer Collaboration)
```bash
python examples/mnist/train.py --config config/swarm_config.yaml --epochs 5 --device cpu
```
Both workers will join the same DHT mesh, coordinate shard leases, gossip gradient updates, and train collaboratively.

---

## 5. Cluster Setup Workflow

### Step A: Configure the Swarm
The authoritative configuration file is [`config/swarm_config.yaml`](file:///c:/Users/ASUS/Desktop/Decentralized-model-training/config/swarm_config.yaml):

```yaml
swarm:
  bootnodes:
    - "/ip4/YOUR_BOOTNODE_IP/tcp/38000/p2p/YOUR_BOOTNODE_PEER_ID"
  dht_prefix: "swarm_experiment_v1"
  lease_timeout_seconds: 300
  target_batch_size: 1024

storage:
  provider: "r2"  # or "s3"
  bucket_name: "p2p-ml-shards"
  endpoint_url: "https://<account-id>.r2.cloudflarestorage.com"
  region_name: "auto"
  access_key_id: "YOUR_ACCESS_KEY"
  secret_access_key: "YOUR_SECRET_KEY"

training:
  learning_rate: 0.001
  batch_size: 64
  clip_grad_norm: 1.0
  use_8bit_optimizer: true

checkpointing:
  save_frequency_epochs: 1
  keep_last_n: 3
  checkpoint_dir: "./checkpoints"

monitoring:
  enable_rich: true
  enable_tensorboard: true
  tensorboard_dir: "./runs"
```

> **Tip:** You can override any setting using environment variables without modifying YAML files:
> ```bash
> export SWARM_STORAGE_BUCKET_NAME="my-production-bucket"
> export SWARM_TRAINING_LEARNING_RATE="0.0005"
> ```

---

### Step B: Launch the Matchmaker Bootnode
The bootnode acts as the public rendezvous point for NAT hole-punching.

- **On a Public VPS (DigitalOcean / AWS / Hetzner)**:
  ```bash
  python -m p2p_ml_swarm.bootnode --port 38000 --host 0.0.0.0
  ```
- Copy the displayed multiaddress and add it to `swarm.bootnodes` in `config/swarm_config.yaml`.

---

### Step C: Prepare & Shard the Dataset
Partition your dataset into self-contained binary shards (`.pt` chunks) and upload them to cloud storage:

```bash
# Prepare and upload directly to Cloudflare R2 / AWS S3:
python examples/mnist/prepare_data.py \
    --output-dir ./mnist_shards \
    --shard-size-mb 10 \
    --upload \
    --config config/swarm_config.yaml
```

---

### Step D: Launch Swarm Worker Nodes
Distribute the code and `config/swarm_config.yaml` to each participating node.

Run the worker on each node:
```bash
# On Laptop A (e.g., CUDA GPU):
python examples/mnist/train.py --config config/swarm_config.yaml --device cuda

# On Laptop B (e.g., CPU or Apple Silicon):
python examples/mnist/train.py --config config/swarm_config.yaml --device cpu
```

Each worker will:
1. Connect to the DHT mesh and discover active peers.
2. Check out an unassigned shard lease.
3. Stream the shard into memory.
4. Perform local forward/backward passes.
5. Quantize gradients to 8-bit integers and gossip with peers via All-Reduce.
6. Commit shard completion back to the DHT.

---

## 6. Training Custom Models

To train your own architecture, pass your model and shard directory to [`SwarmWorker`](file:///c:/Users/ASUS/Desktop/Decentralized-model-training/p2p_ml_swarm/worker.py#L32):

```python
import torch
import torch.nn as nn
from p2p_ml_swarm.worker import SwarmWorker

# 1. Define your custom architecture
class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        return self.fc(x)

model = MyTransformer()

# 2. Instantiate and run SwarmWorker
worker = SwarmWorker(
    config_path="config/swarm_config.yaml",
    model=model,
    device="cuda" if torch.cuda.is_available() else "cpu",
    total_epochs=10,
    criterion=nn.CrossEntropyLoss(),
    tensorboard_dir="./runs"
)

# 3. Start collaborative training
worker.run(shard_keys_or_dir="./my_shards")
```

---

## 7. Monitoring & Metrics

### Real-Time Terminal Dashboard
During training, a live Rich dashboard displays:
- Active peer count in the mesh
- Current epoch and shard lease status
- Rolling average loss and throughput (samples/sec)
- Gradient synchronization latency

### TensorBoard Visualizations
Launch TensorBoard to inspect loss curves, learning rates, and throughput:
```bash
tensorboard --logdir ./runs --port 6006
```
Open [http://localhost:6006](http://localhost:6006) in your browser.

---

## 8. Checkpointing & Resumption

- **Automatic Checkpointing**: Checkpoints are saved atomically at the interval specified in `checkpointing.save_frequency_epochs`.
- **Resuming a Session**: Pass `--resume` to continue training from the most recent checkpoint:
  ```bash
  python examples/mnist/train.py --config config/swarm_config.yaml --resume
  ```

---

## 9. Production Deployment (Docker & Systemd)

### Running the Bootnode with Docker
```bash
docker build -f deployment/docker/Dockerfile.bootnode -t p2p-ml-bootnode .
docker run -d --name bootnode -p 38000:38000 p2p-ml-bootnode
```

### Running the Bootnode as a Linux Systemd Service
```bash
sudo cp deployment/systemd/p2p-ml-bootnode.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now p2p-ml-bootnode
sudo systemctl status p2p-ml-bootnode
```

### Running Workers in Docker
```bash
docker build -f deployment/docker/Dockerfile -t p2p-ml-worker .
docker run --rm -v $(pwd)/config:/app/config p2p-ml-worker \
    python examples/mnist/train.py --config /app/config/swarm_config.yaml
```

---

## 10. Troubleshooting & Verification

### Run the Diagnostic Test Suite
Verify that all subsystems (storage, DHT, optimizer, checkpointing, pipeline) function correctly:
```bash
python -m pytest tests/ -v
```
*(All 67 tests should pass with green status)*

### Query the Knowledge Graph Memory
Use Graphify to inspect codebase symbols and dependency call flows:
```bash
# Query the architecture
graphify query "SwarmWorker"

# View call flow between components
graphify path "SwarmWorker" "SwarmTrainer"

# Open interactive visualization
open graphify-out/graph.html  # On Windows: start graphify-out/graph.html
```

---

## 11. Local ML Cluster GUI (Desktop Application)

The **Local ML Cluster** desktop application provides an intuitive graphical interface to coordinate distributed model training across multiple laptops without needing complex terminal configurations.

### Launching the Desktop GUI

Run from your terminal:
```bash
python -m p2p_ml_swarm.gui
```
Or launch programmatically in Python:
```python
from p2p_ml_swarm.gui import launch_gui

launch_gui()
```

### Connection Modes

1. **Online / VM Mode**:
   - Laptops connect over a secure virtual private overlay network (Tailscale or WireGuard).
   - An authoritative cloud control plane (`python -m p2p_ml_swarm.online_server.api`) coordinates enrollment, health checks, and run generations.
   - Tensor gradients flow directly peer-to-peer between workers via PyTorch DDP TCP rendezvous.
2. **Offline / USB Mode**:
   - Laptops are connected directly using USB-to-Ethernet or USB-C bridge cables on a shared link-local subnet (e.g. `169.254.0.0/16` or `192.168.2.0/24`).
   - Zero Internet connection, cloud storage, or external VM is required.
   - Pre-installed model templates and synthetic offline datasets ensure immediate training readiness anywhere.

### 7-Step GUI Workflow

1. **Home Screen**: Displays local node status, OS, CPU cores, RAM, and detected CUDA GPU.
2. **Connection Mode**: Choose between **Online (VM)** or **Offline (USB)**.
3. **Worker Pairing**: Generate a 6-character cryptographic pairing code (e.g., `K9X2P4`) on the controlling laptop. Peer laptops enter this code to securely enroll and receive their rank assignment (Rank 0 = Coordinator).
4. **Model & Dataset**: Select curated templates (`mnist_cnn`, `fashion_mnist_cnn`, `cifar10_convnet`, `tabular_mlp`), configure batch size, learning rate, and epochs.
5. **Preflight Safety Gate**: Automated verification tests network latency, worker counts, minimum 2GB free disk space, and CUDA/CPU device compatibility before launching.
6. **Live Training Dashboard**: Real-time progress bars, epoch tracking, rolling loss metrics, sample throughput, and live node logs.
7. **Results & Artifact Export**: View training duration, final loss, and export SHA-256 verified model checkpoint manifests.

