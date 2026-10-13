# 🚀 How to Use the Decentralized P2P Machine Learning Swarm

This guide provides end-to-end instructions for configuring, launching, and managing distributed model training across heterogeneous, decentralized nodes (laptops, workstations, cloud VMs).

---

## 📋 Table of Contents
1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Quickstart (Single-Machine Simulation)](#2-quickstart-single-machine-simulation)
3. [Cluster Setup Workflow](#3-cluster-setup-workflow)
   - [Step A: Configure the Swarm (`swarm_config.yaml`)](#step-a-configure-the-swarm)
   - [Step B: Launch the Matchmaker Bootnode](#step-b-launch-the-matchmaker-bootnode)
   - [Step C: Prepare & Shard the Dataset](#step-c-prepare--shard-the-dataset)
   - [Step D: Launch Swarm Worker Nodes](#step-d-launch-swarm-worker-nodes)
4. [Training Custom Models](#4-training-custom-models)
5. [Monitoring & Metrics](#5-monitoring--metrics)
6. [Checkpointing & Resumption](#6-checkpointing--resumption)
7. [Production Deployment (Docker & Systemd)](#7-production-deployment-docker--systemd)
8. [Troubleshooting & Verification](#8-troubleshooting--verification)

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

## 2. Quickstart (Single-Machine Simulation)

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

## 3. Cluster Setup Workflow

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

## 4. Training Custom Models

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

## 5. Monitoring & Metrics

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

## 6. Checkpointing & Resumption

- **Automatic Checkpointing**: Checkpoints are saved atomically at the interval specified in `checkpointing.save_frequency_epochs`.
- **Resuming a Session**: Pass `--resume` to continue training from the most recent checkpoint:
  ```bash
  python examples/mnist/train.py --config config/swarm_config.yaml --resume
  ```

---

## 7. Production Deployment (Docker & Systemd)

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

## 8. Troubleshooting & Verification

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
