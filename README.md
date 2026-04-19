# nemotron_rl_approach

Modular GRPO training for NVIDIA Nemotron, runnable on Kaggle.

---

## Two independent workflows

| What | How |
|---|---|
| Keep code versioned | `git push` to GitHub as normal |
| Sync dataset + kernel to Kaggle | `./sync_kaggle.sh` from the repo root |

They are **not linked** — pushing to GitHub does not update Kaggle. Run the sync manually whenever you want Kaggle to reflect your latest changes.

---

## Syncing to Kaggle

### One-time setup

1. Get your Kaggle API credentials: kaggle.com → **Settings → API → Create New Token** → downloads `kaggle.json`.

2. Place the credentials file:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Install the Kaggle CLI (already a dependency — `uv sync` is enough):
   ```bash
   uv sync
   ```

4. Create the dataset on Kaggle the **first time only**:
   ```bash
   kaggle datasets create -p . --dir-mode tar
   ```
   This reads `dataset-metadata.json` from the repo root and creates `zosov07/nemotron-rl-approach`.

5. Prepare the offline package datasets (one-time, needed for Kaggle notebooks without internet):
   ```bash
   # TRL
   mkdir trl_packages && pip download trl -d trl_packages/
   cat > trl_packages/dataset-metadata.json << 'EOF'
   {"title":"trl-offline","id":"zosov07/trl-offline","licenses":[{"name":"CC0-1.0"}]}
   EOF
   kaggle datasets create -p trl_packages/ --dir-mode tar

   # W&B
   mkdir wandb_packages && pip download wandb -d wandb_packages/
   cat > wandb_packages/dataset-metadata.json << 'EOF'
   {"title":"wandb-offline","id":"zosov07/wandb-offline","licenses":[{"name":"CC0-1.0"}]}
   EOF
   kaggle datasets create -p wandb_packages/ --dir-mode tar
   ```

### sync_kaggle.sh — usage

Run from the repo root:

```bash
./sync_kaggle.sh                           # sync both dataset + experiment kernel (default)
./sync_kaggle.sh "message"                 # same, with a version message
./sync_kaggle.sh "message" --dataset       # dataset only
./sync_kaggle.sh "message" --experiment    # experiment kernel only
```

**What each target does:**

| Flag | Action |
|---|---|
| *(default / `--both`)* | Updates the dataset version, then pushes the experiment kernel |
| `--dataset` | Uploads `nemotron_grpo/`, `notebooks/`, config files as a new version of `zosov07/nemotron-rl-approach` |
| `--experiment` | Pushes `notebooks/nemotron-experiment/` as a new version of the `zosov07/nvidia-nemotron-experiment` kernel |

The experiment kernel reads its sources from `notebooks/nemotron-experiment/kernel-metadata.json`, which declares:
- **Datasets**: `zosov07/nemotron-rl-approach`, `zosov07/trl-offline`, `zosov07/wandb-offline`
- **Competition**: `nvidia-nemotron-3-reasoning-challenge`
- **Model**: `metric/nemotron-3-nano-30b-a3b-bf16`

### Pulling and modifying an existing Kaggle kernel locally

```bash
# Pull kernel + metadata into a local folder
kaggle kernels pull zosov07/nvidia-nemotron-training -p notebooks/nemotron-training --metadata

# Edit the notebook, then push back
(cd notebooks/nemotron-training && kaggle kernels push -p .)
```

### Setting the W&B API key (online mode only)

By default W&B runs **offline** — no API key needed. To stream metrics live:

1. Get your key from <https://wandb.ai/authorize>.
2. In the Kaggle notebook → **Add-ons → Secrets** → **Add a new secret**:
   - Name: `WANDB_API_KEY`
   - Value: your key
   - Toggle it **on** for the notebook.
3. In the notebook config, set:
   ```python
   config = GRPOExperimentConfig(
       ...
       wandb_mode="online",
   )
   ```
4. Enable **Internet: On** for the notebook session (Kaggle sidebar → **Internet**).

The code reads the key automatically via `kaggle_secrets.UserSecretsClient` when `wandb_mode="online"` — nothing else to change.

### Pushing to GitHub

Independent from Kaggle — normal git workflow:

```bash
git add .
git commit -m "your message"
git push origin main
```

---

## Running on Kaggle

Open `notebooks/nemotron-experiment/nvidia-nemotron-experiment.ipynb` on Kaggle. All data sources are declared in `kernel-metadata.json` and attached automatically when pushing via the CLI.

Set **Accelerator → GPU**. Internet can stay **OFF** — W&B runs in offline mode by default.

Output: LoRA adapter in `/kaggle/working/grpo_run/`, zipped as `submission.zip`.

---

## Syncing offline W&B runs

Download `/kaggle/working/wandb/` from the notebook outputs, then on any machine with internet:

```bash
wandb sync /path/to/wandb/offline-run-*
```

---

## Experimenting

Edit `GRPOExperimentConfig` in the notebook to change hyperparameters or swap reward functions:

```python
config = GRPOExperimentConfig(
    model_path="...",
    reward_functions=["accuracy_reward"],  # or: cosine_reward, format_reward, length_reward
    learning_rate=5e-6,
    max_grad_norm=0.1,                     # optional — leave out to use TRL default
    wandb_run_name="lr5e-6-accuracy",
)
```

To add a new reward function: one entry in `nemotron_grpo/rewards.py::REGISTRY`.

---

## Local development

```bash
uv sync
uv pip install -e .
python -c "from nemotron_grpo import GRPOExperimentConfig; print('ok')"
python -c "from nemotron_grpo.rewards import REGISTRY; print(list(REGISTRY))"
```
