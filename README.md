# nemotron_rl_approach

Modular GRPO training for NVIDIA Nemotron, runnable on Kaggle.

## One-time setup

### 1. Connect the repo to Kaggle (automated sync)

Every push to `main` that touches `nemotron_grpo/`, `notebooks/`, or `pyproject.toml` will automatically update the Kaggle dataset via the GitHub Action in `.github/workflows/kaggle-sync.yml`.

**One-time bootstrap:**

**a) Configure Kaggle credentials locally**

Download `kaggle.json` from kaggle.com → Account → API → **Create New Token**, then:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

The `kaggle` CLI is already a project dependency — install it with:

```bash
uv sync
```

**b) Create the dataset on Kaggle**

The dataset must exist before the Action can update it. Run once from the repo root (`dataset-metadata.json` is read automatically):

```bash
kaggle datasets create -p .
```

This creates the dataset at `zosov07/nemotron-rl-approach`. Content uploaded here doesn't matter — the Action will overwrite it on the next push.

**c) Add secrets to the GitHub repo**

Go to **Settings → Secrets and variables → Actions → New repository secret** and add:

- `KAGGLE_USERNAME` → your Kaggle username
- `KAGGLE_KEY` → the `key` value from your `kaggle.json`

**d) Push to trigger the first sync**

```bash
git add .
git commit -m "initial project structure"
git push origin main
```

The Action runs and uploads `nemotron_grpo/`, `notebooks/`, `pyproject.toml`, and `CLAUDE.md`. Files listed in `.kaggleignore` (`.venv`, `inputs/`, `outputs/`, `uv.lock`, etc.) are excluded.

After this, every push to `main` keeps the Kaggle dataset in sync automatically. You can also trigger it manually from **GitHub → Actions → Sync to Kaggle Dataset → Run workflow**.

### 2. Prepare the offline TRL dataset

```bash
mkdir -p trl-offline/trl_packages
pip download trl -d trl-offline/trl_packages

# Create dataset-metadata.json for this dataset
echo '{"title":"trl-offline","id":"zosov07/trl-offline","licenses":[{"name":"CC0-1.0"}]}' \
  > trl-offline/dataset-metadata.json

kaggle datasets create -p trl-offline
```

Mounted in Kaggle at `/kaggle/input/trl-offline/trl_packages`.

### 3. Prepare the offline W&B dataset

```bash
mkdir -p wandb-offline/wandb_packages
pip download wandb -d wandb-offline/wandb_packages

echo '{"title":"wandb-offline","id":"zosov07/wandb-offline","licenses":[{"name":"CC0-1.0"}]}' \
  > wandb-offline/dataset-metadata.json

kaggle datasets create -p wandb-offline
```

Mounted in Kaggle at `/kaggle/input/wandb-offline/wandb_packages`.

### 4. Attach the base model and competition data

The following should already be attached or searchable on Kaggle:

- **Model**: `metric/nemotron-3-nano-30b-a3b-bf16` → mounted at `/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`
- **Competition data**: `nvidia-nemotron-3-reasoning-challenge` → `train.csv` at `/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv`

### 5. (Optional) Online W&B logging

By default W&B runs in **offline** mode (no internet needed). To enable live logging:

1. Create a W&B account and copy your API key from <https://wandb.ai/authorize>.
2. In the notebook → **Add-ons → Secrets** → add a secret named `WANDB_API_KEY` with that value, toggled on.
3. In the notebook config, set `wandb_mode="online"`.
4. Enable **Internet: On** for the notebook session.

## Running on Kaggle

1. Open `notebooks/nvidia-nemotron-training.ipynb` on Kaggle.
2. **Add-ons → Add data** — attach all datasets:
   - `nemotron-rl-approach` (this repo)
   - `trl-offline`
   - `wandb-offline`
   - `nvidia-nemotron-3-reasoning-challenge`
   - The Nemotron model (via Models tab)
3. Set **Accelerator → GPU** (T4×2 or P100). Internet can stay **OFF** — W&B defaults to offline mode.
4. Run all cells top-to-bottom.
5. Final output: LoRA adapter in `/kaggle/working/grpo_run/`, zipped as `submission.zip`.

## Syncing offline W&B runs

After downloading `/kaggle/working/wandb/` from the notebook output, run on any internet-connected machine with `wandb` installed:

```bash
wandb sync /path/to/downloaded/wandb/offline-run-*
```

## Experimenting

Edit the `GRPOExperimentConfig` in the notebook to change hyperparameters or swap reward functions:

```python
config = GRPOExperimentConfig(
    model_path="...",
    reward_functions=["cosine_reward"],        # try a single reward
    learning_rate=5e-6,                        # override a default
    max_grad_norm=0.1,                         # enable an optional kwarg
    wandb_run_name="lr5e-6-cosine-only",
    wandb_tags=["ablation"],
)
```

Available reward functions: `cosine_reward`, `format_reward`, `length_reward`.

To add a new reward function, add one entry to `nemotron_grpo/rewards.py::REGISTRY`.

## Local development

```bash
uv sync
uv pip install -e .
python -c "from nemotron_grpo import GRPOExperimentConfig; print('ok')"
python -c "from nemotron_grpo.rewards import REGISTRY; print(list(REGISTRY))"
```
