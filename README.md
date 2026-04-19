# nemotron_rl_approach

Modular GRPO training for NVIDIA Nemotron, runnable on Kaggle.

## One-time setup

### 1. Connect the repo to Kaggle (automated sync)

Every push to `main` that touches `nemotron_grpo/`, `notebooks/`, or `pyproject.toml` will automatically update the Kaggle dataset via the GitHub Action in `.github/workflows/kaggle-sync.yml`.

**One-time bootstrap:**

1. Edit `dataset-metadata.json` — replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username.
2. Create the dataset on Kaggle first (it must exist before the action can update it):
   - Kaggle → **Create → New Dataset** → name it `nemotron-rl-approach`.
3. Add two secrets to the GitHub repo (**Settings → Secrets and variables → Actions**):
   - `KAGGLE_USERNAME` — your Kaggle username.
   - `KAGGLE_KEY` — your Kaggle API key (from **kaggle.com → Account → API → Create New Token**).
4. Push to `main` — the action will run and upload `nemotron_grpo/`, `notebooks/`, `pyproject.toml`, and `CLAUDE.md` (`.kaggleignore` excludes `.git`, `.venv`, `inputs/`, `outputs/`, and lock files).

After this, every push to `main` keeps the Kaggle dataset in sync automatically. You can also trigger it manually from **GitHub → Actions → Sync to Kaggle Dataset → Run workflow**.

### 2. Prepare the offline TRL dataset

On a machine with internet access:

```bash
mkdir trl_packages
pip download trl -d trl_packages
```

Upload the `trl_packages/` folder as a Kaggle dataset named `trl-offline` (path becomes `/kaggle/input/trl-offline/trl_packages`).

### 3. Prepare the offline W&B dataset

On a machine with internet access:

```bash
mkdir wandb_packages
pip download wandb -d wandb_packages
```

Upload the `wandb_packages/` folder as a Kaggle dataset named `wandb-offline` (path becomes `/kaggle/input/wandb-offline/wandb_packages`).

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
