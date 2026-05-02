# Aquin Tests - QLoRA Fine-Tuning

A simple and lightweight QLoRA training pipeline test for fine-tuning causal LLMs (e.g. LLaMA Instruct) using 4-bit quantization and LoRA adapters, with optional Aquin session tracking.

---

## Files

| File | Purpose |
|------|---------|
| `train_qlora_aquin.py` | Core trainer - model loading, LoRA setup, training loop, generation, and save |
| `run_training.py` | Entry point - configure and kick off a training run |
| `data_loader.py` | Loads a `.jsonl` dataset and formats it into LLaMA Instruct chat template |
| `aquin_session.py` | Helper for wrapping a training run in an Aquin session |

---

## Setup

```bash
pip install torch transformers peft bitsandbytes accelerate aquin[inspect]
```

---

## Dataset Format

The pipeline expects a `.jsonl` file where each line is:

```json
{"prompt": "Your question here", "completion": "The expected answer here"}
```

Each row gets formatted into the LLaMA Instruct chat template automatically. Only the completion tokens contribute to the loss (prompt tokens are masked).

---

## `QLoRAConfig`

Dataclass that holds all training hyperparameters. Pass an instance to `QLoRATrainer`.

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | *(required)* | HuggingFace model ID or local path |
| `r` | `16` | LoRA rank |
| `alpha` | `32` | LoRA scaling factor |
| `dropout` | `0.05` | LoRA dropout |
| `lr` | `2e-4` | AdamW learning rate |
| `max_length` | `128` | Max token length per sample |
| `batch_size` | `2` | Per-step batch size |
| `grad_accum_steps` | `4` | Gradient accumulation steps - effective batch = `batch_size * grad_accum_steps` |
| `warmup_ratio` | `0.10` | Fraction of total steps used for LR warmup |
| `target_modules` | `("q_proj", "v_proj")` | Which weight matrices to apply LoRA to |
| `epochs` | `5` | Number of training epochs |

---

## `TextDataset`

Wraps a list of strings or dicts into a PyTorch dataset with tokenization and loss masking.

**Constructor**
```python
TextDataset(data, tokenizer, max_length)
```
- `data` - list of strings, or list of dicts with a `"text"` key and optional `"response_start_marker"` key
- `tokenizer` - HuggingFace tokenizer
- `max_length` - sequences are padded/truncated to this length

**`__getitem__(idx)`**

Returns a tokenized sample with `input_ids`, `attention_mask`, and `labels`. Two masking rules are applied:
1. Padding tokens are masked from the loss (`labels = -100`)
2. If `response_start_marker` is provided, all tokens before the marker are also masked - so only the completion contributes to the loss

---

## `QLoRATrainer`

Main class. Handles model setup, data loading, training, generation, and saving.

**Constructor**
```python
QLoRATrainer(cfg: QLoRAConfig)
```

Initializes in three stages:
1. Loads the base model in **4-bit NF4** quantization (bfloat16 compute, double quantization) via BitsAndBytes
2. Prepares the model for QLoRA: enables gradient checkpointing, wraps with `LoraConfig` via PEFT
3. Creates an **AdamW** optimizer with weight decay `0.01` over trainable parameters only

---

### `build_loader(dataset)`

```python
trainer.build_loader(dataset: List) -> DataLoader
```

Wraps `dataset` in a `TextDataset` and returns a shuffled `DataLoader` using the batch size from config.

---

### `setup_scheduler(total_steps)`

```python
trainer.setup_scheduler(total_steps: int)
```

Attaches a cosine LR schedule with linear warmup to the optimizer. `total_steps` should be `len(loader) * epochs` - **not** divided by `grad_accum_steps`. Warmup length is `total_steps * warmup_ratio`.

---

### `train_step(batch, global_step)`

```python
trainer.train_step(batch: dict, global_step: int) -> float
```

Runs a single forward + backward pass.

- Scales loss by `1 / grad_accum_steps` before backprop
- Only calls `optimizer.step()` and `scheduler.step()` when `(global_step + 1) % grad_accum_steps == 0`
- Clips gradients to `max_norm=1.0` before each optimizer step
- Clears CUDA cache every 50 steps
- Returns the unscaled loss value

---

### `train(dataset, session, epochs)`

```python
trainer.train(dataset, session=None, epochs=None) -> model
```

Main training loop.

- `dataset` - list of raw samples (passed to `build_loader`)
- `session` - optional Aquin session; if provided, `session.step(loss)` is called after each batch
- `epochs` - overrides `cfg.epochs` if provided

Prints average loss per epoch. Returns the trained model.

---

### `generate(prompt, max_new_tokens)`

```python
trainer.generate(prompt: str, max_new_tokens: int = 128) -> str
```

Runs inference on a prompt. Wraps the prompt in the LLaMA Instruct chat template before tokenizing. Uses greedy decoding (`do_sample=False`).

---

### `save(path)`

```python
trainer.save(path: str)
```

Saves the LoRA adapter weights and tokenizer to `path` using `save_pretrained`.

---

## `load_jsonl(path)`

```python
load_jsonl(path: str) -> List[dict]
```

Reads a `.jsonl` file and returns a list of formatted samples ready for `QLoRATrainer`.

- Skips blank lines and lines with invalid JSON (prints a warning)
- Formats each `{"prompt", "completion"}` row into the LLaMA Instruct chat template
- Adds `response_start_marker` so `TextDataset` can mask the prompt from the loss
- Raises `ValueError` if no valid rows are found

---

## `run_with_aquin(trainer, model, optimizer, api_key, dataset)`

```python
run_with_aquin(trainer, model, optimizer, api_key, dataset)
```

Convenience wrapper that creates an Aquin session, runs training, and calls `session.stop()` when done. Sets `project="qlora-research"` and `run_name="experiment-1"` by default.

---

## Using with Aquin

[Aquin](https://aquinlabs.com) lets you stream training metrics, track gradients, and inspect your model live in a dashboard while training runs.

### Quick Setup

1. Go to [aquinlabs.com](https://aquinlabs.com) and open the **application**
2. Select **SDK** as your main mode
3. Paste your API key (found at **Settings -> API Keys**, prefix looks like `aq-...`) and hit enter
4. Start your training run - metrics stream in live automatically

That's it. No extra config, no log files to upload.

> For a full breakdown of every parameter and feature, see the [Aquin SDK docs](https://www.aquinlabs.com/research/sdk).

---

### What Gets Tracked

Every `session.step()` call streams the following to the dashboard:

| Signal | Description |
|--------|-------------|
| `loss` | Scalar loss per step |
| `learning_rate` | LR from the optimizer's first param group |
| `gradNorms` | Per-layer gradient norms |
| `weightNorms` | Per-layer weight norms |
| `deadLayers` | Any layer with a gradient norm of exactly 0 |
| `optimizerState` | Adam momentum and variance norms |
| `stepMs` | Wall time between steps |

On `session.stop()`, Aquin automatically runs **ModelDiff** (behavioral comparison vs. base) and **SAEDiff** (which internal features shifted) and posts the results to the dashboard.

---

### Code

```python
from aquin import attach_qlora

session = attach_qlora(
    model=trainer.model,
    optimizer=trainer.optimizer,
    api_key="aq-...",
    project="qlora-research",   # optional
    run_name="experiment-1",    # optional
)

trainer.train(dataset=dataset, session=session)

session.stop()
```

`attach_qlora()` tags the session as QLoRA in the dashboard so it renders the correct badge and signals. The `session` object is passed into `trainer.train()` and `session.step(loss)` is called automatically after each batch.

---

### Optional: Export to W&B or MLflow

Aquin supports forwarding the same step payload to external loggers via sinks:

```python
from aquin.sinks import WandbSink, MLflowSink

session.addSink(WandbSink(project="my-project"))
# or
session.addSink(MLflowSink(run_name="experiment-1"))
```

Add sinks before calling `trainer.train()`.

---

## Key Design Notes

- **4-bit NF4 quantization** via BitsAndBytes with double quantization and bfloat16 compute
- **Left-side padding** to keep sequences aligned with causal attention in the Instruct format
- **Prompt masking** - only response tokens are included in the loss, not the prompt
- **Cosine LR schedule** with warmup spans the full training run (not divided by grad accumulation steps)
- **Gradient checkpointing** enabled to reduce VRAM usage
