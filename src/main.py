import os
from aquin import attach_qlora
from train_qlora_aquin import QLoRAConfig, QLoRATrainer
from data_loader import load_jsonl

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
API_KEY = "aq-..."
DATA_PATH = "./src/dataset.jsonl"

# =========================================================
# LOAD JSONL DATASET
# =========================================================

dataset = load_jsonl(DATA_PATH)

# =========================================================
# TRAINER
# =========================================================

cfg = QLoRAConfig(
    model_name=MODEL_NAME,
    r=32,
    alpha=64,
    dropout=0.05,
    lr=1e-4,
    max_length=256,
    batch_size=2,
    grad_accum_steps=8,     # effective batch = 16
    warmup_ratio=0.10,
    target_modules=(
        "q_proj", "k_proj", "v_proj", "o_proj",
        # "gate_proj", "up_proj", "down_proj",
    ),
    epochs=3,
)

trainer = QLoRATrainer(cfg)

# =========================================================
# AQUIN SESSION
# =========================================================

session = attach_qlora(
    model=trainer.model,
    optimizer=trainer.optimizer,
    api_key=API_KEY,
)

# =========================================================
# TRAIN
# =========================================================

print("Training on JSONL dataset...")

trainer.train(dataset=dataset, session=session)

# =========================================================
# FINISH
# =========================================================

session.stop()
trainer.save("./qlora_adapter")
print("Done.")
