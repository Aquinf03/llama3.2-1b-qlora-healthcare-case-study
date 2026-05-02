import torch
from dataclasses import dataclass
from typing import List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from torch.utils.data import DataLoader


# =========================================================
# CONFIG
# =========================================================

@dataclass
class QLoRAConfig:
    model_name: str
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    lr: float = 2e-4
    max_length: int = 128
    batch_size: int = 2
    grad_accum_steps: int = 4
    warmup_ratio: float = 0.10
    target_modules: tuple = (
        "q_proj", "v_proj",
    )
    epochs: int = 5


# =========================================================
# DATASET WRAPPER
# =========================================================

class TextDataset:
    """
    Accepts list[str] or list[dict] with optional response_start_marker.
    Masks padding tokens AND prompt tokens from loss.
    """

    def __init__(self, data: List, tokenizer, max_length: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        marker = None

        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or str(item)
            marker = item.get("response_start_marker")
        else:
            text = str(item)

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        tokens["labels"] = tokens["input_ids"].clone()

        # 1. Mask padding tokens
        pad_id = self.tokenizer.pad_token_id
        tokens["labels"][tokens["labels"] == pad_id] = -100

        # 2. Mask prompt - only compute loss on the response
        if marker:
            marker_ids = self.tokenizer.encode(
                marker, add_special_tokens=False)
            input_ids = tokens["input_ids"].tolist()
            marker_len = len(marker_ids)
            response_start = None
            for i in range(len(input_ids) - marker_len + 1):
                if input_ids[i: i + marker_len] == marker_ids:
                    response_start = i + marker_len
                    break
            if response_start is not None:
                tokens["labels"][:response_start] = -100

        return tokens


# =========================================================
# QLoRA TRAINER CORE
# =========================================================

class QLoRATrainer:
    def __init__(self, cfg: QLoRAConfig):
        self.cfg = cfg

        # 1. NF4 QUANTIZED BASE MODEL
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # ← CHANGED: pad on the right causes issues with Instruct chat format;
        #   padding on the left ensures the actual sequence aligns with the model's
        #   causal attention correctly
        self.tokenizer.padding_side = "left"

        # 2. PREP FOR QLoRA
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.alpha,
            lora_dropout=cfg.dropout,
            target_modules=list(cfg.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # 3. OPTIMIZER
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.lr,
            weight_decay=0.01,
        )

        self.scheduler = None

    # =========================================================
    # DATA LOADING
    # =========================================================

    def build_loader(self, dataset: List):
        ds = TextDataset(dataset, self.tokenizer, self.cfg.max_length)
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

    # =========================================================
    # TRAIN SETUP
    # =========================================================

    def setup_scheduler(self, total_steps: int):
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            # ← FIXED: use raw total steps so the cosine schedule spans the
            #   full training run, not just 1/grad_accum_steps of it
            num_training_steps=total_steps,
        )

    # =========================================================
    # TRAIN STEP
    # =========================================================

    def train_step(self, batch, global_step: int):
        self.optimizer.zero_grad()

        batch = {k: v.to(self.model.device) for k, v in batch.items()}
        outputs = self.model(**batch)

        if outputs.loss is None:
            return 0.0

        loss = outputs.loss / self.cfg.grad_accum_steps
        loss.backward()

        if (global_step + 1) % self.cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        if (global_step + 1) % 50 == 0:
            torch.cuda.empty_cache()

        return loss.item() * self.cfg.grad_accum_steps

    # =========================================================
    # MAIN TRAIN LOOP (AQUIN HOOK READY)
    # =========================================================

    def train(self, dataset, session=None, epochs: Optional[int] = None):
        epochs = epochs or self.cfg.epochs
        loader = self.build_loader(dataset)

        # ← FIXED: pass raw steps (len * epochs), NOT divided by grad_accum_steps.
        #   Previous version divided here, causing the LR to hit zero halfway
        #   through training and the model stopped learning entirely.
        total_steps = len(loader) * epochs
        self.setup_scheduler(total_steps)

        self.model.train()

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                loss = self.train_step(batch, global_step)
                epoch_loss += loss

                if session:
                    session.step(loss)

                global_step += 1

            avg = epoch_loss / len(loader)
            print(f"[Epoch {epoch + 1}/{epochs}] avg_loss={avg:.4f}")

        return self.model

    # =========================================================
    # GENERATION
    # =========================================================

    def generate(self, prompt: str, max_new_tokens: int = 128):
        # ← CHANGED: wrap prompt in Instruct chat template for correct inference
        formatted = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = self.tokenizer(
            formatted, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
