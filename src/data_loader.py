import json


def load_jsonl(path: str):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] bad JSON at line {i}: {e}")
                continue

            prompt = item.get("prompt", "")
            completion = item.get("completion", "")

            # ← CHANGED: switched from ### Instruction / ### Response format
            #   to LLaMA Instruct chat template. This matches the exact format
            #   the Instruct model was RLHF-trained on, giving it a much lower
            #   starting loss and faster convergence.
            text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                f"{completion}<|eot_id|>"
            )

            data.append({
                "text": text,
                # marker tells TextDataset where the response begins so
                # only the completion tokens contribute to the loss
                "response_start_marker": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            })

    if not data:
        raise ValueError("No valid JSONL rows found")

    return data
