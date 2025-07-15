# 📚 LoRA & QLoRA: Parameter-Efficient Fine-Tuning for Large Language Models

> **TL;DR:** Fine-tune huge language models affordably by training tiny adapter matrices instead of all parameters. **LoRA** does this on full-precision models. **QLoRA** adds **4-bit quantization** for maximum memory efficiency — without sacrificing performance.

---

## 🎯 What Problem Does This Solve?

Fine-tuning large language models (LLMs) like **LLaMA** or **GPT** traditionally requires:

* 🚀 **Massive GPU memory** (100 GB+ for 7B models)
* 🕒 **Days/weeks of training time**
* 💰 **Expensive hardware infrastructure**
* 📦 **Multiple full model copies** for each fine-tuned version

**LoRA** and **QLoRA** make this process **affordable, fast, and memory-efficient** by fine-tuning just a tiny fraction of parameters.

---

## 🔧 What is LoRA (Low-Rank Adaptation)?

### 📝 Concept

Instead of updating full weight matrices during fine-tuning:

1. **Freeze pretrained model weights** `W`
2. **Introduce small trainable matrices** `A` and `B`
3. **Train only these adapters**
4. Combine them at inference:

$$
W_{\text{new}} = W + A \times B
$$

---

### 📐 Mathematical Foundation

For a weight matrix $W \in \mathbb{R}^{d \times d}$:

* Updates are decomposed as: $\Delta W = A \times B$
* Where:

  * $A \in \mathbb{R}^{d \times r}$
  * $B \in \mathbb{R}^{r \times d}$
* $r \ll d$ (typically 4, 8, 16)

**Why "Low-Rank"?**
Because the product $A \times B$ has rank ≤ $r$, forcing efficient, generalizable adaptations.

---

### 💾 Memory Benefits

* 7B model (FP16) → **\~13GB**
* LoRA adapters → **\~10–50MB**
* Total trainable params: **<1% of the model**

---

## ⚡ What is QLoRA?

### 🚀 Core Enhancement

**QLoRA = LoRA + 4-bit quantization of base model**

### 🔍 Process

1. Quantize pretrained model to **4-bit NF4 (Normalized Float 4-bit)**
2. Add LoRA adapters to attention layers
3. Train adapters in **16-bit precision**
4. Keep base model frozen at **4-bit**

### 📉 Memory Savings

| Model | FP16    | 4-bit NF4 |
| :---- | :------ | :-------- |
| 7B    | \~13GB  | \~3.5GB   |
| 13B   | \~24GB  | \~6.5GB   |
| 70B   | \~120GB | \~35GB    |

Yes — a **70B model can fine-tune on a single A100** using QLoRA.

---

## 🏗️ Where LoRA is Applied

Typically inserted into:

* `q_proj` (query projection)
* `v_proj` (value projection)
* `k_proj` (key projection)
* `o_proj` (output projection)
* Gate/Up projections in MLP layers

---

## 📝 Key Hyperparameters

| Hyperparameter   | Description                       | Typical Values |
| :--------------- | :-------------------------------- | :------------- |
| `rank (r)`       | LoRA adapter dimension            | 4–64           |
| `alpha`          | Scaling factor (usually `2×rank`) | 16–128         |
| `dropout`        | Regularization                    | 0.05–0.1       |
| `target_modules` | Which layers to adapt             | See above      |

**Example LoRA Config:**

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## 📊 LoRA vs QLoRA vs Full Fine-tuning

| Method           | Memory (7B) | Training Time | Performance | Flexibility |
| :--------------- | :---------- | :------------ | :---------- | :---------- |
| Full Fine-tuning | \~130GB     | Days          | 100%        | Low         |
| LoRA             | \~15GB      | Hours         | 95–99%      | High        |
| QLoRA            | \~9GB       | Hours         | 95–99%      | High        |

---

## 🛠️ Libraries & Tools

* `peft` (Hugging Face): LoRA / QLoRA adapters
* `bitsandbytes`: 4-bit quantization backends
* `transformers`: Pretrained model hub
* `trl`: Training utilities

**Installation:**

```bash
pip install torch transformers peft bitsandbytes trl
```

---

## 🎯 Use Cases

* Task-specific fine-tuning (chat, summarization, coding)
* Domain adaptation (medical, legal, finance)
* Instruction-tuning and alignment
* Multiple adapters for a single base model

**When to pick:**

* Use **LoRA** for <13B models on 24GB+ GPUs
* Use **QLoRA** for 13B+ models or <24GB setups

---

## 🔍 Technical Deep Dive

### Quantization Formats:

* `FP16`: 16-bit floating point
* `INT8`: 8-bit integer
* `NF4`: 4-bit normalized float (used in QLoRA)

### Training Flow:

* Forward pass: Quantized base + LoRA adapters
* Backward pass: Only adapters updated
* Optimizer updates: Only LoRA params

### Inference Modes:

* Merged (apply `W + AB` directly)
* Separate (switchable adapters)
* Multi-adapter stacking

---

## 📈 Best Practices

* Start with `rank=16`, `alpha=32`
* LoRA Dropout: 0.05–0.1
* High-quality, clean, balanced data
* Track perplexity and overfitting
* Use gradient accumulation for memory limits

---

## 🚀 Advanced Techniques

* **Multi-LoRA**: Load multiple adapters dynamically
* **DoRA**: Decomposed LoRA for even higher efficiency
* **AdaLoRA**: Dynamically adjust adapter rank per layer

---

## 🛡️ Common Pitfalls

* Too high learning rate → instability
* Low rank → underfitting
* Missing target modules → no effect
* Long sequences + small VRAM → OOM errors

---

## 📚 Resources

* 📄 [LoRA Paper (2021)](https://arxiv.org/abs/2106.09685)
* 📄 [QLoRA Paper (2023)](https://arxiv.org/abs/2305.14314)
* 📖 [Hugging Face PEFT Docs](https://huggingface.co/docs/peft/index)
* 📝 [BitsAndBytes Repo](https://github.com/TimDettmers/bitsandbytes)

---

## 🔄 Quick Commands

```bash
# Install dependencies
pip install torch transformers peft bitsandbytes trl

# Fine-tune with LoRA or QLoRA
python train.py --model_name "meta-llama/Llama-2-7b-hf" \
                --dataset "your_dataset" \
                --lora_rank 16 \
                --quantization 4bit

# Merge LoRA adapters with base model for inference
python merge_lora.py --base_model "model_path" \
                     --lora_weights "lora_path" \
                     --output_path "merged_model"
```

---

## 💡 Key Takeaways

✅ LoRA trains <1% of parameters
✅ QLoRA uses 4-bit models for major memory savings
✅ Comparable performance to full fine-tuning
✅ Fine-tune 13B–70B models on a single GPU
✅ Extremely flexible, efficient, and production-friendly

---

> **For AI/LLM developers**: LoRA and QLoRA are now essential tools for anyone serious about practical large model fine-tuning without enterprise hardware.

