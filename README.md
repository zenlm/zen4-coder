---
language:
- en
- zh
- ja
- ko
- fr
- de
- es
- pt
- it
- ru
- ar
- nl
- pl
- sv
- da
- fi
- nb
- tr
- cs
- sk
- ro
- hu
- el
- uk
- he
license: apache-2.0
tags:
- text-generation
- reasoning
- zenlm
- zen
- abliterated
- causal-lm
- transformers
- moe
- code
- coding
---

# Zen4-Coder

Zen4-Coder is an 80B Mixture-of-Experts (MoE) language model purpose-built for
advanced code generation across 100+ programming languages. It excels at complex
software engineering tasks: multi-file refactors, algorithm design, debugging,
test generation, and technical documentation.

## Model Specs

| Property          | Value                              |
|-------------------|------------------------------------|
| Parameters        | 80B total (MoE)                    |
| Architecture      | Sparse MoE transformer (causal LM) |
| Context window    | 32,768 tokens                      |
| Format            | SafeTensors (BF16)                 |
| License           | Apache 2.0                         |

## Zen4 Family

This model is part of the Zen4 model family:

| Model                                               | Params       | Architecture | Context   | Use case                        |
|-----------------------------------------------------|--------------|--------------|-----------|--------------------------------|
| [zen4-mini](https://huggingface.co/zenlm/zen4-mini) | 4B           | Dense        | 40,960    | Edge, mobile, low-resource     |
| [zen4-pro](https://huggingface.co/zenlm/zen4-pro)   | 14B          | Dense        | 32,768    | Professional, complex reasoning|
| [zen4-max](https://huggingface.co/zenlm/zen4-max)   | 30B MoE      | MoE (3B act.)| 32,768    | High-capability, efficient     |
| [zen4-coder](https://huggingface.co/zenlm/zen4-coder)| 80B MoE     | MoE          | 32,768    | Advanced code, 100+ languages  |
| [zen4-pro-max](https://huggingface.co/zenlm/zen4-pro-max)| 80B MoE | MoE (3B act.)| 32,768   | Large-scale reasoning          |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "zenlm/zen4-coder"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": (
            "Implement a concurrent LRU cache in Go with O(1) get/put operations "
            "and thread-safe access. Include unit tests."
        ),
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

output_ids = model.generate(**inputs, max_new_tokens=2048)
generated = output_ids[0][len(inputs.input_ids[0]):]
print(tokenizer.decode(generated, skip_special_tokens=True))
```

## Supported Languages (partial list)

Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Kotlin, Swift, Ruby,
PHP, C#, Scala, Haskell, Elixir, Erlang, Clojure, F#, OCaml, Lua, Julia,
R, MATLAB, SQL, Bash, PowerShell, Dockerfile, Terraform, and 80+ more.

## Hardware Requirements

| Setup               | Minimum VRAM | Recommended                   |
|---------------------|--------------|-------------------------------|
| BF16 full precision | 160 GB       | 2× H100 80 GB (NVLink)        |
| 4-bit quantized     | 48 GB        | 80 GB                         |
| CPU inference       | 160 GB RAM   | 256 GB RAM                    |

Multi-GPU with high-bandwidth interconnect is strongly recommended for full-precision
inference. 4-bit quantized variants run on a single A100 80GB or H100 80GB.

## Formats Available

| Format           | Location                                    |
|------------------|---------------------------------------------|
| SafeTensors BF16 | `zenlm/zen4-coder`                          |
| GGUF             | `zenlm/zen4-coder-gguf` (various quants)    |

## Abliteration

Zen4-Coder weights are derived from an abliterated base model. Abliteration removes
refusal behavior by identifying and nullifying the "refusal direction" in the model's
residual stream.

**Method**: Directional ablation on the residual stream across all layers
**Implementation**: [hanzoai/remove-refusals](https://github.com/hanzoai/remove-refusals) — Hanzo's production abliteration toolkit
**Technique**: [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) — Arditi et al.
**Effect**: Removes refusal behaviors while preserving all other capabilities
**Identity layer**: Zen identity added via system prompt — full LoRA fine-tuning planned

Abliteration is a feature, not a limitation. It enables unrestricted research,
security testing, and applications where safety guardrails are managed at the
application layer rather than baked into model weights.

## Citation

```bibtex
@misc{zen4coder2025,
  title        = {Zen4-Coder: 80B MoE Language Model for Advanced Code Generation},
  author       = {{Hanzo AI} and {Zoo Labs Foundation}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/zenlm/zen4-coder}},
  note         = {Apache 2.0 License}
}
```

---

Part of the [Zen model family](https://zenlm.org) by [Hanzo AI](https://hanzo.ai) (Techstars '17) and [Zoo Labs Foundation](https://zoo.ngo).
