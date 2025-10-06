from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

use_cached_model=True
model_id = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=use_cached_model)

# Load once (this triggers the MXFP4 → BF16 conversion)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,     # keep BF16
    device_map="cpu",              # or "auto" if you have VRAM for BF16
    local_files_only=use_cached_model,
)

# Remove leftover quantization metadata so the model is treated as vanilla BF16
if hasattr(model.config, "quantization_config"):
    del model.config.quantization_config          # fixed in Transformers ≥4.47[1]

# Save the plain BF16 weights
model.save_pretrained("/leonardo_work/EUHPC_D17_077/fshahinf/dequantized/gpt-oss-20b-bf16")
tokenizer.save_pretrained("/leonardo_work/EUHPC_D17_077/fshahinf/dequantized/gpt-oss-20b-bf16")