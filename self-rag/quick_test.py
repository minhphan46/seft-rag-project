import os

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_DIR = r"C:\Users\minhp\Desktop\development\seft-rag-project\selfrag_llama2_7b"
# Bắt buộc khi `device_map="auto"` phải offload .bin ra đĩa (VRAM/RAM không đủ)
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "_quick_test_offload")
os.makedirs(OFFLOAD_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
load_kw = dict(
    dtype=dtype,
    low_cpu_mem_usage=True,
)
if torch.cuda.is_available():
    load_kw["device_map"] = "auto"
    load_kw["offload_folder"] = OFFLOAD_DIR
else:
    load_kw["device_map"] = None

model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kw)
if not torch.cuda.is_available():
    model = model.to("cpu")

prompt = "### Instruction:\nCan you tell me the difference between llamas and alpacas?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
embed_device = model.get_input_embeddings().weight.device
inputs = {k: v.to(embed_device) for k, v in inputs.items()}

# Dọn generation config mặc định của checkpoint để tránh cảnh báo không cần thiết.
model.generation_config.max_length = None
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None

print("Generating... (this can take long on CPU/offload)")
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(out[0], skip_special_tokens=False))