import sys
import os
import types
import importlib.util
import importlib.machinery
import torch

# =================================================================
# 🛡️ Windows 防禦系統 V14 (Checkpoint Edition)
# 目的：增加 Epoch 數至 3，並加入自動存檔功能，防止過擬合導致白忙一場
# =================================================================

# 1. 強制關閉編譯功能
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_NO_model_card"] = "1"

# 2. 🔥 智慧轉接頭 V3 (AMP Polyfills)
try:
    if not hasattr(torch.amp, "custom_fwd"):
        from torch.cuda.amp import custom_fwd as legacy_fwd
        from torch.cuda.amp import custom_bwd as legacy_bwd

        def smart_custom_fwd(*args, **kwargs):
            if "device_type" in kwargs or len(args) == 0:
                def decorator(func): return legacy_fwd(func)
                return decorator
            return legacy_fwd(*args, **kwargs)

        def smart_custom_bwd(*args, **kwargs):
            if "device_type" in kwargs or len(args) == 0:
                def decorator(func): return legacy_bwd(func)
                return decorator
            return legacy_bwd(*args, **kwargs)

        torch.amp.custom_fwd = smart_custom_fwd
        torch.amp.custom_bwd = smart_custom_bwd

    if not hasattr(torch.amp, "is_autocast_available"):
        def mock_is_autocast_available(device_type):
            return device_type == "cuda"
        torch.amp.is_autocast_available = mock_is_autocast_available

except Exception as e:
    print(f"⚠️ AMP Patch Warning: {e}")

# 3. 閹割 torch.compile
def dummy_compile(model=None, *, fullgraph=False, dynamic=False, backend="inductor", mode=None, options=None, disable=False):
    def decorator(func): return func
    if model and callable(model): return model
    return decorator
torch.compile = dummy_compile

# 4. 偽造 torch._inductor
try:
    class MockConfig:
        def is_fbcode(self): return False
    class MockDeviceProperties:
        def __init__(self, *args, **kwargs): pass

    mock_modules = [
        "torch._inductor", "torch._inductor.config", 
        "torch._inductor.runtime", "torch._inductor.runtime.hints",
        "torch._inductor.test_operators", "torch._inductor.utils"
    ]
    for mod_name in mock_modules:
        m = types.ModuleType(mod_name)
        if mod_name == "torch._inductor.config": m.is_fbcode = lambda: False
        sys.modules[mod_name] = m
        if "." in mod_name:
            parent, child = mod_name.rsplit(".", 1)
            if parent in sys.modules: setattr(sys.modules[parent], child, m)

    sys.modules["torch._inductor.runtime.hints"].DeviceProperties = MockDeviceProperties
    if not hasattr(torch, "_inductor"): setattr(torch, "_inductor", sys.modules["torch._inductor"])
except Exception as e: pass

# 5. 偽造 torchvision
try:
    if "torchvision" not in sys.modules:
        m_tv = types.ModuleType("torchvision")
        m_tv.__spec__ = importlib.machinery.ModuleSpec(name="torchvision", loader=None)
        m_tv.__version__ = "0.19.1" 
        m_ops = types.ModuleType("torchvision.ops")
        m_ops.__spec__ = importlib.machinery.ModuleSpec(name="torchvision.ops", loader=None)
        m_ops.nms = lambda *args, **kwargs: args[0]
        m_tv.ops = m_ops
        sys.modules["torchvision"] = m_tv
        sys.modules["torchvision.ops"] = m_ops
except Exception as e: pass

# =================================================================
# 正式程式碼
# =================================================================

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

MAX_SEQ_LENGTH = 2048
DTYPE = None 
LOAD_IN_4BIT = True

def main():
    print("🚀 [Training V14] 3 Epochs 進階訓練模式啟動...")
    
    # 載入模型
    print("📦 載入 Qwen 2.5-7B (4-bit)...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    # 配置 LoRA
    print("🔧 配置 LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, 
        bias = "none", 
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
    )

    # 載入數據
    print("📂 載入數據集...")
    dataset = load_dataset("json", data_files="uruha_final_train.json", split="train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for inst, inp, out in zip(instructions, inputs, outputs):
            text = f"<|im_start|>system\n{inst}<|im_end|>\n<|im_start|>user\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            texts.append(text)
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched = True,)
    
    # 計算總步數
    total_steps_per_epoch = len(dataset) // (2 * 4) # batch_size 2 * grad_accum 4 = 8
    print(f"🔥 開始訓練 (資料量: {len(dataset)})")
    print(f"💡 預計每個 Epoch 步數: {total_steps_per_epoch}")
    print(f"💡 總 Epochs: 3 (總步數約 {total_steps_per_epoch * 3})")
    
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, 
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            
            # 🔥 關鍵修改：3 個 Epochs
            num_train_epochs = 3, 
            
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10, 
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            
            # 💾 安全機制：存檔策略
            save_strategy = "steps",
            save_steps = 500,     # 每 500 步存一個檔
            save_total_limit = 3, # 最多保留 3 個存檔，避免硬碟爆掉
        ),
    )

    trainer_stats = trainer.train()
    print(f"✅ 訓練完成！耗時: {trainer_stats.metrics['train_runtime']} 秒")
    
    print("💾 儲存最終結果...")
    model.save_pretrained("uruha_lora_adapters")
    tokenizer.save_pretrained("uruha_lora_adapters")
    print("🎉 訓練結束！請執行 05_merge.py。")

if __name__ == "__main__":
    main()