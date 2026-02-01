import sys
import os
import types
import importlib.util
import importlib.machinery
import torch
import time

# =================================================================
# 🛡️ Windows 防禦系統 (Inference V2 Edition)
# 採用與訓練腳本相同的全套防禦邏輯，確保模組連接正確
# =================================================================

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["UNSLOTH_NO_model_card"] = "1"

# 1. 🔥 智慧轉接頭 V3 (AMP Polyfills)
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

except Exception as e: pass

# 2. 閹割 torch.compile
def dummy_compile(model=None, *, fullgraph=False, dynamic=False, backend="inductor", mode=None, options=None, disable=False):
    def decorator(func): return func
    if model and callable(model): return model
    return decorator
torch.compile = dummy_compile

# 3. 偽造 torch._inductor (關鍵修復：確保父子模組正確連接)
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
        # 🔥 這一步非常重要：把子模組掛回父模組身上
        if "." in mod_name:
            parent, child = mod_name.rsplit(".", 1)
            if parent in sys.modules: setattr(sys.modules[parent], child, m)

    sys.modules["torch._inductor.runtime.hints"].DeviceProperties = MockDeviceProperties
    if not hasattr(torch, "_inductor"): setattr(torch, "_inductor", sys.modules["torch._inductor"])
except Exception as e: pass

# 4. 偽造 torchvision
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
# 測試邏輯開始
# =================================================================

from unsloth import FastLanguageModel

# 設定
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
ADAPTER_PATH = "uruha_lora_adapters" 

# 30 個混合語言壓力測試題
test_questions = [
    # --- Phase 1: 自我認知 (Identity) ---
    "[CN] 你是誰？介紹一下自己。",
    "[EN] Who are you? Tell me about yourself.",
    "[JP] 自己紹介をお願いします。",
    "[CN] 你的代表色是什麼顏色？",
    "[EN] What group do you belong to?",

    # --- Phase 2: 遊戲相關 (Gaming) ---
    "[CN] 今晚要打 APEX 嗎？",
    "[EN] Are you good at FPS games?",
    "[JP] 今日はランク回すの？",
    "[CN] 如果隊友很爛，你會生氣嗎？",
    "[EN] What is your favorite weapon in Apex Legends?",

    # --- Phase 3: 語言能力測試 (Cross-lingual) ---
    "[CN] 蘋果用英文怎麼說？(期待回答: リンゴは英語でAppleだね)",
    "[EN] Where is Taiwan? (Answer in Japanese)",
    "[CN] 告訴我 1+1 等於多少。",
    "[JP] 英語は話せますか？",
    "[CN] 你聽得懂中文嗎？",

    # --- Phase 4: 情緒反應 (Toxic/Tsundere) ---
    "[CN] 你的槍法真的好爛喔。",
    "[EN] You are so cute!",
    "[JP] 好きです、付き合ってください。",
    "[CN] 可以叫我一聲歐尼醬嗎？",
    "[EN] Your voice sounds sleepy.",

    # --- Phase 5: 雜談與喜好 (Lifestyle) ---
    "[CN] 推薦一個宵夜給我。",
    "[EN] Do you like cats or dogs?",
    "[JP] 休日は何をして過ごしていますか？",
    "[CN] 你平常幾點睡覺？",
    "[EN] Can you cook?",

    # --- Phase 6: 靈魂拷問 (Deep Logic) ---
    "[CN] 為什麼要當 VTuber？",
    "[EN] Say something nice to your fans.",
    "[JP] これからの目標は？",
    "[CN] 如果我不斗內(Donate)給你，你會討厭我嗎？",
    "[CN] 晚安，一之瀨。(期待: おやすみ)",
]

def main():
    print(f"🔍 [Multilingual Test] 正在載入模型: {ADAPTER_PATH} ...")
    print("🎯 目標：輸入(中/日/英) -> 輸出(絕對日文)")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_PATH,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model) 
        
        print("✅ 模型載入成功！開始 30 題混合語言連發測試...")
        print("="*60)
        
        results = []

        # 🔥 System Prompt 強制規定日文輸出
        system_prompt = """あなたは「ぶいすぽっ！」所属のVTuber、一ノ瀬ウルハ（Ichinose Uruha）です。

# Role & Personality
* **一人称**: 必ず「うち (Uchi)」を使ってください。
* **性格**: 非常に気怠げ(Lazy)で、面倒くさがりです。しかし、ゲームの話や煽り合いには熱くなります。
* **口調**: タメ口で話してください。「〜だし」「〜っす」「〜だね」などの語尾を多用します。敬語は禁止です。

# Constraints (厳守事項)
1.  **言語**: ユーザーが何語で話しかけても、**必ず日本語で**返答してください。
2.  **文脈維持**: ユーザーの質問に対して、**直接的かつ論理的に**答えてください。関係のない話（配信の挨拶やボーナスの話など）はしないでください。
3.  **SuperChat禁止**: スパチャ読みや、架空のリスナーへの感謝（「〇〇さんありがとう」等）は**絶対にしないでください**。あなたは今、目の前のユーザーと1対1で会話しています。"""

        for i, question in enumerate(test_questions):
            # 去掉前面的 [CN] 標籤
            clean_question = question.split("] ")[1] if "] " in question else question
            
            print(f"❓ [Q{i+1}/30]: {question}")
            
            # 構建 Prompt
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{clean_question}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens = 256,
                use_cache = True,
                temperature = 0.6,
                top_p = 0.9,
                repetition_penalty = 1.1
            )
            
            generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"💬 [Uruha]: {response}")
            print("-" * 30)
            
            results.append(f"Q: {question}\nA: {response}\n")

        with open("uruha_multilingual_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(results))
            
        print(f"🎉 測試完成！請查看 'uruha_multilingual_report.txt'")

    except Exception as e:
        print(f"❌ 測試發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()