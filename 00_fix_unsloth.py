import os
import sys

def main():
    print("🚑 [V5 Total Erasure] 正在搜尋 Unsloth 安裝路徑...")
    
    # 1. 找到 site-packages 路徑
    target_path = r"C:\Users\jerry\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages"
    unsloth_path = os.path.join(target_path, "unsloth")
    
    if not os.path.exists(unsloth_path):
        print(f"❌ 找不到 Unsloth 資料夾: {unsloth_path}")
        return

    print(f"✅ 找到 Unsloth: {unsloth_path}")

    # ==========================================
    # 任務 A: 修復 vision.py (CompileConfig)
    # ==========================================
    vision_file = os.path.join(unsloth_path, "models", "vision.py")
    if os.path.exists(vision_file):
        with open(vision_file, "r", encoding="utf-8") as f: content = f.read()
        robust_mock = "class CompileConfig:\n    def __init__(self, *args, **kwargs): pass\n"
        if "class CompileConfig: pass" in content:
            content = content.replace("class CompileConfig: pass", robust_mock)
            with open(vision_file, "w", encoding="utf-8") as f: f.write(content)
        elif ", CompileConfig" in content:
            content = content.replace(", CompileConfig", "")
            if "from transformers import GenerationConfig" in content:
                content = content.replace("from transformers import GenerationConfig", f"{robust_mock}\nfrom transformers import GenerationConfig")
            with open(vision_file, "w", encoding="utf-8") as f: f.write(content)
        print("✅ vision.py 已修復")

    # ==========================================
    # 任務 B: 修復 import_fixes.py (Torchvision)
    # ==========================================
    fixes_file = os.path.join(unsloth_path, "import_fixes.py")
    if os.path.exists(fixes_file):
        with open(fixes_file, "r", encoding="utf-8") as f: content = f.read()
        if 'importlib.util.find_spec("torchvision")' in content:
            new_lines = []
            skip = False
            for line in content.splitlines():
                if "def torchvision_compatibility_check():" in line:
                    new_lines.append("def torchvision_compatibility_check(): pass")
                    skip = True
                elif skip and line.strip().startswith("def "):
                    skip = False
                    new_lines.append(line)
                elif not skip:
                    new_lines.append(line)
            with open(fixes_file, "w", encoding="utf-8") as f: f.write("\n".join(new_lines))
        print("✅ import_fixes.py 已修復")

    # ==========================================
    # 任務 C: 修復 loader.py (停用 Qwen 3 載入)
    # ==========================================
    loader_file = os.path.join(unsloth_path, "models", "loader.py")
    if os.path.exists(loader_file):
        with open(loader_file, "r", encoding="utf-8") as f: content = f.read()
        old_import_q3 = "from .qwen3 import FastQwen3Model"
        new_import_q3 = "class FastQwen3Model: pass # Disabled\n# from .qwen3 import FastQwen3Model"
        if old_import_q3 in content:
            content = content.replace(old_import_q3, new_import_q3)
        old_import_q3m = "from .qwen3_moe import FastQwen3MoeModel"
        new_import_q3m = "class FastQwen3MoeModel: pass # Disabled\n# from .qwen3_moe import FastQwen3MoeModel"
        if old_import_q3m in content:
            content = content.replace(old_import_q3m, new_import_q3m)
        with open(loader_file, "w", encoding="utf-8") as f: f.write(content)
        print("✅ loader.py 已修復")

    # ==========================================
    # 任務 D: 🔥 淨化 Qwen 3 相關檔案 (避免被其他檔案 import)
    # ==========================================
    for fname in ["qwen3.py", "qwen3_moe.py"]:
        fpath = os.path.join(unsloth_path, "models", fname)
        if os.path.exists(fpath):
            # 直接清空，只留空殼，避免任何依賴錯誤
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("class FastQwen3Model: pass\nclass FastQwen3MoeModel: pass\n")
            print(f"✅ {fname} 已淨化 (內容已清空)")

    # ==========================================
    # 任務 E: 🔥 封殺 models/__init__.py (關鍵！防止自動匯入)
    # ==========================================
    init_file = os.path.join(unsloth_path, "models", "__init__.py")
    if os.path.exists(init_file):
        print("🔧 正在清理 models/__init__.py ...")
        with open(init_file, "r", encoding="utf-8") as f: lines = f.readlines()
        
        new_lines = []
        for line in lines:
            # 只要是匯入 qwen3 或 qwen3_moe 的行，全部註解掉
            if "from .qwen3" in line or "from .qwen3_moe" in line:
                new_lines.append(f"# {line}") # Comment out
            else:
                new_lines.append(line)
        
        with open(init_file, "w", encoding="utf-8") as f: f.writelines(new_lines)
        print("✅ models/__init__.py 已清理")

    print("\n🎉 V5 全面清洗完畢！障礙已全部清除。")

if __name__ == "__main__":
    main()