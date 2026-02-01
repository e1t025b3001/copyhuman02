import json
import os
import re

# 設定檔名
INPUT_FILE = "uruha_final_train.json"
OUTPUT_FILE = "uruha_clean_train.json"

# 定義要殺掉的關鍵字 (髒資料特徵)
# 只要 output 裡包含這些字，這筆資料就整筆刪掉
BLACKLIST_KEYWORDS = [
    "スーパーチャット", "スパチャ", "Super Chat", "SuperChat",
    "メンバーシップ", "メンシプ", "Membership",
    "ありがとうございます", "ありがとうございまーす", # 雖然有點激進，但通常唸SC才會這樣連著講
    "ナイスパ", "ないすぱ", # Nice Superchat
    "下記", "概要欄", # 直播常見用語
    "待機所", "配信", # 視情況，有時候這會導致文不對題
    "￥", "¥" # 金額符號
]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}")
        return

    print(f"🧹 正在清洗資料，去除 SC 與雜訊...")
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_data = []
    
    for entry in data:
        output_text = entry["output"]
        
        # 檢查是否包含禁語
        is_dirty = False
        for keyword in BLACKLIST_KEYWORDS:
            if keyword in output_text:
                is_dirty = True
                # print(f"  🗑️ 移除髒資料: {output_text[:30]}...") # 想看刪了什麼可以打開這行
                break
        
        # 額外過濾：如果回答太短 (例如只有 "ん" 或 "はい")，可能也沒營養
        if len(output_text) < 2:
            is_dirty = True

        if not is_dirty:
            cleaned_data.append(entry)
            
    print("-" * 30)
    print(f"📊 統計報告:")
    print(f"   原始資料數: {original_count}")
    print(f"   剩餘資料數: {len(cleaned_data)}")
    print(f"   🗑️ 共刪除了: {original_count - len(cleaned_data)} 筆髒資料")
    print("-" * 30)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 已儲存乾淨的資料集至: {OUTPUT_FILE}")
    print("💡 請在 04_train.py 中將檔名改為 'uruha_clean_train.json' 並重新訓練！")

if __name__ == "__main__":
    main()