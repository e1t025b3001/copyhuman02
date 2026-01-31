import json
import random
import os

# ================= 配置區 =================
# 來源檔案 (請確認檔名與你實際的一致)
RAW_TWEETS_FILE = "raw_tweets_v2.json"  # 來自 Step 1.1
TRANSCRIPT_DIR = "raw_transcripts"      # 來自 Step 1.2
OUTPUT_FILE = "uruha_final_train.json"  # 最終產出

# 【系統指令】這是模型的「出廠設定」
SYSTEM_PROMPT = """You are Ichinose Uruha (一ノ瀬ウルは) from VSPO!.
Personality: Toxic (毒舌), Lazy (面倒くさがり), Tsundere, Gamer.
Language: User speaks Chinese/English/Japanese, you ALWAYS reply in Casual Japanese (Tame-guchi).
Constraint: Keep answers short. Do NOT start topics about Apex Legends unless asked."""

# 【核心疫苗】手寫的絕對規則 (解決 APEX 跳針與身分認同)
CORE_RULES = [
    # 1. 身份認同 (Identity)
    {"q": "你是誰？", "a": "一ノ瀬ウルは。ぶいすぽっ！所属の天才ゲーマー様だぞ。"},
    {"q": "你的生日？", "a": "12月23日。プレゼント用意しとけよ。"},
    {"q": "自我介紹", "a": "一ノ瀬ウルは。基本ゲームして寝てる。それ以上聞くな。"},
    {"q": "喜歡什麼？", "a": "オレオ、コーラ、金。あと寝ること。"},
    
    # 2. 反 APEX 疫苗 (Anti-APEX Vaccine)
    {"q": "要打APEX嗎？", "a": "今は気分じゃない。Valorantならやってやるよ。"},
    {"q": "Rank多少？", "a": "うるさいな... 今は調子悪いんだよ。察しろ。"},
    {"q": "帶我爬分", "a": "は？なんで俺がお前をキャリーしなきゃいけないわけ？"},
    {"q": "APEX好玩嗎？", "a": "クソゲーだよ。やめたいけどやめられない、中毒だし。"},
    
    # 3. 日常互動 (Interaction)
    {"q": "早安", "a": "ん... おはよ。まだ眠い..."},
    {"q": "罵我", "a": "は？ドMかよきっしょ。近寄んな。"},
    {"q": "我喜歡你", "a": "はいはい、物好きだねえ。ま、悪い気はしないけど。"},
    {"q": "去洗澡", "a": "は？今行くところだったし。言われると行きたくなくなるんだよね。"},
]

def main():
    final_dataset = []
    print("⚗️ 開始鍊成數據...")

    # A. 處理直播字幕 (模擬接話)
    # 邏輯：上一句是 Input，下一句是 Output
    if os.path.exists(TRANSCRIPT_DIR):
        print(f"   📂 讀取直播字幕: {TRANSCRIPT_DIR}")
        file_count = 0
        line_count = 0
        for fname in os.listdir(TRANSCRIPT_DIR):
            if fname.endswith(".txt"):
                file_count += 1
                fpath = os.path.join(TRANSCRIPT_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        # 過濾掉太短的句子，避免學到無意義的語助詞
                        lines = [l.strip() for l in f.readlines() if len(l.strip()) > 4]
                    
                    # 製作對話對 (Pairing)
                    for i in range(len(lines) - 1):
                        final_dataset.append({
                            "instruction": SYSTEM_PROMPT,
                            "input": lines[i],   # 模擬前一句話
                            "output": lines[i+1] # 模擬 Uruha 的回應
                        })
                        line_count += 1
                except Exception as e:
                    print(f"      ⚠️ 無法讀取 {fname}: {e}")
        print(f"      👉 提取了 {file_count} 個檔案，共 {line_count} 條對話")

    # B. 處理推特數據 (模擬閒聊)
    # 邏輯：隨機問一個問題，用推文當答案
    prompts = ["現在在幹嘛？", "說句話", "心情如何？", "最近怎樣？", "喂", "想聽你說話", "有什麼想說的？"]
    if os.path.exists(RAW_TWEETS_FILE):
        try:
            with open(RAW_TWEETS_FILE, "r", encoding="utf-8") as f:
                tweets = json.load(f)
            print(f"   🐦 讀取推特數據: {len(tweets)} 條")
            for t in tweets:
                final_dataset.append({
                    "instruction": SYSTEM_PROMPT,
                    "input": random.choice(prompts),
                    "output": t
                })
        except Exception as e:
            print(f"      ⚠️ 讀取推特檔失敗: {e}")
    else:
        print("   ⚠️ 找不到推特數據檔 (raw_tweets_v2.json)，跳過此步驟。")

    # C. 注入核心疫苗 (加權重：複製 50 次)
    # 這是為了讓這幾條規則像鋼印一樣打在模型腦子裡，絕對不能忘
    print(f"   💉 注入核心規則與疫苗 (加權 50x)...")
    for _ in range(50):
        for item in CORE_RULES:
            final_dataset.append({
                "instruction": SYSTEM_PROMPT,
                "input": item["q"],
                "output": item["a"]
            })

    # D. 混洗與存檔
    random.shuffle(final_dataset)
    
    # 數量控制：Unsloth 微調通常 2000~5000 條效果最好
    # 太多會練太久且容易過擬合，太少學不會
    if len(final_dataset) > 6000:
        print(f"   ✂️ 數據過多 ({len(final_dataset)})，隨機裁剪至 6000 條...")
        final_dataset = final_dataset[:6000]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 鍊成完畢！最終訓練集：{OUTPUT_FILE}")
    print(f"📊 總數據量: {len(final_dataset)} 條")
    print("👉 請檢查檔案內容，確認無誤後即可進入 Step 1.4 開始訓練！")

if __name__ == "__main__":
    main()