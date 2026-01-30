import json
import time
import random
from DrissionPage import ChromiumPage, ChromiumOptions

# 目標網址
TARGETS = [
    "https://twitter.com/uruhasub/with_replies", # 小帳回覆 (最真實)
    "https://twitter.com/uruhasub",              # 小帳主頁
    "https://twitter.com/uruha_ichinose"         # 本帳
]
OUTPUT_FILE = "raw_tweets_v2.json"

def main():
    print("🚀 初始化 DrissionPage (比 Selenium 更強的隱形爬蟲)...")
    
    # 設定瀏覽器選項
    co = ChromiumOptions()
    # co.incognito() # 如果想要無痕模式可以打開，但建議不要，這樣可以吃你的 Chrome 登入資訊
    
    # 啟動瀏覽器
    page = ChromiumPage(co)
    
    collected_data = []
    
    try:
        # 1. 前往登入頁面 (如果已經登入過，這裡會自動跳轉)
        page.get("https://twitter.com/home")
        
        print("\n" + "="*50)
        print("⚠️ 【請手動操作】")
        print("1. 請確認瀏覽器是否已開啟。")
        print("2. 如果還沒登入 Twitter，請現在手動登入。")
        print("3. 確認看到 Twitter 首頁後，回來這裡按 [Enter] 開始。")
        print("="*50 + "\n")
        input("👉 準備好後請按 Enter...")

        # 2. 開始爬取
        for url in TARGETS:
            print(f"🔍 正在前往: {url}")
            page.get(url)
            time.sleep(3)
            
            # 簡單的防呆：檢查是否真的進去了
            if "login" in page.url:
                print("❌ 偵測到未登入，請重新登入後再試。")
                break

            consecutive_no_new = 0
            
            # 每個連結滾動 30 次
            for i in range(30):
                print(f"   📜 滾動中 ({i+1}/30)...")
                
                # 抓取所有推文元素 (DrissionPage 的語法很簡潔)
                # 這裡抓取 data-testid 為 tweetText 的 div
                tweets = page.eles('css:[data-testid="tweetText"]')
                
                new_count = 0
                for t in tweets:
                    txt = t.text.replace("\n", " ")
                    
                    # 過濾垃圾資訊
                    if len(txt) > 3 and "http" not in txt and txt not in collected_data:
                        # 簡單過濾掉單純 @別人 的回覆 (我們要有內容的)
                        if not txt.startswith("@"):
                            collected_data.append(txt)
                            new_count += 1
                            print(f"      ✅ 收錄: {txt[:20]}...")
                
                if new_count == 0:
                    consecutive_no_new += 1
                else:
                    consecutive_no_new = 0
                
                # 如果連續 3 次沒新東西，就換下一頁
                if consecutive_no_new >= 3:
                    print("🚫 無新資料，跳轉下一個目標。")
                    break

                # 滾動到底部
                page.scroll.to_bottom()
                
                # 隨機等待 (模擬人類閱讀)
                time.sleep(random.uniform(2, 5))

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
    
    # 存檔
    print(f"\n💾 正在儲存 {len(collected_data)} 條資料...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成！檔案已存為 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()