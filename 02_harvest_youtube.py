import os
import yt_dlp
import whisper
import torch

# 🔴 請在這裡填入 Uruha 的「純雜談」直播連結 (2-3 部即可)
# 建議找標題有「雑談」「歌枠」或是「記念」的，避開 FPS 遊戲回
URLS = [
    "https://www.youtube.com/watch?v=Rhb8ORhO2wg", 
    "https://www.youtube.com/watch?v=S-5v02P3XiY",
    "https://www.youtube.com/watch?v=JyZgMkuogVs",
    "https://www.youtube.com/watch?v=D1ZzgFl-qMo",
    "https://www.youtube.com/watch?v=ZUrEQulrB3k",
    "https://www.youtube.com/watch?v=fz0xE-ACteU",
    "https://www.youtube.com/watch?v=UaNyVP8jz6E",
    "https://www.youtube.com/watch?v=yqipuZrka44",
    "https://www.youtube.com/watch?v=6o5YhcuU_wE",
    "https://www.youtube.com/watch?v=XkSwqcGipwk",
    "https://www.youtube.com/watch?v=NeeCC9Y1rs4"
]

OUTPUT_DIR = "raw_transcripts"

def main():
    # 1. 檢查 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 運算裝置: {device} (4070 Ti 應該要是 cuda)")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 載入 Whisper 模型
    # 你的顯卡夠強，直接用 large-v3 獲取最高精準度
    print("📥 正在載入 Whisper Large-V3 模型...")
    model = whisper.load_model("large-v3", device=device)

    for url in URLS:
        print(f"\n🎥 正在處理: {url}")
        
        # A. 下載音訊 (使用 yt-dlp)
        temp_audio = "temp_audio.mp3"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'mp3',}],
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            continue

        # B. AI 聽寫 (Transcribing)
        print("🎙️ AI 正在聽寫中 (這需要一點時間)...")
        result = model.transcribe(temp_audio, language="Japanese") # 強制指定日文

        # C. 存檔與清洗
        video_id = url.split("v=")[-1]
        save_path = f"{OUTPUT_DIR}/{video_id}.txt"
        
        with open(save_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                text = segment["text"].strip()
                # 過濾掉太短的噪音，保留完整的句子
                if len(text) > 5:
                    f.write(text + "\n")
        
        print(f"✅ 已儲存字幕: {save_path}")
        
        # 清理暫存檔
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    print("\n🎉 所有影片處理完成！請進行下一步：數據合成。")

if __name__ == "__main__":
    main()