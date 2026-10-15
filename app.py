import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from huggingface_hub import hf_hub_download, login # 関数のインポートを追記
# ────────────
# 1. Hugging Face トークンの取得とログイン
# ────────────
# Streamlit の "Secrets" に保存したトークンを読み込み
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
# Hugging Face 上で認証セッションを開始
login(token=hf_token)
# ────────────
# 2. モデル読み込み関数の定義
# ────────────
@st.cache_resource
def load_yolo_model(repo_id: str, filename: str) -> YOLO:
# モデルファイルを認証付きでダウンロード
model_path = hf_hub_download(
repo_id=repo_id,
filename=filename,
token=hf_token
)

Python
# YOLO モデルを初期化
model = YOLO(model_path)
return model
# ────────────
# 3. 実際にモデルをロード(初回だけダウンロード/初期化される)
# ────────────
MODEL_REPO = "matsubara0413/sushi-detection-model" # ←自分のリポジトリに書き換えてください
MODEL_FILE = "best.pt"
model = load_yolo_model(repo_id=MODEL_REPO, filename=MODEL_FILE)
# ────────────
# 4. streamlitアプリの中身
# ────────────
st.title("🍣 寿司ネタ 検知アプリ")
st.write("画像をアップロードすると、寿司ネタを検出します。")


# Input
upload_img = st.file_uploader("画像ファイルを選択", type = ['png', 'jpg'])

# Process
if upload_img is not None:

    # 読み込んだ画像をPythonで扱えるようにするための処理を記載
    bytes_data = upload_img.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8),
    cv2.IMREAD_COLOR)

    # 物体検知を実行
    results = model(img, conf=0.7)

    st.write("検出数:", len(results[0].boxes))

    for box in results[0].boxes:
      st.write("クラス:", int(box.cls[0]))
      st.write("名前:", results[0].names[int(box.cls[0])])
      st.write("信頼度:", float(box.conf[0]))

    # 結果を出力
    output_img = results[0].plot(labels=True, conf=True)

    st.image(output_img, channels="BGR", caption="出力画像")

# Output
    st.image(output_img, channels="BGR", caption='出力画像')
