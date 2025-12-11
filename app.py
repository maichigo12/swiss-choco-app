import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ===============================
# 1. スイスチョコレート データセット
# ===============================

data = pd.DataFrame([
    ["Lindt Excellence 70%", 70, 2, 5, 1, 0, 500],
    ["Lindt Milk", 30, 5, 1, 5, 0, 400],
    ["Toblerone Milk", 30, 4, 1, 4, 1, 350],
    ["Toblerone Dark", 50, 2, 4, 1, 1, 380],
    ["Cailler Milk", 30, 5, 1, 4, 0, 420],
    ["Cailler Dark", 60, 2, 4, 1, 0, 450],
    ["Frey Milk", 30, 4, 1, 4, 0, 300],
    ["Frey Hazelnut", 30, 4, 1, 4, 1, 320],
    ["Läderach Milk", 35, 4, 2, 5, 1, 900],
    ["Läderach Dark 70%", 70, 2, 5, 1, 0, 1000],
],
columns=["name", "cacao", "sweet", "bitter", "milk", "nuts", "price"]
)

X = data[["cacao", "sweet", "bitter", "milk", "nuts", "price"]]

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN モデル（類似度でおすすめを出す）
model = NearestNeighbors(n_neighbors=3)
model.fit(X_scaled)

# ===============================
# 2. Streamlit アプリ UI
# ===============================

st.title("🇨🇭 スイスチョコレート診断アプリ 🍫")
st.write("あなたの好みに合うスイスチョコを3つおすすめします！")

st.subheader("① 好みを選んでください")

sweet = st.slider("甘さ (Sweet)", 1, 5, 3)
bitter = st.slider("苦味 (Bitter)", 1, 5, 3)
milk_taste = st.slider("ミルク感 (Milk)", 1, 5, 3)
nuts = st.selectbox("ナッツは好き？ (Nuts)", ["嫌い", "好き"])
nuts = 1 if nuts == "好き" else 0
price_preference = st.slider("価格帯（安い=0 → 高い=1000）", 0, 1000, 500)

# -------------------------
# 診断ボタン
# -------------------------
if st.button("おすすめを診断する"):
    st.subheader("② あなたにおすすめのスイスチョコは… 🍫✨")

    user = [[
        30,              # ユーザー入力がないため仮に30%
        sweet,
        bitter,
        milk_taste,
        nuts,
        price_preference
    ]]

    user_scaled = scaler.transform(user)
    distances, indices = model.kneighbors(user_scaled)

    recommended = data.iloc[indices[0]]

    st.table(recommended)

    st.success("診断が完了しました！")
