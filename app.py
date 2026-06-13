import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="콘크리트 압축강도 예측", page_icon="🧱", layout="wide")

# ---------- style (clean / Toss-ish) ----------
st.markdown("""
<style>
.stApp { background:#f9fafb; }
h1,h2,h3 { color:#191f28; font-weight:700; letter-spacing:-0.3px; }
[data-testid="stMetricValue"] { color:#3182f6; font-weight:800; }
div.stButton>button {
  background:#3182f6; color:#fff; border:0; border-radius:12px;
  padding:0.6rem 1.4rem; font-weight:700; }
div.stButton>button:hover { background:#1b64da; color:#fff; }
section[data-testid="stSidebar"] { background:#fff; border-right:1px solid #eef1f4; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta


try:
    model, scaler, meta = load_artifacts()
except Exception as e:
    st.error("model.pkl / scaler.pkl / metadata.json 파일이 필요합니다. "
             "노트북 8단계에서 생성해 같은 폴더에 두세요.")
    st.stop()

feats = meta["features"]
labels = meta["labels_ko"]
ranges = meta["ranges"]
use_scaler = meta["use_scaler"]


def predict(row_df):
    X = row_df[feats].values.astype(float)
    if use_scaler:
        X = scaler.transform(X)
    return model.predict(X)


def grade(mpa):
    if mpa < 20:  return "낮음 (비구조용)", "#e03131"
    if mpa < 40:  return "보통 (일반 구조용)", "#f08c00"
    if mpa < 60:  return "높음 (고강도)", "#2f9e44"
    return "매우 높음 (초고강도)", "#1971c2"


# ---------- header ----------
st.title("🧱 콘크리트 압축강도 예측기")
st.caption(f"모델: {meta['model_name']}  ·  Test R² = {meta['test_r2']:.3f}  ·  단위: kg/m³ (재령은 일)")

tab_pred, tab_batch, tab_info = st.tabs(["단일 예측", "배치 예측(CSV)", "모델 정보"])

# ---------- single prediction ----------
with tab_pred:
    st.subheader("배합 입력")
    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(feats):
        lo, hi, mean = ranges[f]
        step = 1.0 if (hi - lo) > 50 else 0.1
        vals[f] = cols[i % 4].slider(
            labels[f], float(round(lo, 1)), float(round(hi, 1)),
            float(round(mean, 1)), step=step)

    if st.button("압축강도 예측"):
        row = pd.DataFrame([vals])
        mpa = float(predict(row)[0])
        g, color = grade(mpa)
        c1, c2 = st.columns([1, 2])
        c1.metric("예측 압축강도", f"{mpa:.2f} MPa")
        c2.markdown(
            f"<div style='padding:1rem 1.2rem;background:{color}15;"
            f"border-left:4px solid {color};border-radius:8px;margin-top:8px;'>"
            f"<b style='color:{color}'>{g}</b></div>", unsafe_allow_html=True)

        # 물-시멘트비 참고 지표
        if vals.get("시멘트", 0) > 0:
            wc = vals["물"] / vals["시멘트"]
            st.caption(f"물-시멘트비(W/C) ≈ {wc:.2f}  (낮을수록 강도↑)")

# ---------- batch ----------
with tab_batch:
    st.subheader("CSV 업로드 → 일괄 예측")
    st.caption("필요 컬럼: " + ", ".join(feats))
    up = st.file_uploader("CSV 파일", type=["csv"])
    if up is not None:
        bdf = pd.read_csv(up)
        miss = [c for c in feats if c not in bdf.columns]
        if miss:
            st.error(f"누락된 컬럼: {miss}")
        else:
            bdf["predicted_strength_MPa"] = predict(bdf).round(2)
            st.dataframe(bdf, use_container_width=True)
            st.download_button(
                "결과 CSV 다운로드",
                bdf.to_csv(index=False).encode("utf-8-sig"),
                "predictions.csv", "text/csv")

# ---------- info ----------
with tab_info:
    st.subheader("변수 중요도")
    imp = meta.get("importances", {})
    if imp:
        s = pd.Series(imp).rename(index=labels).sort_values(ascending=False)
        st.bar_chart(s)
    else:
        st.info("선택된 모델은 변수 중요도를 제공하지 않습니다.")
    st.subheader("입력 변수 범위 (학습 데이터 기준)")
    rng = pd.DataFrame(
        [{"변수": labels[f], "최소": round(ranges[f][0], 1),
          "최대": round(ranges[f][1], 1), "평균": round(ranges[f][2], 1)} for f in feats])
    st.dataframe(rng, use_container_width=True, hide_index=True)
