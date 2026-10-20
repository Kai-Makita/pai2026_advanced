from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ..config import ROOT


st.set_page_config(page_title="Mobile Manipulation IL/RL", layout="wide")
st.title("Mobile Manipulation: BC / PPO / BC→PPO")
runs = sorted((ROOT / "outputs").glob("*"), reverse=True)
selected = st.selectbox("Experiment", runs, format_func=lambda p: p.name if p else "none") if runs else None
if selected:
    st.code((selected / "config.yaml").read_text(encoding="utf-8") if (selected / "config.yaml").exists() else "config unavailable")
    for csv_file in selected.rglob("*.csv"):
        st.subheader(csv_file.stem); df = pd.read_csv(csv_file); st.dataframe(df)
        numeric = df.select_dtypes("number")
        if not numeric.empty: st.line_chart(numeric)
    for json_file in selected.rglob("*.json"):
        if json_file.name != "metadata.json":
            st.subheader(json_file.stem); st.json(json.loads(json_file.read_text(encoding="utf-8")))
    for image in selected.rglob("*.png"):
        st.image(str(image), caption=image.name)

