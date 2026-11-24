"""Academic analytics placeholder: Model Comparison."""
import streamlit as st

from _setup import check_ready

st.set_page_config(page_title="ParkedIn: Model Comparison", layout="wide")
st.title("Model Comparison")
st.caption("A future workspace for comparing saved forecast and pricing evaluation results.")
check_ready(st)

with st.container(border=True):
    st.subheader("Analytics workspace")
    st.info("This section is reserved for the academic analytics layer. Content will be added without changing the existing forecasting or pricing workflows.")
