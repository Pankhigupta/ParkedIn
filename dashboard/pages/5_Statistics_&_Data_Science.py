"""Academic analytics placeholder: Statistics & Data Science."""
import streamlit as st

from _setup import check_ready

st.set_page_config(page_title="ParkedIn: Statistics & Data Science", layout="wide")
st.title("Statistics & Data Science")
st.caption("Descriptive statistics, data quality, distributions, and methodological context.")
check_ready(st)

with st.container(border=True):
    st.subheader("Analytics workspace")
    st.info("This section is reserved for the academic analytics layer. Content will be added without changing the existing forecasting or pricing workflows.")
