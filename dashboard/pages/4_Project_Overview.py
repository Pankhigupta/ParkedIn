"""Academic analytics placeholder: Project Overview."""
import streamlit as st

from _setup import check_ready

st.set_page_config(page_title="ParkedIn: Project Overview", layout="wide")
st.title("Project Overview")
st.caption("Overview of the parking-pricing problem, dataset, system architecture, and research objectives.")
check_ready(st)

with st.container(border=True):
    st.subheader("Analytics workspace")
    st.info("This section is reserved for the academic analytics layer. Content will be added without changing the existing forecasting or pricing workflows.")
