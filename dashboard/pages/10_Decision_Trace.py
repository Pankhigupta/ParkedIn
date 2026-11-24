"""Evidence trail from source data through forecasting to pricing decisions."""
import streamlit as st

from _setup import ROOT  # noqa: F401
from analytics.academic_story import gather_academic_evidence, decision_trace

st.set_page_config(page_title="ParkedIn: Decision Trace", layout="wide")
st.title("Decision Trace")
st.caption("A traceable path from observed parking conditions to the system's price recommendation.")

try:
    evidence = gather_academic_evidence()
except Exception as exc:
    st.error(f"Academic evidence could not be loaded from the project artifacts: {exc}")
    st.stop()

trace = decision_trace(evidence)
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

if go is not None:
    stages = [row["stage"] for row in trace]
    short = ["Observed parking\nproblem", "Statistical\ntests", "EDA\npatterns", "Existing\nfeatures",
             "Temporal\nstructure", "Model\nselection", "Held-out\nevaluation", "Price\nrecommendation"]
    fig = go.Figure()
    for i, label in enumerate(short):
        y = len(short) - i
        fig.add_shape(type="rect", x0=0.16, x1=0.84, y0=y - .34, y1=y + .34,
                      line=dict(color="#244c66", width=1.5), fillcolor="#edf4f8")
        fig.add_annotation(x=.5, y=y, text=f"<b>{i+1}. {label}</b>", showarrow=False,
                           font=dict(size=13, color="#17354a"))
        if i < len(short)-1:
            fig.add_annotation(x=.5, y=y-.49, text="", showarrow=True,
                               axref="x", ayref="y", ax=.5, ay=y-.65,
                               arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#53778a")
    fig.update_layout(height=700, margin=dict(l=20,r=20,t=10,b=10),
                      xaxis=dict(visible=False, range=(0,1)), yaxis=dict(visible=False, range=(.4,8.6)),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Interpret the pipeline chart", expanded=False):
        st.markdown("**What is shown?** The eight stages that carry the actual project data through to pricing.\n\n"
                    "**Why does it matter?** It makes the analytical dependencies and hand-offs visible.\n\n"
                    "**How did it influence project decisions?** The detailed evidence records below connect each stage's finding to its data and decision.")
else:
    st.warning("Install Plotly (`pip install -r requirements.txt`) to render the pipeline diagram. The evidence trail remains available below.")

st.subheader("Stage evidence")
for idx, row in enumerate(trace, start=1):
    with st.expander(f"{idx}. {row['stage']}", expanded=(idx == 1)):
        c1, c2 = st.columns(2)
        c1.markdown("**Finding**\n\n" + row["finding"])
        c2.markdown("**Evidence**\n\n" + row["evidence"])
        st.markdown("**Decision**\n\n" + row["decision"])

st.caption("Statistical associations are descriptive evidence, not causal effects. Revenue figures use the project's configured elasticity scenarios and are estimates, not observed transactions.")
