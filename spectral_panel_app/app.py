"""
Streamlit app for the spectral flow cytometry panel selection tool.

Run locally with:
    streamlit run app.py

Deployed for the lab at (fill in once deployed):
    https://<your-app-name>.streamlit.app
"""

from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

from panel_selector import (
    load_spectra,
    classify_fluorophores,
    select_panel,
    summarize_panel,
    plot_spectra,
    plot_panel,
)

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "Unmixing_Matrix_20260715_unmixing_all.xlsx"


def _save_fig_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    return buf.getvalue()

st.set_page_config(page_title="Spectral Panel Picker", layout="wide")

st.title("🔬 Spectral Flow Cytometry Panel Picker")
st.caption(
    "Pick the N most spectrally-distinct fluorophores from the lab's measured "
    "library (fluorescent proteins + HaloTag/SNAPtag dyes), balancing average "
    "separation against worst-case separation."
)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load(file_bytes_or_path):
    return load_spectra(file_bytes_or_path)


with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader(
        "Upload an unmixing-matrix export (.xlsx)",
        type=["xlsx"],
        help="Same format as the CytExpert/CytoFLEX export: a 'Channel' column "
             "followed by one column per fluorophore. If you don't upload one, "
             "the lab's default library is used.",
    )

if uploaded is not None:
    spectra, control = _load(uploaded)
    data_source = f"uploaded file: **{uploaded.name}**"
else:
    if not DEFAULT_DATA_PATH.exists():
        st.error(
            f"No file uploaded, and no default library found at "
            f"`{DEFAULT_DATA_PATH}`. Please upload a file."
        )
        st.stop()
    spectra, control = _load(DEFAULT_DATA_PATH)
    data_source = "the lab's default library"

fp_names_all, dye_names_all = classify_fluorophores(spectra)
all_names = fp_names_all + dye_names_all

st.sidebar.success(
    f"Loaded {len(fp_names_all)} FPs + {len(dye_names_all)} dyes "
    f"across {spectra.shape[0]} channels, from {data_source}."
)


# ---------------------------------------------------------------------------
# 2. Panel settings
# ---------------------------------------------------------------------------

st.sidebar.header("2. Panel settings")

max_n = len(all_names)
N = st.sidebar.number_input(
    "N -- total colors in panel", min_value=2, max_value=max_n, value=min(8, max_n), step=1
)
M = st.sidebar.number_input(
    "M -- max HaloTag/SNAPtag dyes allowed", min_value=0, max_value=len(dye_names_all),
    value=min(1, len(dye_names_all)), step=1,
)

st.sidebar.header("3. Fine-tuning (optional)")

required_names = st.sidebar.multiselect(
    "Required fluorophores (already committed to)",
    options=all_names,
    help="These are always included in the panel; the search only picks the "
         "remaining N minus this-many colors.",
)
excluded_names = st.sidebar.multiselect(
    "Excluded fluorophores (not usable)",
    options=[n for n in all_names if n not in required_names],
    help="These are removed from consideration entirely.",
)

balance = st.sidebar.slider(
    "Objective balance: worst-case \u2194 average separation",
    min_value=0.0, max_value=1.0, value=0.5, step=0.05,
    help="0 = purely maximize the minimum pairwise distance (protect the "
         "hardest-to-unmix pair). 1 = purely maximize the average pairwise "
         "distance. 0.5 = balance both.",
)
w_min, w_avg = 1.0 - balance, balance

with st.sidebar.expander("Advanced"):
    use_median = st.checkbox("Use median instead of mean for the 'average' term", value=False)
    seed = st.number_input("Random seed", min_value=0, value=0, step=1)
    n_restarts = st.number_input(
        "Heuristic search restarts (larger N only)", min_value=20, max_value=2000, value=200, step=20
    )

run = st.sidebar.button("Find best panel", type="primary", width='stretch')


# ---------------------------------------------------------------------------
# 3. Run + display
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["result"] = None

if run:
    overlap = set(required_names) & set(excluded_names)
    if overlap:
        st.error(f"These fluorophores are both required and excluded: {sorted(overlap)}")
    else:
        try:
            with st.spinner("Searching for the best panel..."):
                result = select_panel(
                    spectra, N=int(N), M=int(M),
                    fp_names=fp_names_all, dye_names=dye_names_all,
                    required_names=required_names, excluded_names=excluded_names,
                    w_avg=w_avg, w_min=w_min, use_median=use_median,
                    n_restarts=int(n_restarts), seed=int(seed),
                )
            st.session_state["result"] = result
        except ValueError as e:
            st.error(str(e))
            st.session_state["result"] = None

result = st.session_state["result"]

if result is not None:
    st.subheader("Selected panel")

    key = "mean_dist" if "mean_dist" in result else "median_dist"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Colors", len(result["panel"]))
    c2.metric("Mean/median pairwise distance", f"{result[key]:.3f}")
    c3.metric("Min pairwise distance", f"{result['min_dist']:.3f}")
    c4.metric("Search type", "exact" if result["exact"] else "heuristic")

    st.write(", ".join(f"**{n}**" for n in result["panel"]))

    tab1, tab2, tab3 = st.tabs(["Spectra plot", "Distance heatmap", "Details / export"])

    with tab1:
        fig, _ = plot_spectra(spectra, result["panel"])
        st.pyplot(fig, width='stretch')
        st.download_button(
            "Download spectra plot (PNG)",
            data=_save_fig_bytes(fig),
            file_name=f"panel_spectra_{N}N_{M}M.png",
            mime="image/png",
        )

    with tab2:
        fig2, _ = plot_panel(result)
        st.pyplot(fig2, width='stretch')
        st.download_button(
            "Download heatmap (PNG)",
            data=_save_fig_bytes(fig2),
            file_name=f"panel_matrix_{N}N_{M}M.png",
            mime="image/png",
        )

    with tab3:
        st.text(summarize_panel(result))
        st.dataframe(result["dist_matrix"].round(3), width='stretch')
        st.download_button(
            "Download distance matrix (CSV)",
            data=result["dist_matrix"].to_csv().encode("utf-8"),
            file_name=f"panel_distances_{N}N_{M}M.csv",
            mime="text/csv",
        )
else:
    st.info("Set your panel size (N) and dye cap (M) in the sidebar, then click **Find best panel**.")
