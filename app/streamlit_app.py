import streamlit as st


start_page = st.Page(
    "start.py", title="Start Page",  default=True
)
dataset = st.Page(
    "page/introduction.py", title="Introduction"
)

distribution = st.Page("page/distribution.py", title="Distributions of the ZooLake dataset")

splits = st.Page("page/splits.py", title="Split overview")


pg = st.navigation(
        {
            "Start": [start_page],
            "Dataset": [dataset,distribution,splits],
        }
    )


pg.run()