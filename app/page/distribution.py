
import os

import logging
import json

from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

logger = logging.getLogger(__name__)  



st.set_page_config(layout="wide") 
# load diffrent components
@st.cache_data
def load_data():
    df = pd.read_csv("data/interim/overview.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
@st.cache_data
def load_class_elements():
    with open("class_map.json") as f:
        class_map = json.load(f)
    class_order = list(class_map.keys())
    class_labels = {str(k): v for k, v in class_map.items()}

    return class_order, class_labels


df = load_data()

class_order, class_labels = load_class_elements()

# prepare categories



@st.cache_data
def prepare_class_counts(data, class_order):
    counts = data["class"].value_counts().reindex(class_order, fill_value=0).reset_index()
    counts.columns = ["class", "count"]
    return counts

unfiltered_class_counts = prepare_class_counts(df, class_order)



# check the current working directory

def check_current_work_dir():
    if not os.path.isfile("setup.py") or  os.path.basename(os.getcwd()).endswith('notebooks'):
        print("Changing the current directory to the parent directory containing the setup.py file")

        # move one folder up
        os.chdir("..")
        print(f"New current directory: {os.getcwd()}, it will remain this working directory for the rest of the notebook")

    if not os.path.isfile("setup.py"):
        raise Exception("setup.py not found in the current directory")
check_current_work_dir()


st.markdown("# Distribution of the plankton species")            

@st.cache_data
def prepare_class_counts(data, class_order):
    counts = data["class"].value_counts().reindex(class_order, fill_value=0).reset_index()
    counts.columns = ["class", "count"]
    return counts

unfiltered_class_counts = prepare_class_counts(df, class_order)

# Display the plot
col1, col2 = st.columns([1, 3])

with col1:
    """
    The distirbution of the classes in the dataset faces the common challenge of class imblanace,
    as some species are more frequent than others. The imbalance of the data is influenced on the 
    naturaly abudaunce of the planktons species.
    """
with col2:
    # goupby date and class
    df_grouped = df.groupby(["date", "class"]).size().reset_index(name="count")





    # plot time series of number of images over time
    fig = px.bar(
        df_grouped,
        x="date",
        y="count",
        color="class",
        labels={"class": "class", "count": "count"},
    )  

    st.plotly_chart(fig)


    def calculate_class_counts(data, class_order, end_date):
        # Filtere direkt in der Funktion und berechne die Counts
        filtered_data = data[data["date"] <= end_date]
        counts = filtered_data["class"].value_counts().reindex(class_order, fill_value=0).reset_index()
        counts.columns = ["class", "count"]
        return counts

    # Initialisierung mit ungefilterten Counts
    if "end_date" not in st.session_state or "class_counts" not in st.session_state:
        st.session_state["end_date"] = df["date"].max()  # Initialisiere mit maximalem Datum
        st.session_state["class_counts"] = calculate_class_counts(df, class_order, st.session_state["end_date"])

    # Slider zur Auswahl des Enddatums (unterhalb des Plots)
    new_end_date = st.slider(
        "Select an end date:",
        min_value=df["date"].min(),
        max_value=df["date"].max(),
        value=st.session_state["end_date"],
        format="YYYY-MM-DD",
        key="end_date_bottom"
    )

    # Aktualisiere `class_counts`, wenn sich das Enddatum ändert
    if new_end_date != st.session_state["end_date"]:
        st.session_state["end_date"] = new_end_date
        st.session_state["class_counts"] = calculate_class_counts(df, class_order, st.session_state["end_date"])

    class_counts = st.session_state["class_counts"]

    # Plotting the categorical distribution using Plotly bar chart
    fig = px.bar(
        class_counts,
        x="count",
        y="class",
        text_auto=True,
        labels={"class": "class", "count": "count"},
        title= f"Distribution of plankton classes {st.session_state['end_date']}"
    )

    # y axis adjustments
    fig.update_yaxes(title_text="Class")
    fig.update_yaxes(categoryorder='total ascending') 
    fig.update_layout(yaxis={'dtick':1})
    fig.update_traces(textposition='outside')

    # x axis adjustments
    fig.update_xaxes(title_text="Count")
    fig.update_xaxes(range=[0, unfiltered_class_counts["count"].max()+100]) 

    # plot layout adjustments
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))   

    st.plotly_chart(fig)

# Funktion zur Berechnung der class_counts

