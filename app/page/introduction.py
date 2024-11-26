import matplotlib.pyplot as plt
import os
import json
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import streamlit as st

# import neeeded libs for the visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image

def check_current_work_dir():
    if not os.path.isfile("setup.py") or  os.path.basename(os.getcwd()).endswith('notebooks'):
        print("Changing the current directory to the parent directory containing the setup.py file")

        # move one folder up
        os.chdir("..")
        print(f"New current directory: {os.getcwd()}, it will remain this working directory for the rest of the notebook")

    if not os.path.isfile("setup.py"):
        raise Exception("setup.py not found in the current directory")
check_current_work_dir()


df = pd.read_csv("data/interim/overview.csv")

st.set_page_config(layout="wide") 

"""
# Introduction to the ZooLake dataset

The ZooLake dataset is an open data project from Eawag that aims to automate the classification of 35 different lake plankton species using deep learning and other machine learning algorithms. The objective of the image classification is to enable the monitoring of the different plankton populations over time, as plankton are effective indicators of environmental change and ecosystem health in freshwater habitats.
"""
col1, col2 = st.columns([1, 1])

with col1:
    

    """
    

    ##### Data Collection
    The collection of images of plankton is an ongoing process, with the objective of improving the classification through the addition of more images. the most recent images that have not yet been manually labelled by a taxonomist can be viewed at the webpage [Aquascope](https://aquascope.ch/)  under the heading 'Latest Greifensee.' The new unlabelled images are being labelled manually by taxonomists over time. Once a sufficient number of images have been labelled or used for a scientific paper, a new labelled collection is published. These diffrent collections of labeld images represent the ZooLake versions. 

    ##### Open Data
    As of the present date, 20 September 2024, they are a two version of the ZooLake data set aviable for the public on the eawag open research page eric. This inculde following Versions:

    - [ZooLake](https://opendata.eawag.ch/dataset/deep-learning-classification-of-zooplankton-from-lakes),  is the initial version of the dataset referenced by the paper 'Deep Learning Classification of Lake Zooplankton' with a tota of 17900 labelled images.

    - [Zoolake2.0](https://data.eawag.ch/dataset/data-for-producing-plankton-classifiers-that-are-robust-to-dataset-shift), second version of the data set, which include more labelled data and the introduction of the *out-of-dataset (OOD)*. The OOD was utilised by C. Cheng et al. (2024) in their research into producing plankton classifiers that are robust to dataset shift. It also mentioned there,  that the ZooLake2.0 images come with a 2-year gap of the fist ZooLake Version and a total of 24'000 images

    -  [Zoolake3.0](https://opendata.eawag.ch/) incomming...

    """


with col2:
    df = pd.read_csv("data/interim/overview.csv")

    with open("class_map.json") as f:
        class_map = json.load(f)


    # prepare categories

    class_order = list(class_map.keys())
    class_labels = {str(k): v for k, v in class_map.items()}


    st.markdown("##### Image of a plankton")
    selected_class = st.selectbox("Select a class to display an image:", options=class_order)

    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = {}


    if selected_class not in st.session_state["selected_images"]:
        # Randomly sample one image for the selected class and store it
        random_image = df[df["class"] == selected_class].sample(1).to_dict(orient="records")[0]
        st.session_state["selected_images"][selected_class] = random_image

    # Retrieve the stored random image for the selected class
    random_image = st.session_state["selected_images"][selected_class]
    # Display the image
    try:
        
        image_path = os.path.join("data", "interim","ZooLake", random_image["class"], random_image["image"])
        image = Image.open(image_path)
        s_col1, s_col2, s_col3 = st.columns([1,6,1])

        with s_col1:
            st.write("")

        with s_col2:
            st.image(image, caption=f"Random image of {selected_class}")

        with s_col3:
            st.write("")
        
    except FileNotFoundError:
        st.write("No image available for this class.")


















        
   







