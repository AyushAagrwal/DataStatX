# Description: This is a test file to test the functionality of the DataStatX application.

import streamlit as st
import pandas as pd
from lida import Manager, TextGenerationConfig, llm
from dotenv import load_dotenv
import os
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import openai
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

# Initialize LIDA
lida = Manager(text_gen=llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

# Set page config , such as title, favicon, layout, etc.
st.set_page_config(
    page_title="DataStatX",
    page_icon="./logo.ico",
    # layout="wide",
    initial_sidebar_state="auto",
)

#
# with open("style.css") as f:
#     css = f.read()
#     st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar
menu = st.sidebar.selectbox("Choose an Option", ["Data Analysis", "Question based Graph"])

# Main content
if menu == "Data Analysis":
    logger.info("Data Analysis option selected.")
    st.subheader("Basic Statistical Analysis of your Data")
    # File uploader
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        try:
            logger.info("CSV file uploaded.")
            df = pd.read_csv(file_uploader)
            # Filter out numerical columns
            numerical_columns = df.select_dtypes(include=['number'])


            # Display the data
            st.write("### Summary Statistics:")
            st.write(numerical_columns.describe().round(3))

            # Display mean, median and mode
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("### Mean:")
                st.write(numerical_columns.mean().round(3))
            
            with col2:
                st.write("### Median:")
                st.write(numerical_columns.median().round(3))

            with col3:
                st.write("### Mode:")
                mode_data = df.mode().iloc[0]
                st.write(mode_data)
            
            # Display correlation graph
            st.write("### Correlation Graph:")
            corr = numerical_columns.corr()
            fig = px.imshow(corr,
                            labels=dict(x="Features", y="Features", color="Correlation", color_continuous_scale=px.colors.sequential.Viridis, template="plotly_dark", xgap=1, ygap=1, aspect="auto"),
                            x=corr.columns,
                            y=corr.columns)
            #update layout
            fig.update_layout(width=800, height=800)
            st.plotly_chart(fig)

        # Handle exceptions
        except (IndexError,Exception) as e:
            logger.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            st.error("Sometimes I cant perform certain tasks or try to change the logic of the question.", "Please try again.", "If the problem persists, please contact the developer.", "Thank you.")

# Question based Graph
elif menu == "Question based Graph":
    logger.info("Question based Graph option selected.")
    st.subheader("Write a Query to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        try:
            logger.info("CSV file uploaded.")
            path_to_save = "uploaded_csv.csv"
            with open(path_to_save, "wb") as f:
                f.write(file_uploader.getvalue())
            text_area = st.text_area("Query your Data to Generate Graph", height=200)
            if st.button("Generate Graph"):
                if len(text_area) > 0:
                    st.info("Your Query: " + text_area)
                    try:
                        # using the summarize method to generate the summary
                        summary = lida.summarize("uploaded_csv.csv", summary_method="default", textgen_config=textgen_config)
                        user_query = text_area
                        #using the visualize method to generate the graph
                        charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                        charts[0]
                        # Convert the base64 string to image
                        image_base64 = charts[0].raster
                        img = base64_to_image(image_base64)
                        with st.spinner("Generating Graph..."):
                            #resize the image
                            img = img.resize((1200, 800))
                            st.image(img)
                    # Handle OpenAI API errors
                    except openai.error.APIError as api_error:
                        logger.error(f"OpenAI API error occurred: {str(api_error)}")
                        st.error(f"OpenAI API error occurred: {str(api_error)}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
