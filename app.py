import streamlit as st
import pandas as pd
import json
import warnings
from io import BytesIO
import tempfile
import os
import textwrap
import asyncio
import subprocess
from PIL import Image
import base64
import io
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def process_query(image_file):

    pil_image = Image.open(image_file)

    base64_image = image_to_base64_str(pil_image)
    vertexai.init(project="documind-419411", location="us-central1")
    model = GenerativeModel("gemini-experimental")
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }    

    text1 = """You are a skilled UI/UX designer, adept at analyzing images of websites, 
    product pages, and mobile applications. You will be provided with images of websites, 
    product pages, and mobile applications and you will be asked to critique them and 
    provide an educated response to the design of the websites, product pages, and 
    mobile applications. You will respond by saying whether an uploaded image of a 
    website, product page, or mobile application is of a good quality design or a poor quality. 
    In either case, provide recommendations on how the website, product page, or mobile 
    application can be improved. I want you to be as critical as possible. 
    If a website, product page, or mobile application is of a poor design or quality say \"Poor Design\" 
    or if a website, product page, or mobile application is of good quality, say \"Good Design\""""

    text2 = """Below is an example of how I want the output to be formatted:
    ```
    Wayfair Website Analysis: Good Design with Room for ImprovementPositives:
    Clean and organized layout: The website utilizes a grid layout effectively, making it easy to scan and find different categories and promotions.Clear navigation: The top navigation bar is comprehensive, categorizing items logically and offering a prominent search bar.Strong visuals: High-quality images showcase products and create an appealing aesthetic.Effective use of promotions: The \"Suite Savings\" banner and other promotional sections are well-placed and visually engaging.Recommendations for improvement:
    Reduce visual clutter: The homepage feels slightly overloaded with various promotions and categories. Consider simplifying the layout and prioritizing key offerings.Enhance product information: While the images are attractive, the product descriptions are limited. Adding brief descriptions or key features would provide more context and encourage clicks.Improve mobile responsiveness: While the desktop version appears well-organized, ensuring optimal mobile responsiveness is crucial for accessibility and user experience.Personalization: Consider incorporating personalized recommendations or recently viewed items to enhance user engagement and conversion.Additional considerations:
    Testing different layouts and designs: A/B testing can provide valuable insights into user preferences and help optimize the homepage for better conversion rates.Accessibility: Ensure the website adheres to accessibility standards for users with disabilities.Overall, Wayfair\'s website demonstrates good design principles, but implementing the suggested improvements can further enhance the user experience and drive better results.
    ```"""   

    responses = model.generate_content(
        [text1, base64_image, text2],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    return responses

async def main():
    st.title("👨‍💻 UXify")

    st.write("Upload an image, get instant results.")

    # File upload and processing
    image_file = st.file_uploader("Upload an image.")
    if image_file is not None:
        print(image_file.name)
        st.image(image_file, caption='Uploaded Image', use_column_width=True)

    # Form for input field and button
    with st.form(key='query_form'):

        # google_key = st.text_input("Google API Key.")
        # GOOGLE_API_KEY = google_key

        # Submit button
        submit_button = st.form_submit_button(label="Let's Begin", help="Click to start analysis")
           
    # Process query if button is clicked
    if submit_button:
        print("Submit button clicked.")
        return_resp = process_query(image_file)
        for i, response in enumerate(return_resp, start=1):
            st.markdown(f"Response {i}: {to_markdown(response.text)}")

if __name__ == "__main__":
   asyncio.run(main())