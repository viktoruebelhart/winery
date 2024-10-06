import streamlit as st
import gdown
import tensorflow as tf
import io 
from PIL import Image 
import numpy as np
import pandas as pd
import plotly.express as px

@st.cache_resource
def load_model():
    #https://drive.google.com/file/d/1NSsQconZZViIPqI5Z-2tWqK1AVXoN-0h/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1NSsQconZZViIPqI5Z-2tWqK1AVXoN-0h'

    gdown.download(url, 'model_16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path ='model_16bits.tflite')
    interpreter.allocate_tensors()
    return interpreter
    

def load_image():
    uploaded_file = st.file_uploader('Drag and drop the image here or click to select an image', type=['png', 'jpeg', 'jpg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('The image was loaded successfully')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    
def forecast(interpreter,image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'],image)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['BlackMeasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probability (%)'] = 100*output_data[0]

    fig = px.bar(df,y='classes',x='probability (%)',orientation ='h', text='probability (%)',
                  title='Probability of grape disease class')
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classify vine leaves"
    )
    st.write("# Classify vine leaves!")
    #load model
    interpreter = load_model()
    #load image 
    image = load_image()
    #classify
    if image is not None:

        forecast(interpreter,image)

if __name__ == "__main__":
    main()