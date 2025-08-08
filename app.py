import  streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

model_path = os.path.join(os.path.dirname(__file__), "rfc.pkl")
rfc = pickle.load(open(model_path, 'rb'))

st.title('Forest Cover Type Prediction')

image= Image.open('img.jpeg')
st.image(image, use_container_width=True )

user_input = st.text_input('Enter all cover types features ')
if user_input:
    user_input = user_input.split(',')
    features = np.array([user_input],dtype= np.float64)
    prediction= rfc.predict(features).reshape(1,-1)
    prediction = int(prediction.item())

    cover_type_dict ={
        1:{'name': 'Spruce/Fir','image': 'img_1.jpeg'},
        2: {'name': 'Lodgepole Pine', 'image': 'img_2.png'},
        3: {'name': 'Ponderosa Pine', 'image': 'img_3.png'},
        4: {'name': 'Cottonwood/Willow', 'image': 'img_4.jpeg'},
        5: {'name': 'Aspen', 'image': 'img_5.jpeg'},
        6: {'name': 'Douglas-fir', 'image': 'img_6.png'},
        7: {'name': 'Krummholz', 'image': 'img_7.png'},
    }
    cover_type_info=cover_type_dict.get(prediction)

    if cover_type_info is not None:
        forest_name = cover_type_info['name']
        forest_image = cover_type_info['image']

        col1,col2=st.columns([2,3])
        with col1:
            st.write('This is predict cover type')
            st.write(f"<h1 style= 'font-size:50px; font-weight:bold'>{forest_name}</h1>",unsafe_allow_html=True)

        with col2:
            final_image=Image.open(forest_image)
            st.image(final_image,caption=forest_name,use_container_width =True)
