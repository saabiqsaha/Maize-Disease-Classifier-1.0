import streamlit as st
from keras.models import load_model
from PIL import Image

from utils import classify

#set title
st.title('Maize Disease Classifier')
st.text('by Mohammed Saha')

#set header
st.header('upload an image of a maize skin')


#upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

#load classifier
model = load_model('model/keras_model.h5')


#load classnames
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

#display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image)

    #classify image
    class_name, conf_score = classify(image, model, class_names)

    #write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}".format(conf_score))
