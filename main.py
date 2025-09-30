import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify

# Set title
st.title('Maize Disease Classifier')

# Add description/context
st.markdown("""
### About This App
I built this deep learning application to help farmers detect diseases in maize plants using computer vision. 
The model was trained on thousands of images of healthy and diseased maize leaves to provide instant, accurate diagnoses.

**How it works:**
1. Upload an image of a maize leaf
2. The AI model analyzes the image
3. Get instant disease detection results with confidence scores

This tool can help farmers identify crop health issues early and take appropriate action to protect their harvest.
""")

# Set header
st.header('Upload an image of a maize leaf')
st.text('by Mohammed Abdulai')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load classifier
model = load_model('model/keras_model.h5')

# Load class names
with open('model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# Display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Maize Leaf Image', use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## Disease Detection Result: {}".format(class_name))
    st.write("### Confidence Score: {:.2%}".format(conf_score))
    
    # Add interpretation help
    if conf_score > 0.90:
        st.success("High confidence prediction - the model is very certain about this diagnosis.")
    elif conf_score > 0.70:
        st.warning("Moderate confidence prediction - consider consulting an agricultural expert for confirmation.")
    else:
        st.error("Low confidence prediction - please upload a clearer image or consult an expert.")