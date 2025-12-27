# importing the required libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# function for processing CNN model prediction
def model_prediction(test_image):
    # loading the trained model and preprocessing the image
    model = tf.keras.models.load_model("./trained_model/plant_disease_trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    # predicting the class
    prediction = model.predict(input_arr, verbose=0)
    result_index = np.argmax(prediction)
    max_value = np.max(prediction)
    # returning the predicted class index and confidence score
    return result_index, max_value


# setting the page meta title
st.set_page_config(page_title='Automated Plant Disease Classification', page_icon="🔬",
                   layout="wide")

# sidebar navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.radio("Select Page",["Overview","Technical Specifications", "Model Performance", "Plant Disease Model", "Future Works"])


# Main Page
if(app_mode=="Overview"):
    # Displaying the header and introductory information
    st.header("Automated Plant Disease Classification System🌱")
    st.markdown("##### Powered by Deep Learning (CNN) & Keras")
    st.markdown("[Link to the Source Code on GitHub](https://github.com/Oluwatobi-coder/plant-disease-prediction-using-deep-learning-cnn-and-keras)")
    st.markdown("**Author**: Bello Oluwatobi")
    st.markdown("**Last Updated on**: 26th December, 2025")
    # adding a visual separator
    st.markdown("---")
    image_1 = Image.open("./intro_image.jpg")
    new_size = (600, 250) 
    # resizing the image
    resized_image = image_1.resize(new_size)
    st.image(resized_image, width="stretch")
    # introductory text
    st.markdown("""
    **Welcome to the Automated Plant Disease Classification System!**
    
    I developed this system to apply computer vision to the challenge of crop health. This project focuses on the automated detection and classification of 38 distinct plant diseases, providing a scalable way to ensure large-scale agriculture sustainability through deep learning.

    ### Usage Instructions
    1.	**Data Input:** Navigate to the **Plant Disease Model** and upload a high-resolution image of an affected plant for analysis.
    2.	**Automated Processing:** The CNN architecture analyzes the image features to detect plant health condition.
    3.	**Classification Output:** View the model’s classification of the plant and the confidence level of the prediction.

    """)

# Technical Specifications
elif(app_mode=="Technical Specifications"):
    # displaying the technical specifications of the dataset and model
    st.header("Technical Specifications")
    # content about dataset and model
    st.markdown("""
                #### About The Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this [github repository](https://github.com/spMohanty/PlantVillage-Dataset).
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)
    st.markdown("""
                #### About The Model
                This project utilizes a Sequential Convolutional Neural Network (CNN) architecture for high-dimensional image classification. . The architecture consists of five hierarchical blocks of convolutional layers, progressively increasing in depth from 32 to 512 filters. This allows the model to capture everything from simple edges and textures in the initial layers to complex patterns in the deeper layers.
                To ensure the model generalizes well to new, unseen data, I integrated Dropout layers (25 percent and 40 percent). This prevents "overfitting," where a model simply memorizes the training data rather than learning to identify actual diseases. The model was compiled using the Adam optimizer with a fine-tuned learning rate (0.0001), ensuring the model converges smoothly to reach its best possible performance.
                """)

    st.markdown("""
                #### Model Scope
                """)
    st.markdown("""
    This model is optimized for **38 specific health conditions** (Diseased & Healthy) across **14 plant species**. Performance is validated 
    exclusively for these training categories.
    """)
    # detailed list of the classes
    with st.expander("Detailed List of the 38 Plant Health Condition Classes", expanded=True):
        st.write("The model identifies the following specific conditions:")
        # organizing classes by species for readability
        classes = {
            "Apple": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"],
            "Blueberry": ["Healthy"],
            "Cherry": ["Powdery Mildew", "Healthy"],
            "Corn (Maize)": ["Cercospora Leaf Spot", "Common Rust", "Northern Leaf Blight", "Healthy"],
            "Grape": ["Black Rot", "Esca (Black Measles)", "Leaf Blight", "Healthy"],
            "Orange": ["Citrus Greening (Huanglongbing)"],
            "Peach": ["Bacterial Spot", "Healthy"],
            "Pepper (Bell)": ["Bacterial Spot", "Healthy"],
            "Potato": ["Early Blight", "Late Blight", "Healthy"],
            "Raspberry": ["Healthy"],
            "Soybean": ["Healthy"],
            "Squash": ["Powdery Mildew"],
            "Strawberry": ["Leaf Scorch", "Healthy"],
            "Tomato": [
                "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold", 
                "Septoria Leaf Spot", "Spider Mites", "Target Spot", 
                "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
            ]
        }
        # displaying the classes in a structured format
        st.json(classes) 

# Model Performance
elif(app_mode=="Model Performance"):
    # displaying the model performance metrics and learning curves
    st.header("Model Performance")
    # displaying key performance metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("**Training Accuracy**", "99.13%")
    col2.metric("**Training Loss**", "2.72%")
    col3.metric("**Validation Accuracy**", "96.67%")
    col4.metric("**Validation Loss**", "11.57%")
    # adding a visual separator
    st.divider()
    # displaying learning curves
    st.subheader("Training & Validation Accuracy/Loss Curves")
    st.subheader("Model Learning Curves")
    st.write("The curves below show the convergence of the CNN architecture over 10 epochs.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("./model_results_images/accuracy_results.png")
        st.markdown("<p style='text-align: center; font-weight: bold;'>Fig 1: Training and Validation Accuracy Curves over 10 Epochs</p>", unsafe_allow_html=True)
        st.markdown("**Observation:** The training accuracy steadily increases, indicating effective learning. The validation accuracy closely follows, suggesting good generalization without overfitting.")
    with col2:
        st.image("./model_results_images/loss_results.png")
        st.markdown("<p style='text-align: center; font-weight: bold;'>Fig 2: Training and Validation Loss Curves over 10 Epochs</p>", unsafe_allow_html=True)
        st.markdown("**Observation:** Both training and validation loss decrease over epochs, indicating that the model is effectively minimizing error on both datasets.")

# Prediction Page
elif(app_mode=="Plant Disease Model"):
    # displaying the plant disease prediction interface
    st.header("Disease Recognition")
    # file uploader for plant image
    uploaded_image = st.file_uploader("**Choose a Plant Image:**", type=["jpg","jpeg","png"])

    # analyzing the uploaded image
    if uploaded_image is not None:
        st.image(uploaded_image, width=250)
        time.sleep(1) # simulating loading time
        st.success("Image loaded successfully! Click **Analyze Image** to analyze.")
        if(st.button("Analyze Image")):
            with st.spinner('processing image...'):
                time.sleep(2)  # simulating a delay for processing
                # getting model prediction and confidence score
                result_index, max_value = model_prediction(uploaded_image)
                # reading class names
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy']
            # displaying the prediction results
            st.write("Model Prediction")
            predicted_class = f"Class: **{class_name[result_index]}**"
            confidence_text = f"Confidence Score: **{max_value:.2%}**"
            st.success(f"{predicted_class} \n\n {confidence_text}")
    else:
        st.warning("⚠️ No file detected. Please upload an image to enable analysis.")
        st.button("Analyze Image", disabled=True, help="Please upload an image first.")
    st.markdown("---")
    # providing sample test images for download and use
    st.markdown("### Download Sample Test Images")
    st.markdown("You can download the sample images to test the model:")
    st.markdown("[Download Sample Test Images](https://drive.google.com/drive/folders/1FDVDvcanXH1_muWZ4aV7LuzXFQT4bW32?usp=sharing)")


# Future Works
elif(app_mode=="Future Works"):
    # displaying future works and improvements
    st.header("Future Works & Improvements")
    st.write("Presented in this section are areas for further improvement on this project:")
    
    st.write("""
        **1. Dataset Expansion:** Incorporating additional crops (e.g., Grapes, Corn) to enhance model robustness.
        """)

    st.write("""
        **2. Real-Time Detection:** Integrating the model with a live video feed to allow for real-time disease spotting via a live drone or smartphone camera feed.
        """)

    st.write("""
        **3. Treatment Recommendation System:** Connecting the model to a treatment recommendation system to provide tailored organic and chemical treatment recommendations.
        """)