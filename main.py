import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd
from random import shuffle
import time
import matplotlib.pyplot as plt
from PIL import Image

from io import BytesIO

# Page Icon
icon = Image.open("./Design/icon.png")
st.set_page_config(page_title="Art Stylelify", page_icon=icon)

# st.set_page_config("Art Stylelify", page_icon="💃🏻")

# New Line Function
def new_line(n=1):
    for _ in range(n):
        st.markdown('\n')

# Tensor to Image Function
def tensor_to_image(tensor):
  '''converts a tensor to an image'''
  tensor_shape = tf.shape(tensor)
  number_elem_shape = tf.shape(tensor_shape)
  if number_elem_shape > 3:
    assert tensor_shape[0] == 1
    tensor = tensor[0]
  return tf.keras.preprocessing.image.array_to_img(tensor) 

# Load Image Function
def load_img(path_to_img):
  '''loads an image as a tensor and scales it to 512 pixels'''
  max_dim = 512
  image = tf.io.read_file(path_to_img)
  image = tf.image.decode_jpeg(image)
  image = tf.image.convert_image_dtype(image, tf.float32)

  shape = tf.shape(image)[:-1]
  shape = tf.cast(tf.shape(image)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  image = tf.image.resize(image, new_shape)
  image = image[tf.newaxis, :]
  image = tf.image.convert_image_dtype(image, tf.uint8)

  return image

# Load Images Function
def load_images(content_path, style_path):
  '''loads the content and path images as tensors'''
  content_image = load_img("{}".format(content_path))
  style_image = load_img("{}".format(style_path))

  return content_image, style_image

# Session State Function
def session_state():
    if 'content_img_name' not in st.session_state:
        st.session_state.content_img_name = None
    if 'style_img_name' not in st.session_state:
        st.session_state.style_img_name = None
    
    if 'imgs_path_content' not in st.session_state:
        st.session_state.imgs_path_content = None

    if 'imgs_path_style' not in st.session_state:
        st.session_state.imgs_path_style = None

    if 'style_uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    
    if 'content_uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

# Images Name Per Artist Function
def imgs_name_per_artist(artist):
    imgs = [i.split(".")[0].split("- ")[1] for i in os.listdir(path) if i.startswith(artist)]
    return imgs

session_state()
model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Logo
col, col2, col3 = st.columns([0.4,1,0.4])
st.image("./Design/logo.png", use_column_width=True)
new_line(1)

# Description
st.markdown("""Unleash your creativity with Art Styleify! Upload your image, choose an 
art style, and watch the magic happen. Embrace iconic artistic aesthetics by Van 
Gogh, Max Ernst, and more. Download and share your unique masterpieces effortlessly. 
Express your inner artist today! ✨




""", unsafe_allow_html=True)
new_line(1)

st.subheader("🎨 Examples")
st.write("The following are examples of the styles you can choose from. Below them, you can upload your own image and choose the style you want.")          
new_line(1)

# define path
path = "./assets/"

data_df = pd.DataFrame(
    {
        "Van Gogh": imgs_name_per_artist("Van Gogh"),
        "Van Gogh Images": [
            "https://uploads3.wikiart.org/images/vincent-van-gogh/avenue-in-voyer-d-argenson-park-at-asnieres-1887(1).jpg!Large.jpg",
            "https://uploads0.wikiart.org/00213/images/vincent-van-gogh/antique-3840759.jpg!Large.jpg",
            "https://uploads4.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!Large.jpg",
            "https://uploads5.wikiart.org/images/vincent-van-gogh/wheatfield-with-crows-1890.jpg!Large.jpg"
        ],

        "Josef Sima": imgs_name_per_artist("Josef Sima"),
        "Josef Sima Images": [
            "https://uploads4.wikiart.org/images/josef-ma/les-ombres-1960.jpg!Large.jpg",
            "https://uploads5.wikiart.org/images/josef-ma/mer-1960.jpg",
            "https://uploads7.wikiart.org/images/josef-ma/temp-te-f-camp-1954.jpg" ,
            "https://uploads7.wikiart.org/images/josef-ma/vejce-1927.jpg!PinterestSmall.jpg"
        ],
        
        "Man Ray": imgs_name_per_artist("Man Ray"),
        "Man Ray Images" : [
            "https://uploads2.wikiart.org/images/man-ray/hills.jpg!Large.jpg",
            "https://uploads7.wikiart.org/images/man-ray/ridgefield-landscape.jpg",
            "https://uploads6.wikiart.org/images/man-ray/silhouette.jpg",
            "https://uploads6.wikiart.org/images/man-ray/untitled-1.jpg"
        ],
        
        "Max Ernst": imgs_name_per_artist("Max Ernst"),
        "Max Ernst Images": [
            "https://uploads1.wikiart.org/images/max-ernst/forest-and-dove-1927.jpg!Large.jpg",
            "https://uploads4.wikiart.org/images/max-ernst/landscape-with-sun-1909.jpg!Large.jpg",
            "https://uploads8.wikiart.org/images/max-ernst/self-portrait-1909.jpg!Large.jpg",
            "https://uploads1.wikiart.org/images/max-ernst/towers-1916.jpg!Large.jpg"
        ],

        "Wassily Kandinsky": imgs_name_per_artist("Wassily Kandinsky"),
        "Wassily Kandinsky Images": [
            "https://uploads8.wikiart.org/images/wassily-kandinsky/composition-iv-1911.jpg!Large.jpg",
            "https://uploads8.wikiart.org/images/wassily-kandinsky/composition-vi-1913.jpg!Large.jpg",
            "https://uploads0.wikiart.org/images/wassily-kandinsky/composition-vii-1913.jpg!Large.jpg",
            "https://uploads7.wikiart.org/images/wassily-kandinsky/composition-x-1939.jpg!Large.jpg"
        ],

    }
)

st.data_editor(
    data_df,
    column_config = {
        "Van Gogh Images" : st.column_config.ImageColumn(
            "Van Gogh Images", help = "Van Gogh Images"
        ),

        "Josef Sima Images" : st.column_config.ImageColumn(
            "Josef Sima Images", help = "Josef Sima Images"
        ),

        "Man Ray Images" : st.column_config.ImageColumn(
           "Man Ray Images", help = "Man Ray Images"
        ),

        "Max Ernst Images" : st.column_config.ImageColumn(
            "Max Ernst Images", help = "Max Ernst Images"
        ),

        "Wassily Kandinsky Images" : st.column_config.ImageColumn(
            "Wassily Kandinsky Images", help = "Wassily Kandinsky Images"
        ),

    }

)

# Image Selection
new_line(4)
col1, col2 = st.columns(2, gap = 'large')


imgs_path_content = [img_path.split(".")[0] for img_path in os.listdir(path)]
imgs_path_style = [img_path.split(".")[0] for img_path in os.listdir(path)]
st.session_state.imgs_path_content = imgs_path_content
st.session_state.imgs_path_style = imgs_path_style

imgs_path_content = st.session_state.imgs_path_content
imgs_path_style = st.session_state.imgs_path_style

# Style Image
with col1:
    st.markdown("<h5 align='center'> Select Style Image </h3>", unsafe_allow_html=True)
    style_img = st.selectbox("Select the Style Image",  imgs_path_style + ["Upload Your Image"] )

    if style_img != "Upload Your Image" :
        st.image(path + style_img  + ".jpg" , use_column_width=True)
        style_img_path = path + style_img + ".jpg"
        st.session_state.style_img_name = style_img
    
    elif style_img == "Upload Your Image":
        uploaded_file = st.file_uploader("Choose an image...", type= ["jpg", "png", "jpeg", "jfif"], key="Style")
        if uploaded_file :

            # Read image as pixels
            img = Image.open(uploaded_file)
            img = img.resize((img.size[0]*2, img.size[1]*2))
            img.save(uploaded_file.name, format="JPEG")

            st.image(uploaded_file, use_column_width=True)
            style_img_path = uploaded_file.name
            st.session_state.style_uploaded_file = uploaded_file
            st.session_state.style_img_name = uploaded_file.name

# Content Image
with col2:
    st.markdown("<h5 align='center'> Select Content Image </h3>", unsafe_allow_html=True)
    content_img = st.selectbox("Select the Content Image", imgs_path_content + ["Upload Your Image"] )

    if content_img != "Upload Your Image" :
        st.image(path + content_img + ".jpg", use_column_width=True)
        content_img_path = path + content_img + ".jpg"
        st.session_state.content_img_name = content_img

    elif content_img == "Upload Your Image":
        uploaded_file = st.file_uploader("Choose an image...", type= ["jpg", "png", "jpeg", "jfif"], key="Content")
        if uploaded_file :

            # Read image as pixels
            img = Image.open(uploaded_file)
            img = img.resize((img.size[0]*2, img.size[1]*2))
            img.save(uploaded_file.name, format="JPEG")
            
            st.image(uploaded_file, use_column_width=True)
            content_img_path = uploaded_file.name
            st.session_state.content_uploaded_file = uploaded_file
            st.session_state.content_img_name = uploaded_file.name


    
new_line(3)
col1, col2, col3 = st.columns([1,1,1], gap='large')
if col2.button(" Make the Art 󠀽󠀽🖌️", use_container_width=True, ):

    # Prgoress Bar
    progress_bar = col2.progress(0)
    for percent_complete in range(100):
        time.sleep(0.001)
        progress_bar.progress(percent_complete + 1)

    content_img, style_img = load_images(content_img_path, style_img_path)

    stylized_image = model(tf.image.convert_image_dtype(content_img, tf.float32), 
                            tf.image.convert_image_dtype(style_img, tf.float32))[0]
    

    new_line(2)
    cola, colb, colc = st.columns([0.5,1.5,0.5], gap='small')

    colb.markdown("<h3 align='center'> Stylized Image </h3>", unsafe_allow_html=True)
    colb.image(tensor_to_image(stylized_image), use_column_width=True, caption="Stylized Image")
    new_line(2)

    
    buf = BytesIO()
    img = tensor_to_image(stylized_image)
    if img.size[0] < 500 or img.size[1] < 500:        
        img = img.resize((img.size[0]*2, img.size[1]*2))
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

    colx, coly, colz = st.columns([0.5,0.5,0.5], gap='large')
    with coly:
        btn = st.download_button(
        label="Download Image",
        data=byte_im,
        file_name=f"{st.session_state.content_img_name} Styled By {st.session_state.style_img_name}.png",
        mime="image/jpeg",
        use_container_width=True
        )

