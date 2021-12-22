import cv2
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image, ImageEnhance
import cv2 as cv
from super_resolution import sr


@st.cache
def load_image(img):
    im = Image.open(img)
    return im


face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
smile = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')


@st.cache
def detect_faces(image):
    img = np.array(image.convert('RGB'))
    img = cv.cvtColor(img, 1)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # face detection
    faces = face.detectMultiScale(gray_img, 1.1, 4)
    # Draw rectangle
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces


@st.cache
def detect_eyes(image):
    img = np.array(image.convert('RGB'))
    img = cv.cvtColor(img, 1)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # face detection
    eyes_ = eyes.detectMultiScale(gray_img, 1.3, 5)
    # Draw rectangle
    for (x, y, w, h) in eyes_:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img, eyes_


@st.cache
def detect_smiles(image):
    img = np.array(image.convert('RGB'))
    img = cv.cvtColor(img, 1)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # face detection
    smiles = smile.detectMultiScale(gray_img, 1.1, 4)
    # Draw rectangle
    for (x, y, w, h) in smiles:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img, smiles


@st.cache
def gray_scale(image):
    img = np.array(image.convert('RGB'))
    new_image = cv.cvtColor(img, 1)
    new_image = cv.cvtColor(new_image, cv.COLOR_BGR2GRAY)
    return new_image


@st.cache
def contrast(image, c_rate):
    enhancer = ImageEnhance.Contrast(image)
    new_image = enhancer.enhance(c_rate)
    return new_image


@st.cache
def brightness(image, bright_rate):
    enhancer = ImageEnhance.Brightness(image)
    new_image = enhancer.enhance(bright_rate)
    return new_image


@st.cache
def blur(image, blur_rate):
    img = np.array(image.convert('RGB'))
    new_image = cv.cvtColor(img, 1)
    new_image = cv.GaussianBlur(new_image, (11, 11), blur_rate)
    return new_image


def main():
    """ Image Enhancement Application"""

    st.title('Image Enhancement Application')

    st.info('Enhancing Images and Feature Detection')
    activities = ['Enhancement', 'Detection', 'About']
    enhance_types = ['Original', 'Gray-Scale', 'Contrast', 'Brightness', 'Blur', 'Super-Resolution']
    choice = st.sidebar.selectbox('Select Activity', activities)
    if choice != 'About':
        st.text('Check About section for more Information')

    original_image = None
    if choice == 'Enhancement':
        st.subheader('Image Enhancement')
        image_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            img = load_image(image_file)
            original_image = img
            st.text('Original Image')
            st.image(img, width=512)

        enhance_type = st.sidebar.radio('Enhance Type', enhance_types)

        if enhance_type == 'Original':
            st.write(original_image)
        if enhance_type == 'Super-Resolution':
            if image_file is not None:
                if st.button('enhance'):
                    orignal_image, enhanced_image = sr(image_file)

                    col1, col2 = st.columns(2)
                    col1.image([orignal_image], width=512, caption='Original (Real)')
                    col2.image([enhanced_image], width=512, caption='High Resolution Image (Generated)')

        elif enhance_type == 'Gray-Scale':
            st.image(gray_scale(original_image))
        elif enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast', 0.5, 10.0)
            new_image = contrast(original_image, c_rate)
            st.image(new_image)
        elif enhance_type == 'Brightness':
            b_rate = st.sidebar.slider('Brightness', 0.5, 10.0)
            new_image = brightness(original_image, b_rate)
            st.image(new_image)
        elif enhance_type == 'Blur':
            b_rate = st.sidebar.slider('Blur', 0.5, 10.0)
            new_image = blur(original_image, b_rate)
            st.image(new_image)

    elif choice == 'Detection':
        st.subheader('Face Features Detection')
        task = ['Faces', 'Smiles', 'Eyes']
        image_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            img = load_image(image_file)
            original_image = img
            st.text('Original Image')
            st.image(img)

        feature_choices = st.sidebar.selectbox('Find Features', task)
        if st.button('process'):
            if feature_choices == 'Faces':
                img, faces = detect_faces(original_image)
                st.image(img)
                st.success('Found: {} faces'.format(len(faces)))
            if feature_choices == 'Smiles':
                img, faces = detect_smiles(original_image)
                st.image(img)
                st.success('Found: {} faces'.format(len(faces)))
            if feature_choices == 'Eyes':
                img, faces = detect_eyes(original_image)
                st.image(img)
                st.success('Found: {} faces'.format(len(faces)))

    elif choice == 'About':

        components.html(
            """
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>

                    <div class="card border-secondary"> <div class="card-header" style="font-weight:bold;">Image 
                    Enhancement</div> <div> <span class="badge badge-warning">OpenCV</span> <span class="badge 
                    badge-warning">Pytorch</span> <span class="badge badge-warning">numpy</span> <span class="badge 
                    badge-warning">Pillow</span> <span class="badge badge-warning">Torchvision</span> </div> <div 
                    class="card-body text-secondary"> <h5 class="card-title">Image Enhancement using <i>PIL</i> and 
                    <i>Pytorch</i> </h5> <p class="card-text" style="color:black">Image Enhancement from <strong>Brightness</strong>, 
                    <strong>Contrast</strong>, <strong>Gray-Scale</strong> and <strong>Blur</strong> and also Change 
                    Image Resolution using <strong>Super-Resolution</strong> with <strong>Deep Learning</strong>. 
                    </p> </div> </div> 


            """,
            height=200,
        )

        components.html(
            """
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
            
                    <div class="card border-secondary"> <div class="card-header" style="font-weight:bold;">Feature 
                    Detection</div> <div> <span class="badge badge-warning">OpenCV</span> <span class="badge 
                    badge-warning">Pytorch</span> <span class="badge badge-warning">numpy</span> <span class="badge 
                    badge-warning">Pillow</span> <span class="badge badge-warning">Torchvision</span> </div> <div 
                    class="card-body text-secondary"> <h5 class="card-title">Feature Detection <i>using 
                    Opencv</i></h5> <p class="card-text" style="color:black">Detect Features such as 
                    <strong>Faces</strong>, <strong>Smiles</strong> and <strong>Eyes</strong> using Opencv and with 
                    the help of Opencv haarcascade .</p> </div> </div> 

             
            """,
            height=200,
        )

        components.html(
            """
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
        
            <div class="card text-white bg-dark"> <div class="card-header">Find Me</div> <div class="card-body">
            <a href="https://abdelhakmokri.pythonanywhere.com/" class="badge badge-light">Website</a><br> 
            <a href="https://github.com/mokri" class="badge badge-light">Github</a> <br> 
            <a href="https://www.instagram.com/abdelhakmokri/" class="badge badge-light">Instagram</a> <br> 
            <a href="https://www.facebook.com/ABDELHAKMOKRI/" class="badge badge-light">Facebook</a> <br> 
            <a href="https://twitter.com/abdelhakmokri" class="badge badge-light">Twitter</a> <br> 
            <a href="https://www.linkedin.com/in/abdelheq-mokri-b55425160/" class="badge badge-light">Linkedin</a>
            <hr> 
                <p>Copyright &copy; <script>document.write(new Date().getFullYear())</script> Abdelheq Mokri</p>
            """,
            height=300,
        )


if __name__ == '__main__':
    main()
