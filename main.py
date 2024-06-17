import streamlit as st
from PIL import Image, ImageDraw
from pix2pix import Pix2Pix, generate_sequence_sharpened
import torch

st.markdown("<h1 style='text-align: center; color: #7091E6;'>System wizualizacji terapii naczyniowej Dermogen</h1>", unsafe_allow_html=True)

st.text("\n\n")

st.text("W celu skorzystania z systemu wgraj zdjęcie twarzy.\n"
        "Następnie wybierz interesujący Cię fragment.")

pix2pix = Pix2Pix(3, 3)

uploaded_file = st.file_uploader("Wgraj Pix2Pix", type=["ckpt", "pt", "pth"])

if uploaded_file is not None:
    try:
        with st.spinner("Ładowanie modelu..."):
            pix2pix.load_state_dict(torch.load(uploaded_file, map_location=torch.device('cpu')))
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.warning('Bez dostarczonego modelu otrzymasz losowy szum', icon="⚠️")


def generate_images(pic):
    results = generate_sequence_sharpened(pix2pix, pic)
    results.insert(0, pic)
    return results


initial_image_path = 'data/example_pic_best.png'

try:
    initial_image = Image.open(initial_image_path)
except FileNotFoundError:
    initial_image = None

uploaded_image = st.file_uploader('Wgraj zdjęcie', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
else:
    image = initial_image

if image is not None:
    image_width, image_height = image.size

    st.sidebar.header('Wybierz kadr')

    size = 224

    max_x = max(0, image_width - size)
    max_y = max(0, image_height - size)

    x = st.sidebar.slider('Przesuń w prawo', 0, max_x, 0)
    y = st.sidebar.slider('Przesuń w dół', 0, max_y, 0)

    preview_image = image.copy()
    draw = ImageDraw.Draw(preview_image)
    draw.rectangle([x, y, x + size, y + size], outline="blue", width=3)

    st.image(preview_image, caption='Obraz z wybranym oknem', use_column_width=True)

    if st.sidebar.button('Wybierz'):
        left = x
        top = y
        right = x + size
        bottom = y + size

        cropped_image = image.crop((left, top, right, bottom))

        generated_images = generate_images(cropped_image)

        modified_images = []

        for generated in generated_images:
            modified_image = image.copy()
            modified_image.paste(generated, (left, top))
            draw = ImageDraw.Draw(modified_image)
            draw.rectangle([left, top, right, bottom], outline="blue", width=2)
            modified_images.append(modified_image)

        st.session_state.modified_images = modified_images

    if 'modified_images' in st.session_state:
        generation_idx = st.slider('Wybierz kolejne efekty', 0, 4, 0)
        st.image(st.session_state.modified_images[generation_idx],
                 caption=f'Efekty po {generation_idx}. zabiegu', use_column_width=True)
else:
    st.write("Brak dostępnych obrazów. Wgraj proszę nowe zdjęcie.")

