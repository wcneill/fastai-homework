import os
import PIL.Image
from fastai.vision import *
import streamlit as st

st.title('Squash It!!')

defaults.device = torch.device('cpu')

path = Path('data/squash')
inferencer = load_learner(path)

classes = [
    'acorn', 'butternut', 'kabocha', 'patty', 'pumpkin',
    'red kuri', 'spaghetti', 'sweet dumpling', 'tromboncino', 'zucchini'
]

saved_fn = "predict_this.jpg"

img_bytes = st.file_uploader("Squash It!!", type=['png', 'jpg', 'jpeg'])
if img_bytes is not None:

    st.write("Image Uploaded Successfully:")

    # Get Image from Bytes, create thumbnail and save
    PILImage = PIL.Image.open(img_bytes)
    PILImage.thumbnail((500, 500))
    PILImage.save(saved_fn)

    # Convert PIL image to a FastAI Image (FastAI cannot do inference on PIL images)
    fastai_img = open_image(saved_fn)

    # Display the image
    st.image(PILImage, caption="What is this gourd!?")

    pred_class, pred_idx, outputs = inferencer.predict(open_image(saved_fn))
    st.write(f"This gourd is a {pred_class} with probability {outputs[pred_idx].item():.4f}")

    for type, prob in zip(classes, outputs):
        st.write(f"Probability of {type}: {prob:.7f}")

    os.remove(saved_fn)


