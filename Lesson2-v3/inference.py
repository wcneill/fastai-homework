import os
import PIL.Image
from fastai.vision import *
import torchvision.transforms as T
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

img_bytes = st.file_uploader("Upload a photo of your funky looking squash!", type=['png', 'jpg', 'jpeg'])
if img_bytes is not None:

    st.write("Image Uploaded Successfully:")

    # Get Image from Bytes, create thumbnail
    PILImage = PIL.Image.open(img_bytes)
    PILImage.thumbnail((500, 500))

    # Display the image
    st.image(PILImage, caption="What is this gourd!?")

    # Convert PIL image to a FastAI Image (FastAI cannot do inference on PIL images)
    img_tensor = T.ToTensor()(PILImage)
    img_fastai = Image(img_tensor)

    # Make inference on image, display results:
    pred_class, pred_idx, outputs = inferencer.predict(img_fastai)
    st.write(f"This gourd is a {pred_class} with probability {outputs[pred_idx].item():.4f}")

    for type, prob in zip(classes, outputs):
        st.write(f"Probability of {type}: {prob:.7f}")



