import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform
import pandas as pd


plt = platform.system()
temp = pathlib.PosixPath
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Transportni klassifikatsiya qiluvchi model')


# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    # model
    dls = ImageDataLoaders.from_folder('transport_model.pkl', valid_pct=0.2, item_tfms=Resize(460), batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)])
    # model = learn.export('transport_model.pkl')
    # model = data.load_learner('transport_model.pkl')
    # data = load_learner('transport_model.pkl')

    # prediction
    pred, pred_id, probs=model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
