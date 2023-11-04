
# Model uchun ma'lumotlarni yuklash
data = load_learner('transport_model.pkl', cpu=False)

# Rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)

    # prediction
    pred, pred_id, probs = data.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id] * 100:.1f}%')

    # plotting
    fig = px.bar(x=probs * 100, y=data.dls.vocab)
    st.plotly_chart(fig)

























# import streamlit as st
# from fastai.vision.all import *
# import pathlib
# import plotly.express as px
# import platform
# import pandas as pd


# plt = platform.system()
# temp = pathlib.PosixPath
# if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# # title
# st.title('Transportni klassifikatsiya qiluvchi model')


# # rasmni joylash
# file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg'])
# if file:
#     st.image(file)
#     # PIL convert
#     img = PILImage.create(file)
#     # model
#     model = data.load_learner('transport_model.pkl')
#     # data = load_learner('transport_model.pkl')

#     # prediction
#     pred, pred_id, probs=model.predict(img)
#     st.success(f"Bashorat: {pred}")
#     st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

#     # plotting
#     fig=px.bar(x=probs*100, y=model.dls.vocab)
#     st.plotly_chart(fig)
