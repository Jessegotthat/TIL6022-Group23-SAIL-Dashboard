import streamlit as st

st.title("Welcome to Our Streamlit App")

st.write("Here's our group website embedded below:")

st.components.v1.html(
    '<iframe src="https://mygroupwebsite.com" width="100%" height="800"></iframe>',
    height=800
)
