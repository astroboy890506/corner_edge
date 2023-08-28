import cv2
import streamlit as st
import numpy as np

def main():
    st.set_page_config(
        page_title="Image Processing App",
        layout="wide"
    )

    st.title("Image Processing")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_array = uploaded_file.read()
        img = cv2.imdecode(np.frombuffer(img_array, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        st.sidebar.subheader("Image Processing Options")
        process_option = st.sidebar.selectbox("Select an option", ["Detect Edges (Canny)", "Detect Corners (Harris)"])

        if process_option == "Detect Edges (Canny)":
            mask_min = st.sidebar.slider("Canny Mask Minimum Value", min_value=0, max_value=255, value=100)
            mask_max = st.sidebar.slider("Canny Mask Maximum Value", min_value=0, max_value=255, value=200)
            edgesCanny = cv2.Canny(img, mask_min, mask_max)
            st.image(edgesCanny, caption="Edges (Canny)", use_column_width=True)

        elif process_option == "Detect Corners (Harris)":
            corner_quality = st.sidebar.slider("Corner Quality", min_value=0.01, max_value=0.5, value=0.04, step=0.01)
            dst = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, corner_quality)
            dst = cv2.dilate(dst, None)
            img[dst > 0.01 * dst.max()] = [0, 0, 255]
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Corners (Harris)", use_column_width=True)

if __name__ == "__main__":
    main()
