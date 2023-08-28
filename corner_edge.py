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

        st.sidebar.subheader("Image Processing Options")
        process_option = st.sidebar.selectbox("Select an option", ["Detect Edges (Canny)", "Detect Edges (Prewitt)",
                                                                   "Detect Edges (Sobel)", "Detect Corners"])

        if process_option == "Detect Edges (Canny)":
            mask_min = st.sidebar.slider("Canny Mask Minimum Value", min_value=0, max_value=255, value=100)
            mask_max = st.sidebar.slider("Canny Mask Maximum Value", min_value=0, max_value=255, value=200)
            edgesCanny = cv2.Canny(img, mask_min, mask_max)

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(edgesCanny, caption="Edges (Canny)", use_column_width=True)

        elif process_option == "Detect Edges (Prewitt)":
            edgesPrewitt = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(edgesPrewitt, caption="Edges (Prewitt)", use_column_width=True)

        elif process_option == "Detect Edges (Sobel)":
            edgesSobel = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(edgesSobel, caption="Edges (Sobel)", use_column_width=True)

        elif process_option == "Detect Corners":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            corners_img = img.copy()
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(corners_img, (x, y), 3, 255, -1)

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB), caption="Corners", use_column_width=True)

if __name__ == "__main__":
    main()
