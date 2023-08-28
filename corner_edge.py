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
        process_option = st.sidebar.selectbox("Select an option", ["Detect Edges (Canny)", "Detect Edges (Sobel)",
                                                                   "Detect Edges (Prewitt)", "Detect Corners (Harris)"])

        if process_option.startswith("Detect Edges"):
            mask_size = st.sidebar.slider("Mask Size", min_value=3, max_value=7, value=3, step=2)

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

        elif process_option == "Detect Edges (Sobel)":
            edgesSobel = cv2.Sobel(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=mask_size)
            edgesSobel = cv2.convertScaleAbs(edgesSobel)  # Convert to absolute values
            edgesSobel = cv2.normalize(edgesSobel, None, 0, 1, cv2.NORM_MINMAX)  # Normalize to [0.0, 1.0] range

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(edgesSobel, caption="Edges (Sobel)", use_column_width=True)

        elif process_option == "Detect Edges (Prewitt)":
            kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            edgesPrewitt = cv2.filter2D(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), -1, kernel)

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(edgesPrewitt, caption="Edges (Prewitt)", use_column_width=True)

        elif process_option == "Detect Corners (Harris)":
            corner_quality = st.sidebar.slider("Corner Quality", min_value=0.01, max_value=0.5, value=0.04, step=0.01)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            dst = cv2.cornerHarris(gray, 2, 3, corner_quality)
            dst = cv2.dilate(dst, None)
            corners_img = img.copy()
            corners_img[dst > 0.01 * dst.max()] = [0, 0, 255]

            st.subheader("Image Comparison")
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

            with col2:
                st.image(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB), caption="Corners (Harris)", use_column_width=True)

if __name__ == "__main__":
    main()
