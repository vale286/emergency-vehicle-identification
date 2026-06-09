import streamlit as st
from PIL import Image
from traffic_light import traffic_light_status

st.title("Deteksi Kendaraan Darurat dan Lampu Lalu Lintas")

uploaded_file = st.file_uploader("Unggah gambar kendaraan", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Simpan gambar sementara
    image_path = f"temp_image.jpg"
    image.save(image_path)

    # Deteksi kendaraan
    status = traffic_light_status(image_path)
    st.write(f"**Status Lampu Lalu Lintas:** {status}")

    # Tampilkan status lampu
    if status == "Green":
        st.markdown(
            "<div style='width:100px;height:100px;background-color:green;border-radius:50%;margin:auto;'></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='width:100px;height:100px;background-color:red;border-radius:50%;margin:auto;'></div>",
            unsafe_allow_html=True,
        )