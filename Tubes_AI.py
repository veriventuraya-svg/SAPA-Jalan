import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile 

# 0. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="SAPA-Jalan AI",
    page_icon="🛣️",
    layout="wide" 
)

# --- KUSTOMISASI CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    .title-box {
        background-color: #1a1c24;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #2b3040;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #4DA8DA;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8c9baf;
        margin-top: 0;
    }
    </style>
""", unsafe_allow_html=True)

# 1. MUAT MODEL
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 2. FUNGSI GAMBAR KOTAK
def draw_custom_boxes(frame, results):
    tinggi_layar, lebar_layar, _ = frame.shape
    luas_layar = tinggi_layar * lebar_layar
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            nama_asli = r.names.get(cls_id, str(cls_id))
            
            luas_kotak = (x2 - x1) * (y2 - y1)
            persentase_ukuran = (luas_kotak / luas_layar) * 100
            
            if "pothole" in nama_asli.lower() or "lubang" in nama_asli.lower():
                color = (0, 0, 255) # Merah
                status = "AWAS!"
            else:
                if persentase_ukuran >= 15.0:
                    color = (0, 255, 255) # Kuning
                    status = "Hati-hati"
                else:
                    color = (0, 255, 0)   # Hijau
                    status = "Aman"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            text = f"{nama_asli} {(conf*100):.1f}% - {status}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return frame

# 3. FUNGSI CALLBACK WEBRTC (KAMERA LANGSUNG)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_resized = cv2.resize(img, (640, 480))
    results = model(img_resized, imgsz=640, conf=0.50, verbose=False)
    img_annotated = draw_custom_boxes(img_resized, results)
    return av.VideoFrame.from_ndarray(img_annotated, format="bgr24")


# ==========================================
# 4. TAMPILAN ANTARMUKA
# ==========================================
st.markdown("""
    <div class="title-box">
        <h1 class="main-title">🛣️ SAPA-Jalan AI</h1>
        <p class="sub-title">Sistem Analisis Deteksi Kerusakan Jalan Raya Berbasis YOLOv8</p>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("logo unsika.png", width=100)
    st.header("⚙️ Konfigurasi Sistem")
    pilihan_mode = st.radio("Pilih Sumber Video:", ("🔴 Live Camera (Webcam)", "📂 Upload Video Lokal"))
    
    st.markdown("---")
    st.subheader("📖 Indikator Sistem")
    st.error("**🟥 Lubang (AWAS!)**\n\nBahaya tinggi, berpotensi merusak velg/ban.")
    st.warning("**🟨 Retak (Hati-hati)**\n\nRetak memanjang >15%, kurangi kecepatan.")
    st.success("**🟩 Retak (Aman)**\n\nRetak struktural minor.")
    
    st.markdown("---")
    st.caption("Dikembangkan oleh:")
    st.write("**Kelompok 8 Kecerdasan Buatan**")
    st.write("Teknik Elektro Unsika")

# --- AREA UTAMA ---
col_kiri, col_tengah, col_kanan = st.columns([1, 8, 1])

with col_tengah:
    if pilihan_mode == "🔴 Live Camera (Webcam)":
        st.info("💡 **Petunjuk:** Berikan izin akses kamera pada browser, lalu klik tombol **START**.")
        
        rtc_conf = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        webrtc_streamer(
            key="sapa-jalan-live-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_conf,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif pilihan_mode == "📂 Upload Video Lokal":
        st.info("💡 **Petunjuk:** Unggah video rekaman jalan raya dari memori perangkat.")
        
        file_video = st.file_uploader("", type=["mp4", "avi", "mov"])
        
        if file_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(file_video.read())
            
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            tombol_stop = st.button("⏹️ Hentikan Analisis Video", use_container_width=True)
            
            while cap.isOpened() and not tombol_stop:
                ret, frame = cap.read()
                
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                    
                frame_resized = cv2.resize(frame, (640, 480))
                results = model(frame_resized, imgsz=640, conf=0.15, verbose=False)
                frame_annotated = draw_custom_boxes(frame_resized, results)
                
                frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                
            cap.release()
