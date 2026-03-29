"""
===========================================================
TÊN DỰ ÁN  : Air Quality Forecasting – Demo App
MÔ TẢ      : Ứng dụng Streamlit trực quan hóa dữ liệu lịch sử
             chất lượng không khí (24 giờ gần nhất) và dự báo
             PM2.5 bằng mô hình XGBoost tại các mốc t+1h, t+3h,
             t+6h, t+12h, t+24h.
PHIÊN BẢN  : Đã tích hợp XGBoost block7 (North / South cluster).
=============================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 1: IMPORT THƯ VIỆN
# Nhiệm vụ : Nhập tất cả các thư viện cần thiết trước khi sử dụng.
#   - streamlit  : framework xây dựng giao diện web tương tác
#   - pandas     : đọc và xử lý dữ liệu dạng bảng (CSV)
#   - numpy      : tính toán số học và xử lý mảng
#   - plotly     : vẽ biểu đồ tương tác (line chart, subplots, scatter...)
#   - os         : xây dựng đường dẫn file độc lập với hệ điều hành
#   - datetime   : tính toán thời gian tương lai cho các mốc dự báo
#   - warnings   : tắt cảnh báo không cần thiết ra console
#   - inference  : module nội bộ – build features + chạy XGBoost + inverse-transform
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
from inference import predict_all_horizons, CLUSTER_NORTH, CLUSTER_SOUTH


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 2: CẤU HÌNH TRANG (PAGE CONFIG)
# Nhiệm vụ : Thiết lập các thuộc tính cơ bản của trang Streamlit.
# Thực hiện:
#   - page_title           : tên hiển thị trên tab trình duyệt
#   - page_icon            : icon emoji trên tab trình duyệt
#   - layout="wide"        : mở rộng nội dung chiếm toàn chiều rộng màn hình
#   - initial_sidebar_state: thanh sidebar mở sẵn khi tải trang
# Lưu ý: st.set_page_config() PHẢI được gọi đầu tiên, trước mọi lệnh st khác.
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 3: CSS TÙY CHỈNH GIAO DIỆN
# Nhiệm vụ : Ghi đè giao diện mặc định của Streamlit bằng CSS tùy chỉnh
#            để tạo phong cách dark mode với hiệu ứng glassmorphism.
# Thực hiện: Dùng st.markdown() với unsafe_allow_html=True để nhúng thẻ <style>.
# Các thành phần CSS được định nghĩa:
#   - nền toàn trang  : gradient tím đậm → xanh đậm
#   - sidebar         : nền tối trong suốt + viền mờ bên phải
#   - .main-title     : tiêu đề lớn với gradient màu chữ
#   - .aqi-badge      : nhãn tròn hiển thị mức AQI (tốt / trung bình / nguy hiểm...)
#   - .metric-card    : thẻ chỉ số KPI, có hiệu ứng nổi lên khi hover
#   - .section-header : tiêu đề mỗi khu vực (lịch sử, dự báo, chú thích)
#   - .forecast-card  : thẻ hiển thị giá trị dự báo từng mốc thời gian
#   - .info-pill      : viên nhỏ hiển thị thông tin thêm (địa chỉ trạm, tọa độ)
#   - Ghi đè Streamlit: đổi màu nhãn, selectbox, button cho khớp theme tối
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Nền toàn trang: gradient tím → xanh đậm */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
/* Sidebar: nền tối trong suốt, viền mờ bên phải */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.92);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Tiêu đề chính: chữ gradient màu sắc */
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
/* Phụ đề nhỏ bên dưới tiêu đề */
.sub-title {
    color: rgba(203,213,225,0.7);
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Nhãn tròn hiển thị mức chất lượng không khí AQI */
.aqi-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Thẻ chỉ số KPI (PM2.5, Nhiệt độ, Độ ẩm...) */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);         /* hiệu ứng kính mờ */
    transition: transform 0.2s ease, border-color 0.2s ease;
}
/* Hiệu ứng nổi lên khi hover vào thẻ KPI */
.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(167,139,250,0.4);
}
/* Nhãn mô tả bên trên giá trị */
.metric-label {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.9);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
/* Giá trị số lớn chính giữa thẻ */
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
}
/* Đơn vị đo hiển thị bên dưới giá trị */
.metric-unit {
    font-size: 0.85rem;
    color: rgba(148,163,184,0.7);
    margin-top: 2px;
}

/* Tiêu đề phần nội dung (lịch sử, dự báo...) */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Thẻ dự báo từng mốc thời gian */
.forecast-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 155px;
}
/* Hiệu ứng hover thẻ dự báo: viền xanh, nền nhạt */
.forecast-card:hover {
    border-color: rgba(96,165,250,0.5);
    background: rgba(96,165,250,0.07);
    transform: translateY(-2px);
}
/* Nhãn mốc thời gian (t+1h, t+6h...) */
.forecast-hour {
    font-size: 0.78rem;
    color: rgba(148,163,184,0.8);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
/* Giá trị PM2.5 dự báo cỡ lớn */
.forecast-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f1f5f9;
}
/* Đơn vị μg/m³ bên dưới giá trị dự báo */
.forecast-unit {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.6);
}
/* Nhãn mức AQI (Tốt / Trung bình / Không tốt...) */
.forecast-label {
    font-size: 0.7rem;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    
    /* Ép tất cả nhãn đều chiếm đúng không gian của 2 dòng (2 x 1.35em = 2.7em) */
    line-height: 1.35;
    height: 2.7em;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Viên nhỏ hiển thị thông tin trạm (địa danh, tọa độ) */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: rgba(203,213,225,0.85);
    margin-right: 8px;
    margin-bottom: 6px;
}

/* Ghi đè màu chữ tiêu đề mặc định của Streamlit → trắng sáng */
h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
}
/* Ghi đè style nhãn của selectbox / sidebar → nhỏ, viết hoa, màu nhạt */
label, .stSelectbox label, [data-testid="stSidebarContent"] label {
    color: rgba(203,213,225,0.85) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
/* Ghi đè nền và viền của ô selectbox → trong suốt, viền mờ */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
/* Cho phép danh sách dropdown (menu popover) mở rộng chiều ngang 
   để thấy đầy đủ tên dài của trạm thay vì bị cắt bởi width của sidebar */
div[data-baseweb="popover"] > div {
    width: max-content !important;
    min-width: 100% !important;
    max-width: 600px !important;
}
ul[data-baseweb="menu"] {
    width: max-content !important;
    min-width: 100% !important;
}
ul[data-baseweb="menu"] li {
    white-space: nowrap !important;       /* không tự động xuống dòng */
    overflow: visible !important;
    text-overflow: unset !important;      /* bỏ dấu ba chấm ... */
    padding-right: 20px !important;       /* thêm khoảng trống bên phải */
}
/* Ghi đè màu slider → tím */
[data-testid="stSlider"] {
    color: #a78bfa !important;
}
/* Ghi đè style nút bấm → gradient tím-xanh */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s ease !important;
}
/* Hiệu ứng mờ nhẹ khi hover nút */
.stButton > button:hover {
    opacity: 0.88 !important;
}
/* Đường kẻ ngang phân vùng → màu trắng mờ */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
/* Chỉ bỏ cursor thay đổi trên dropdown selectbox – giữ nguyên hover highlight mặc định */
[data-baseweb="select"] [role="option"],
[data-baseweb="menu"] [role="option"],
[data-baseweb="popover"] li,
[data-baseweb="select"] li,
[data-baseweb="select"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="popover"] li:hover,
[data-baseweb="select"] li:hover {
    cursor: default !important;
}
/* Station info card phía dưới */
.station-info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 4px;
}
.station-info-card .sta-name {
    font-size: 1rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.3;
    margin-bottom: 2px;
}
.station-info-card .sta-province {
    font-size: 0.72rem;
    color: rgba(148,163,184,0.7);
    margin-bottom: 10px;
}
.station-info-card .sta-row {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.72rem;
    color: rgba(148,163,184,0.65);
    margin-bottom: 4px;
}
.station-info-card .sta-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 4: HẰNG SỐ VÀ HÀM TIỆN ÍCH
# Nhiệm vụ : Khai báo các giá trị cố định và các hàm dùng chung trong app.
# ─────────────────────────────────────────────────────────────────────────────

# Đường dẫn thư mục chứa file CSV đã làm sạch (data/clean/)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "clean")

# Đường dẫn file thông tin các trạm đo (data/info.csv)
INFO_CSV = os.path.join(os.path.dirname(__file__), "data", "info.csv")

# Phân chia cluster theo vùng địa lý (định nghĩa từ pipeline.py training)
# Cluster North: các trạm phía Bắc (Hà Nội và vùng lân cận)
CLUSTER_NORTH_APP = [1, 4, 5, 16, 17, 27]   # trạm có model xgboost_north_*
# Cluster South: các trạm phía Nam (TP.HCM và vùng lân cận)
CLUSTER_SOUTH_APP = [7, 18, 24, 30, 31, 32]  # trạm có model xgboost_south_*

# Các trạm có XGBoost model (dùng để chạy dự báo thật)
XGB_STATION_IDS = CLUSTER_NORTH_APP + CLUSTER_SOUTH_APP   # [1,4,5,16,17,27,7,18,24,30,31,32]

# Tất cả trạm có file clean CSV trong data/clean/ (bao gồm cả trạm không có XGBoost model)
# Các trạm không có model sẽ hiển thị nhãn "Không có dữ liệu dự báo"
AVAILABLE_STATION_IDS = [1,4,5,16,17,27,7,18,24,30,31,32]

# Bảng ngưỡng AQI theo PM2.5 (μg/m³) của Việt Nam / WHO
# Mỗi hàng: (giá trị thấp, giá trị cao, màu HEX, nhãn tiếng Việt, nhãn tiếng Anh)
AQI_BREAKPOINTS = [
    (0,    12,    "#34d399", "Tốt",                           "Good"),
    (12.1,   35.4,  "#a3e635", "Trung bình",                    "Moderate"),
    (35.5, 55.4,  "#fbbf24", "Không tốt cho nhóm nhạy cảm",   "Unhealthy for Sensitive"),
    (55.5, 150.4, "#f97316", "Không tốt",                     "Unhealthy"),
    (150.5,250.4, "#ef4444", "Rất không tốt",                 "Very Unhealthy"),
    (250.5,500,   "#9b59b6", "Nguy hiểm",                     "Hazardous"),
]


def get_aqi_info(pm25_val):
    """
    Nhiệm vụ : Tra cứu màu sắc và nhãn AQI tương ứng với giá trị PM2.5.
    Tham số  : pm25_val – giá trị PM2.5 (μg/m³)
    Trả về   : (màu_HEX: str, nhãn_tiếng_Việt: str)
    Cách làm : Lặp qua bảng AQI_BREAKPOINTS, so sánh pm25_val với từng khoảng.
               Nếu vượt 500 μg/m³ thì trả về màu "Nguy hiểm" mặc định.
    """
    for lo, hi, color, lv, _ in AQI_BREAKPOINTS:
        if lo <= pm25_val <= hi:
            return color, lv
    return "#9b59b6", "Nguy hiểm"


# Ánh xạ tên hiển thị → tên cột trong CSV cho 6 chỉ số ô nhiễm chính
POLLUTANTS = {
    "PM2.5 (μg/m³)": "pm25",
    "PM10 (μg/m³)":  "pm10",
    "CO (μg/m³)":    "co",
    "O₃ (μg/m³)":   "o3",
    "NO₂ (μg/m³)":  "no2",
    "SO₂ (μg/m³)":  "so2",
}


@st.cache_data(ttl=300)
def load_station_info():
    """
    Nhiệm vụ : Đọc file info.csv chứa thông tin (tỉnh, quận, tọa độ) của các trạm.
    Thực hiện: Lọc chỉ giữ lại những trạm có dữ liệu thực tế (AVAILABLE_STATION_IDS).
    Cache    : Kết quả được cache 300 giây để tránh đọc lại file mỗi lần re-render.
    Trả về   : DataFrame gồm các cột [station, province, district, latitude, longitude].
    """
    df = pd.read_csv(INFO_CSV)
    df = df[df["station"].isin(AVAILABLE_STATION_IDS)].copy()
    return df


@st.cache_data(ttl=300)
def load_station_data(station_id: int):
    """
    Nhiệm vụ : Đọc file CSV dữ liệu sạch của một trạm đo cụ thể.
    Tham số  : station_id – ID số của trạm (ví dụ: 1, 7, 13...)
    Thực hiện: Đọc file clean_station_{id}.csv, parse cột timestamp thành datetime,
               sắp xếp theo thời gian tăng dần.
    Cache    : Cache 300 giây để tránh đọc lại file lớn (~10MB) mỗi lần.
    Trả về   : DataFrame đã sắp xếp theo timestamp.
    """
    path = os.path.join(DATA_DIR, f"clean_station_{station_id}.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def get_last_n_hours(df, n=24):
    """
    Nhiệm vụ : Lấy n hàng cuối cùng trong DataFrame (tương ứng n giờ gần nhất).
    Tham số  : df – DataFrame đã sắp xếp; n – số giờ cần lấy (mặc định 24).
    Trả về   : DataFrame mới (bản sao) chứa n hàng cuối.
    """
    return df.tail(n).copy()


# generate_fake_forecast đã được xóa.
# Thay thế bằng inference.predict_all_horizons() – gọi XGBoost model thật.


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 5: THANH SIDEBAR (BẢN ĐIỀU KHIỂN TRÁI)
# Nhiệm vụ : Hiển thị các bộ điều khiển để người dùng lựa chọn trạm đo
#            và chỉ số ô nhiễm muốn xem trong biểu đồ lịch sử.
# Thực hiện:
#   1. Logo + tên ứng dụng ở đầu sidebar.
#   2. Dropdown chọn trạm đo (hiển thị tên quận, tỉnh).
#   3. Thông tin chi tiết trạm được chọn (tọa độ GPS).
#   4. Dropdown chọn chỉ số ô nhiễm hiển thị trên biểu đồ lịch sử.
#   5. Mô tả ngắn về dự án.
# ─────────────────────────────────────────────────────────────────────────────
station_info = load_station_info()  # Tải thông tin các trạm từ info.csv

with st.sidebar:
    # ── Logo ─────────────────────────────────────────────────────────
    st.markdown("""
        <div style='text-align:center;padding:16px 0 20px 0;'>
            <div style='font-size:2.2rem;margin-bottom:4px;'>🌫️</div>
            <div style='font-size:1.0rem;font-weight:800;color:#a78bfa;
                        letter-spacing:0.06em;'>AQF Demo</div>
            <div style='font-size:0.68rem;color:rgba(148,163,184,0.55);
                        margin-top:2px;'>Air Quality Forecasting</div>
        </div>
    """, unsafe_allow_html=True)

    # Xây dựng options: icon màu theo cluster + tên quận (hiển tỉnh qua format_func)
    station_options = {}
    province_map    = {}
    for _, row in station_info.iterrows():
        sid  = int(row["station"])
        dot  = "🔵" if sid in CLUSTER_NORTH_APP else ("🟠" if sid in CLUSTER_SOUTH_APP else "⚪")
        lbl  = f"{dot} Trạm {sid:02d}  · {row['district']}"
        station_options[lbl] = sid
        province_map[lbl]    = row["province"]


    # ── Selectbox chọn trạm ──────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:rgba(148,163,184,0.6);"
        "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;'>📡 Trạm đo</div>",
        unsafe_allow_html=True
    )
    selected_label = st.selectbox(
        "Chọn trạm đo",
        options=list(station_options.keys()),
        index=0,
        format_func=lambda lbl: f"{lbl}  —  {province_map.get(lbl, '')}",
        label_visibility="collapsed",
    )
    selected_station_id = station_options[selected_label]

    # Lấy dòng thông tin tương ứng với trạm đã chọn
    selected_row = station_info[station_info["station"] == selected_station_id].iloc[0]

    # ── Station info card ────────────────────────────────────────
    if selected_station_id in CLUSTER_NORTH_APP:
        badge_bg  = "rgba(96,165,250,0.18)";  badge_clr = "#60a5fa"
        badge_txt = "🔵 Cluster BẮC · XGBoost North"
    elif selected_station_id in CLUSTER_SOUTH_APP:
        badge_bg  = "rgba(251,146,60,0.18)";  badge_clr = "#fb923c"
        badge_txt = "🟠 Cluster NAM · XGBoost South"
    else:
        badge_bg  = "rgba(148,163,184,0.12)"; badge_clr = "#94a3b8"
        badge_txt = "⚪ Không có model XGBoost"

    st.markdown(f"""
    <div class="station-info-card">
        <div class="sta-name">{selected_row['district']}</div>
        <div class="sta-province">{selected_row['province']}</div>
        <div class="sta-row">🌐 {selected_row['latitude']:.4f}°N &nbsp;{selected_row['longitude']:.4f}°E</div>
        <div class="sta-row">📌 ID trạm: {selected_station_id}</div>
        <div class="sta-badge" style="background:{badge_bg};color:{badge_clr};border:1px solid {badge_clr}55;">
            {badge_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chọn chỉ số pollutant ────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem;font-weight:700;color:rgba(148,163,184,0.6);"
        "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;'>📊 Chỉ số lịch sử</div>",
        unsafe_allow_html=True
    )
    show_pollutant = st.selectbox(
        "Chỉ số hiển thị (lịch sử)",
        options=list(POLLUTANTS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    pollutant_col = POLLUTANTS[show_pollutant]


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6: TIÊU ĐỀ TRANG VÀ TẢI DỮ LIỆU
# Nhiệm vụ : Hiển thị tiêu đề chính, tải dữ liệu trạm được chọn, và trích xuất
#            24 giờ gần nhất cùng giá trị PM2.5 mới nhất để dùng xuyên suốt app.
# ─────────────────────────────────────────────────────────────────────────────

# Tiêu đề và phụ đề hiển thị ở đầu trang chính
st.markdown("<div class='main-title'>🌫️ Air Quality Forecasting</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-title'>Hệ thống giám sát và dự báo chất lượng không khí · "
    f"<b>{selected_row['district']}, {selected_row['province']}</b></div>",
    unsafe_allow_html=True
)

# Tải toàn bộ dữ liệu của trạm, hiển thị spinner trong lúc chờ
with st.spinner("Đang tải dữ liệu…"):
    df = load_station_data(selected_station_id)

# Lấy 24 hàng cuối (= 24 giờ gần nhất, dữ liệu tần suất 1h)
hist_df = get_last_n_hours(df, 24)

# Dòng dữ liệu mới nhất (giờ hiện tại) và giá trị PM2.5 kèm màu AQI
latest      = hist_df.iloc[-1]
latest_pm25 = float(latest["pm25"])
aqi_color, aqi_label = get_aqi_info(latest_pm25)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 7: CÁC THẺ KPI (KEY PERFORMANCE INDICATORS)
# Nhiệm vụ : Hiển thị nhanh 5 chỉ số quan trọng nhất tại thời điểm hiện tại.
# Thực hiện:
#   - Chia trang thành 5 cột ngang bằng nhau (st.columns).
#   - Hàm kpi_card() nhận cột, nhãn, giá trị, đơn vị và icon; render HTML card.
#   - Bên dưới cards: nhãn AQI tròn hiển thị mức chất lượng + thời điểm cập nhật.
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)


def kpi_card(col, label, value, unit, icon=""):
    """
    Nhiệm vụ : Render một thẻ KPI vào cột Streamlit được chỉ định.
    Tham số  :
        col   – đối tượng cột Streamlit (st.columns result)
        label – tên chỉ số (ví dụ: "PM2.5")
        value – giá trị hiển thị dạng chuỗi (ví dụ: "45.2")
        unit  – đơn vị đo (ví dụ: "μg/m³")
        icon  – emoji icon hiển thị cạnh nhãn
    """
    col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{icon} {label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-unit'>{unit}</div>
        </div>
    """, unsafe_allow_html=True)


# Hiển thị 5 thẻ KPI với dữ liệu thực từ hàng mới nhất
kpi_card(col1, "PM2.5",       f"{latest_pm25:.1f}",          "μg/m³", "🌫")
kpi_card(col2, "PM10",        f"{latest['pm10']:.1f}",       "μg/m³", "💨")
kpi_card(col3, "Nhiệt độ",    f"{latest['temp']:.1f}",       "°C",    "🌡")
kpi_card(col4, "Độ ẩm",       f"{latest['rh']:.0f}",         "%",     "💧")
kpi_card(col5, "Tốc độ gió",  f"{latest['wind_spd']:.1f}",  "m/s",   "🌬")

# Nhãn AQI tròn màu động + thời gian cập nhật cuối
st.markdown(f"""
    <div style='margin-top:16px;margin-bottom:8px;'>
        <span style='color:rgba(148,163,184,0.7);font-size:0.83rem;'>Trạng thái chất lượng không khí hiện tại: </span>
        <span class='aqi-badge' style='background:{aqi_color}22;color:{aqi_color};border:1px solid {aqi_color}55;'>
            ● &nbsp;{aqi_label}
        </span>
        <span style='color:rgba(148,163,184,0.55);font-size:0.78rem;margin-left:10px;'>
            Cập nhật lúc {latest['timestamp'].strftime('%H:%M %d/%m/%Y')}
        </span>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 8: BIỂU ĐỒ LỊCH SỬ 24 GIỜ (BIỂU ĐỒ CHÍNH)
# Nhiệm vụ : Vẽ biểu đồ đường kép cho thấy diễn biến chỉ số ô nhiễm và
#            nhiệt độ trong 24 giờ gần nhất.
# Thực hiện:
#   - Dùng make_subplots() tạo 2 hàng dùng chung trục X (timestamp):
#       + Hàng 1 (65% chiều cao): chỉ số ô nhiễm người dùng chọn từ sidebar
#       + Hàng 2 (35% chiều cao): nhiệt độ (°C) – yếu tố ảnh hưởng AQI
#   - Nếu chỉ số là PM2.5: tự động vẽ thêm 3 đường ngưỡng AQI nằm ngang.
#   - Biểu đồ trong suốt (nền = rgba(0,0,0,0)) để khớp dark theme.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Lịch sử chất lượng không khí (24 giờ gần nhất)</div>", unsafe_allow_html=True)

# Tạo canvas 2 subplot chia sẻ trục X (thời gian)
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.65, 0.35],   # Hàng 1 chiếm 65%, hàng 2 chiếm 35%
    vertical_spacing=0.06,
    subplot_titles=("", "")
)

# Chuẩn bị mảng trục X (timestamp) và Y (giá trị chỉ số đã chọn)
pm_col = pollutant_col
y_vals = hist_df[pm_col].values
x_vals = hist_df["timestamp"]

# Trace 1 (hàng trên): đường chỉ số ô nhiễm dạng spline + tô vùng bên dưới
fig.add_trace(
    go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines",
        line=dict(color="#a78bfa", width=2.5, shape="spline"),
        fill="tozeroy",                          # tô từ đường xuống trục 0
        fillcolor="rgba(167,139,250,0.12)",      # tím mờ
        name=show_pollutant,
        hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>" + show_pollutant + ": %{y:.1f}<extra></extra>",
    ),
    row=1, col=1
)

# Nếu chỉ số là PM2.5: thêm 3 đường ngưỡng tham chiếu AQI nằm ngang
if pollutant_col == "pm25":
    for thr, lbl, clr in [(12, "Tốt", "#34d399"), (35.4, "Trung bình", "#fbbf24"), (55.4, "Không tốt", "#f97316")]:
        fig.add_hline(y=thr, line_dash="dot", line_color=clr, line_width=1, opacity=0.5, row=1, col=1)
        # Ghi nhãn mức ngưỡng ở cạnh phải biểu đồ
        fig.add_annotation(x=x_vals.iloc[-1], y=thr, text=f" {lbl}", showarrow=False,
                           font=dict(color=clr, size=10), xanchor="left", row=1, col=1)

# Trace 2 (hàng dưới): nhiệt độ dạng spline + tô vùng xanh dương mờ
fig.add_trace(
    go.Scatter(
        x=x_vals, y=hist_df["temp"],
        mode="lines",
        line=dict(color="#60a5fa", width=2, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.10)",
        name="Nhiệt độ (°C)",
        hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>Nhiệt độ: %{y:.1f}°C<extra></extra>",
    ),
    row=2, col=1
)

# Layout chung: nền trong suốt, chú thích nằm ngang phía trên, tooltip chung
fig.update_layout(
    height=420,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",   # tooltip hiện cả 2 trace khi hover cùng trục X
)
# Định dạng chung cho cả 2 trục X và Y: lưới mờ, font nhỏ
fig.update_xaxes(
    showgrid=True, gridcolor="rgba(255,255,255,0.05)",
    tickfont=dict(size=11), linecolor="rgba(255,255,255,0.1)",
)
fig.update_yaxes(
    showgrid=True, gridcolor="rgba(255,255,255,0.05)",
    tickfont=dict(size=11), linecolor="rgba(255,255,255,0.1)",
    zeroline=False,
)
# Gán nhãn trục Y riêng cho từng hàng subplot
fig.update_yaxes(title_text=show_pollutant, title_font=dict(size=11), row=1, col=1)
fig.update_yaxes(title_text="Nhiệt độ (°C)", title_font=dict(size=11), row=2, col=1)

# Render biểu đồ ra trang (ẩn thanh công cụ plotly)
st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 9: SPARKLINE 6 CHỈ SỐ Ô NHIỄM
# Nhiệm vụ : Hiển thị 6 biểu đồ nhỏ (sparkline) cho thấy xu hướng 24h của
#            từng chỉ số ô nhiễm: PM2.5, PM10, CO, O₃, NO₂, SO₂.
# Thực hiện:
#   - Chia thành 3 cột, mỗi hàng 2 sparkline (6 chỉ số / 3 cột = 2 hàng).
#   - Mỗi ô hiển thị: tên chỉ số, giá trị hiện tại, mũi tên delta (tăng/giảm).
#   - Biểu đồ nhỏ cao 130px, ẩn trục X, hiện trục Y nhỏ.
#   - Màu delta: cam = tăng (xấu hơn), xanh lá = giảm (tốt hơn), xám = không đổi.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🔬 Các chỉ số ô nhiễm (24h)</div>", unsafe_allow_html=True)

poll_cols  = st.columns(3)                 # 3 cột ngang bằng
poll_items = list(POLLUTANTS.items())      # danh sách (tên hiển thị, tên cột) của 6 chỉ số

# Màu đường và màu vùng tô cho từng chỉ số (theo thứ tự trong POLLUTANTS)
LINE_COLORS = ["#a78bfa", "#60a5fa", "#34d399", "#fbbf24", "#f97316", "#ec4899"]
FILL_COLORS = [
    "rgba(167,139,250,0.15)", "rgba(96,165,250,0.15)", "rgba(52,211,153,0.15)",
    "rgba(251,191,36,0.15)",  "rgba(249,115,22,0.15)", "rgba(236,72,153,0.15)",
]

for i, (label, col_name) in enumerate(poll_items):
    with poll_cols[i % 3]:   # phân phối vào 3 cột theo vị trí
        vals = hist_df[col_name].values   # mảng 24 giá trị của chỉ số này

        # Tạo sparkline nhỏ
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=list(range(len(vals))),   # trục X chỉ là chỉ số 0→23 (ẩn đi)
            y=vals,
            mode="lines",
            line=dict(color=LINE_COLORS[i], width=2, shape="spline"),
            fill="tozeroy",
            fillcolor=FILL_COLORS[i],
        ))
        fig_s.update_layout(
            height=130,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),   # ẩn trục X (không cần thấy nhãn thời gian)
            yaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=9), linecolor="rgba(0,0,0,0)"),
        )

        # Tính delta giữa giờ mới nhất và giờ trước đó
        current_val = vals[-1]
        delta_val   = float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0
        delta_icon  = "▲" if delta_val > 0 else ("▼" if delta_val < 0 else "—")
        delta_color = "#f97316" if delta_val > 0 else ("#34d399" if delta_val < 0 else "#94a3b8")

        # Hiển thị tên chỉ số, giá trị hiện tại và mũi tên delta
        st.markdown(f"""
            <div style='margin-bottom:4px;'>
                <span style='font-size:0.72rem;color:rgba(148,163,184,0.8);text-transform:uppercase;letter-spacing:0.06em;'>{label}</span><br>
                <span style='font-size:1.3rem;font-weight:800;color:#f1f5f9;'>{current_val:.1f}</span>
                <span style='font-size:0.78rem;color:{delta_color};margin-left:6px;'>{delta_icon} {abs(delta_val):.1f}</span>
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_s, width='stretch', config=dict(displayModeBar=False))


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 10: KHU VỰC DỰ BÁO PM2.5 (XGBoost block7)
# Nhiệm vụ : Cho phép người dùng chọn mốc thời gian dự báo (t+1h đến t+24h)
#            và hiển thị giá trị PM2.5 dự báo thật từ mô hình XGBoost.
# Thực hiện:
#   A. Gọi predict_all_horizons() từ inference.py:
#        - Xác định cluster (north/south) theo station_id
#        - Build 134 feature từ dữ liệu clean, predict bằng XGBoost
#        - Inverse-transform về μg/m³ bằng scaler đã lưu
#   B. Radio button để chọn mốc dự báo: t+1h / t+3h / t+6h / t+12h / t+24h.
#   C. 5 thẻ card màu AQI động – thẻ được chọn sẽ nổi bật hơn.
#   D. Biểu đồ nối tiếp: lịch sử 24h (tím) → dự báo 5 điểm (xanh nét đứt).
#       + Đường dọc phân chia "Hiện tại" vs "Tương lai".
#       + Các đường ngưỡng AQI nằm ngang.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>🔮 Dự báo PM2.5 · XGBoost</div>", unsafe_allow_html=True)

# Danh sách mốc thời gian dự báo (giờ) – khớp với các model đã train
horizons = [1, 3, 6, 12, 24]

# Kiểm tra xem trạm đang chọn có model XGBoost không
has_model = selected_station_id in XGB_STATION_IDS

if has_model:
    # Gọi XGBoost inference: tự động chọn north/south model theo station_id
    # Toàn bộ logic nằm trong inference.py (build_inference_features + predict + inverse)
    with st.spinner("🤖 Đang chạy mô hình XGBoost dự báo…"):
        forecasts = predict_all_horizons(selected_station_id, df, tuple(horizons))
    region_label = "North" if selected_station_id in CLUSTER_NORTH_APP else "South"
    st.markdown(f"""
        <div style='font-size:0.8rem;color:rgba(148,163,184,0.75);margin-bottom:12px;'>
        🤖 <b>Mô hình:</b> XGBoost block7 · Cluster <b>{region_label}</b> ·
        Dự báo dựa trên {len(df)} điểm dữ liệu lịch sử
        </div>
    """, unsafe_allow_html=True)
else:
    # Trạm không có model → fallback hiển thị giá trị hiện tại cho tất cả mốc
    forecasts = {h: latest_pm25 for h in horizons}
    st.markdown("""
        <div style='font-size:0.82rem;color:rgba(251,191,36,0.8);margin-bottom:16px;'>
        ⚠️ <i>Trạm này không có mô hình XGBoost. Hiển thị giá trị hiện tại làm tham chiếu.</i>
        </div>
    """, unsafe_allow_html=True)

# Ánh xạ mốc giờ → (nhãn ngắn, loại dự báo) – thêm t+3h
label_map = {
    1:  ("Sau 1 giờ",  "Ngắn hạn"),
    3:  ("Sau 3 giờ",  "Ngắn hạn"),
    6:  ("Sau 6 giờ",  "Trung ngắn"),
    12: ("Sau 12 giờ", "Trung hạn"),
    24: ("Sau 24 giờ", "Dài hạn"),
}

# A. Radio button chọn mốc dự báo – mốc được chọn sẽ làm nổi bật thẻ tương ứng
selected_horizon = st.radio(
    "Chọn mốc thời gian dự báo",
    options=horizons,
    format_func=lambda h: label_map[h][0],
    horizontal=True,
    label_visibility="visible",
)

# B. 5 thẻ dự báo (thêm t+3h) – thẻ khớp với selected_horizon có viền màu AQI dày hơn
fc_cols = st.columns(5)
for i, h in enumerate(horizons):
    val              = forecasts[h]
    fc_color, fc_label = get_aqi_info(val)
    tag, range_label = label_map[h]
    is_selected      = h == selected_horizon

    # Thẻ được chọn: viền 2px màu AQI + nền nhạt theo màu AQI
    # Thẻ không chọn: viền 1px trắng mờ + nền xám trong suốt
    border_style = f"2px solid {fc_color}" if is_selected else "1px solid rgba(255,255,255,0.09)"
    bg_style = (
        f"rgba({int(fc_color[1:3],16)},{int(fc_color[3:5],16)},{int(fc_color[5:7],16)},0.08)"
        if is_selected else "rgba(255,255,255,0.04)"
    )

    fc_cols[i].markdown(f"""
        <div class='forecast-card' style='border:{border_style};background:{bg_style};'>
            <div class='forecast-hour'>{tag}</div>
            <div class='forecast-value' style='color:{fc_color};'>{val}</div>
            <div class='forecast-unit'>μg/m³</div>
            <div class='forecast-label' style='color:{fc_color};'>{fc_label}</div>
            <div style='font-size:0.68rem;color:rgba(148,163,184,0.5);margin-top:4px;'>{range_label}</div>
        </div>
    """, unsafe_allow_html=True)

# Khoảng trống nhỏ trước biểu đồ dự báo
st.markdown("<br>", unsafe_allow_html=True)

# C. Chuẩn bị dữ liệu cho biểu đồ lịch sử + dự báo kết hợp
all_times = list(hist_df["timestamp"])   # 24 mốc thời gian lịch sử
all_pm25  = list(hist_df["pm25"])        # 24 giá trị PM2.5 lịch sử

# Tính thời điểm tương lai: cộng thêm h giờ vào mốc mới nhất
future_base  = hist_df["timestamp"].iloc[-1]
future_times = [future_base + timedelta(hours=h) for h in horizons]
future_vals  = [forecasts[h] for h in horizons]

fig_fc = go.Figure()

# Trace lịch sử 24h: đường tím đặc + tô vùng
fig_fc.add_trace(go.Scatter(
    x=all_times, y=all_pm25,
    mode="lines",
    line=dict(color="#a78bfa", width=2.5, shape="spline"),
    fill="tozeroy",
    fillcolor="rgba(167,139,250,0.10)",
    name="Lịch sử (24h)",
    hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>PM2.5: %{y:.1f} μg/m³<extra></extra>",
))

# Trace dự báo: đường xanh nét đứt + điểm tròn màu AQI
fig_fc.add_trace(go.Scatter(
    x=future_times, y=future_vals,
    mode="lines+markers",
    line=dict(color="#60a5fa", width=2.5, dash="dot", shape="spline"),
    marker=dict(
        size=10,
        color=[get_aqi_info(v)[0] for v in future_vals],   # màu điểm theo AQI
        line=dict(color="white", width=2)                  # viền trắng điểm
    ),
    name="Dự báo",
    hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>PM2.5 (dự báo): %{y:.1f} μg/m³<extra></extra>",
))

# Đường dọc phân chia "Hiện tại" và "Tương lai"
fig_fc.add_vline(
    x=future_base,
    line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1.5,
)
# Nhãn "Hiện tại" ở trên đường dọc phân chia
fig_fc.add_annotation(
    x=future_base,
    y=max(all_pm25 + future_vals) * 1.05,
    text="Hiện tại", showarrow=False,
    font=dict(color="rgba(255,255,255,0.5)", size=11),
    bgcolor="rgba(0,0,0,0)",
)

# Các đường ngưỡng AQI nằm ngang tham chiếu cho dễ đọc
for thr, lbl, clr in [(12, "Tốt", "#34d399"), (35.4, "TB", "#fbbf24"), (55.4, "Không tốt", "#f97316")]:
    fig_fc.add_hline(y=thr, line_dash="dot", line_color=clr, line_width=1, opacity=0.4)

# Layout biểu đồ dự báo: cùng phong cách dark mode
fig_fc.update_layout(
    height=320,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter, sans-serif"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        font=dict(size=11), bgcolor="rgba(0,0,0,0)",
    ),
    margin=dict(l=0, r=0, t=28, b=0),
    hovermode="x unified",
    yaxis_title="PM2.5 (μg/m³)",
)
fig_fc.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11))
fig_fc.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(size=11), zeroline=False)

st.plotly_chart(fig_fc, width='stretch', config=dict(displayModeBar=False))


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 11: BẢNG CHÚ THÍCH MÀU AQI
# Nhiệm vụ : Hiển thị bảng đối chiếu 6 mức chất lượng không khí theo tiêu chuẩn
#            AQI PM2.5, giúp người dùng hiểu ý nghĩa các màu trên biểu đồ.
# Thực hiện:
#   - Chia thành 6 cột (khớp với số mức trong AQI_BREAKPOINTS).
#   - Mỗi ô: chấm tròn màu + tên mức + khoảng giá trị (μg/m³).
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-header'>🟢 Bảng chỉ số AQI PM2.5</div>", unsafe_allow_html=True)

legend_cols = st.columns(len(AQI_BREAKPOINTS))   # 6 cột = 6 mức AQI

for i, (lo, hi, color, lv, lv_en) in enumerate(AQI_BREAKPOINTS):
    legend_cols[i].markdown(f"""
        <div style='
            background:{color}18;
            border:1px solid {color}44;
            border-radius:12px;
            padding:12px 8px;
            text-align:center;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            min-height:90px;
            box-sizing:border-box;
        '>
            <!-- Chấm tròn màu đại diện cho mức AQI -->
            <div style='
                width:12px;height:12px;
                border-radius:50%;
                background:{color};
                flex-shrink:0;
                margin-bottom:6px;
            '></div>
            <!-- Tên mức chất lượng – word-break để ngắt dòng đều, không bị cao lệch -->
            <div style='
                font-size:0.65rem;
                font-weight:700;
                color:{color};
                text-transform:uppercase;
                letter-spacing:0.05em;
                line-height:1.3;
                word-break:break-word;
            '>{lv}</div>
            <!-- Khoảng giá trị PM2.5 tương ứng -->
            <div style='
                font-size:0.68rem;
                color:rgba(148,163,184,0.7);
                margin-top:4px;
                white-space:nowrap;
            '>{lo}–{hi} μg/m³</div>
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 12: CHÂN TRANG (FOOTER)
# Nhiệm vụ : Hiển thị thông tin dự án và ghi chú về trạng thái demo ở cuối trang.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center;padding:20px 0 8px 0;
                font-size:0.72rem;color:rgba(148,163,184,0.4);'>
        Air Quality Forecasting Demo &nbsp;·&nbsp; Đồ án 2 - HK7 &nbsp;·&nbsp;
        Dữ liệu từ mạng lưới quan trắc quốc gia &nbsp;·&nbsp;
    </div>
""", unsafe_allow_html=True)
