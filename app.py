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
#            để tạo phong cách sáng, chuyên nghiệp, dễ đọc.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

/* Nền toàn trang: trắng sạch */
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
}
/* Sidebar: nền xanh rất nhạt, viền phải */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Tiêu đề chính */
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Nhãn tròn AQI */
.aqi-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Thẻ chỉ số KPI */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Tiêu đề phần */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e293b;
    margin-top: 1.5rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Thẻ dự báo */
.forecast-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: all 0.2s ease;
    cursor: pointer;
}
.forecast-card:hover {
    border-color: #3b82f6;
    box-shadow: 0 4px 12px rgba(59,130,246,0.12);
    transform: translateY(-2px);
}
.forecast-hour {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
}
.forecast-value {
    font-size: 1.7rem;
    font-weight: 800;
    color: #1e293b;
}
.forecast-unit {
    font-size: 0.75rem;
    color: #94a3b8;
}
.forecast-label {
    font-size: 0.7rem;
    margin-top: 6px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Viên nhỏ thông tin trạm */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #475569;
    margin-right: 8px;
    margin-bottom: 6px;
}

/* Ghi đè Streamlit defaults cho light theme */
h1, h2, h3, h4, h5, h6 {
    color: #1e293b !important;
}
label, .stSelectbox label, [data-testid="stSidebarContent"] label {
    color: #475569 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
}
[data-testid="stSlider"] {
    color: #3b82f6 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}
hr {
    border-color: #e2e8f0 !important;
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
    (12,   35.4,  "#a3e635", "Trung bình",                    "Moderate"),
    (35.4, 55.4,  "#fbbf24", "Không tốt cho nhóm nhạy cảm",  "Unhealthy for Sensitive"),
    (55.4, 150.4, "#f97316", "Không tốt",                     "Unhealthy"),
    (150.4,250.4, "#ef4444", "Rất không tốt",                 "Very Unhealthy"),
    (250.4,500,   "#9b59b6", "Nguy hiểm",                     "Hazardous"),
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
    # Logo và tên app ở đầu sidebar
    st.markdown("""
        <div style='text-align:center;padding: 12px 0 18px 0;'>
            <span style='font-size:2.5rem;'>🌫️</span><br>
            <span style='font-size:1.05rem;font-weight:800;color:#2563eb;letter-spacing:0.04em;'>AQF Demo</span><br>
            <span style='font-size:0.72rem;color:#64748b;'>Air Quality Forecasting</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Xây dựng dict ánh xạ: "🔵 [BẮC] Trạm 1 – Cau Giay, Ha Noi" → 1
    # Thêm icon + nhãn cluster (BẮC/NAM) để người dùng biết trạm thuộc cluster nào
    # và mô hình XGBoost nào sẽ được dùng để dự báo
    station_options = {}
    for _, row in station_info.iterrows():
        sid = int(row["station"])
        if sid in CLUSTER_NORTH_APP:
            cluster_icon = "🔵 [BẮC]"
        elif sid in CLUSTER_SOUTH_APP:
            cluster_icon = "🟠 [NAM]"
        else:
            cluster_icon = "⚪ [–]"  # trạm không có XGBoost model
        label = f"{cluster_icon} Trạm {sid} – {row['district']}, {row['province']}"
        station_options[label] = sid

    # Dropdown chọn trạm đo – kết quả lưu vào selected_station_id
    selected_label = st.selectbox(
        "Chọn trạm đo",
        options=list(station_options.keys()),
        index=0,
    )
    selected_station_id = station_options[selected_label]

    # Lấy dòng thông tin tương ứng với trạm đã chọn
    selected_row = station_info[station_info["station"] == selected_station_id].iloc[0]

    st.markdown("---")

    # Hiển thị thông tin chi tiết của trạm: quận, tỉnh, tọa độ và cluster dự báo
    st.markdown("<div class='metric-label'>Thông tin trạm</div>", unsafe_allow_html=True)
    # Xác định cluster và trạng thái model
    if selected_station_id in CLUSTER_NORTH_APP:
        cluster_badge = "<span style='color:#2563eb;font-weight:700;'>🔵 Cluster BẮC · XGBoost North</span>"
    elif selected_station_id in CLUSTER_SOUTH_APP:
        cluster_badge = "<span style='color:#ea580c;font-weight:700;'>🟠 Cluster NAM · XGBoost South</span>"
    else:
        cluster_badge = "<span style='color:#64748b;'>⚪ Không có model XGBoost</span>"
    st.markdown(f"""
        <span class='info-pill'>📍 {selected_row['district']}</span>
        <span class='info-pill'>🏙 {selected_row['province']}</span><br>
        <span class='info-pill'>🌐 {selected_row['latitude']:.4f}°N, {selected_row['longitude']:.4f}°E</span><br>
        <div style='margin-top:8px;font-size:0.75rem;'>{cluster_badge}</div>
    """, unsafe_allow_html=True)

    # Mặc định hiển thị PM2.5 trên biểu đồ lịch sử
    show_pollutant = "PM2.5 (μg/m³)"
    pollutant_col = "pm25"


# ─────────────────────────────────────────────────────────────────────────────
# PHẦN 6: TIÊU ĐỀ TRANG VÀ TẢI DỮ LIỆU
# Nhiệm vụ : Hiển thị tiêu đề chính, tải dữ liệu trạm được chọn, và trích xuất
#            24 giờ gần nhất cùng giá trị PM2.5 mới nhất để dùng xuyên suốt app.
# ─────────────────────────────────────────────────────────────────────────────

# Tiêu đề và phụ đề hiển thị ở đầu trang chính
st.markdown("<div class='main-title'>🌫️ Dự báo chất lượng không khí</div>", unsafe_allow_html=True)
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
        <span style='color:#64748b;font-size:0.83rem;'>Trạng thái chất lượng không khí hiện tại: </span>
        <span class='aqi-badge' style='background:{aqi_color}18;color:{aqi_color};border:1px solid {aqi_color}55;'>
            ● &nbsp;{aqi_label}
        </span>
        <span style='color:#94a3b8;font-size:0.78rem;margin-left:10px;'>
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
        line=dict(color="#6366f1", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.08)",
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
        line=dict(color="#3b82f6", width=2, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
        name="Nhiệt độ (°C)",
        hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>Nhiệt độ: %{y:.1f}°C<extra></extra>",
    ),
    row=2, col=1
)

# Layout chung: nền trong suốt, chú thích nằm ngang phía trên, tooltip chung
fig.update_layout(
    height=420,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#475569", family="Inter, sans-serif"),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(size=11, color="#475569"),
        bgcolor="rgba(255,255,255,0)",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
)
fig.update_xaxes(
    showgrid=True, gridcolor="#f1f5f9",
    tickfont=dict(size=11, color="#64748b"), linecolor="#e2e8f0",
)
fig.update_yaxes(
    showgrid=True, gridcolor="#f1f5f9",
    tickfont=dict(size=11, color="#64748b"), linecolor="#e2e8f0",
    zeroline=False,
)
fig.update_yaxes(title_text=show_pollutant, title_font=dict(size=11, color="#475569"), row=1, col=1)
fig.update_yaxes(title_text="Nhiệt độ (°C)", title_font=dict(size=11, color="#475569"), row=2, col=1)

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
LINE_COLORS = ["#6366f1", "#3b82f6", "#10b981", "#f59e0b", "#f97316", "#ec4899"]
FILL_COLORS = [
    "rgba(99,102,241,0.08)", "rgba(59,130,246,0.08)", "rgba(16,185,129,0.08)",
    "rgba(245,158,11,0.08)",  "rgba(249,115,22,0.08)", "rgba(236,72,153,0.08)",
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
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(size=9, color="#94a3b8"), linecolor="rgba(0,0,0,0)"),
        )

        # Tính delta giữa giờ mới nhất và giờ trước đó
        current_val = vals[-1]
        delta_val   = float(vals[-1] - vals[-2]) if len(vals) >= 2 else 0
        delta_icon  = "▲" if delta_val > 0 else ("▼" if delta_val < 0 else "—")
        delta_color = "#f97316" if delta_val > 0 else ("#34d399" if delta_val < 0 else "#94a3b8")

        # Hiển thị tên chỉ số, giá trị hiện tại và mũi tên delta
        st.markdown(f"""
            <div style='margin-bottom:4px;'>
                <span style='font-size:0.72rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;'>{label}</span><br>
                <span style='font-size:1.3rem;font-weight:800;color:#1e293b;'>{current_val:.1f}</span>
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
        <div style='font-size:0.8rem;color:#64748b;margin-bottom:12px;'>
        🤖 <b>Mô hình:</b> XGBoost block7 · Cluster <b>{region_label}</b> ·
        Dự báo dựa trên {len(df)} điểm dữ liệu lịch sử
        </div>
    """, unsafe_allow_html=True)
else:
    # Trạm không có model → fallback hiển thị giá trị hiện tại cho tất cả mốc
    forecasts = {h: latest_pm25 for h in horizons}
    st.markdown("""
        <div style='font-size:0.82rem;color:#d97706;margin-bottom:16px;'>
        ⚠️ <i>Trạm này không có mô hình XGBoost. Hiển thị giá trị hiện tại làm tham chiếu.</i>
        </div>
    """, unsafe_allow_html=True)

# Ánh xạ mốc giờ → (nhãn ngắn, loại dự báo) – thêm t+3h
label_map = {
    1:  ("t+1h",  "Ngắn hạn"),
    3:  ("t+3h",  "Ngắn hạn"),
    6:  ("t+6h",  "Trung ngắn"),
    12: ("t+12h", "Trung hạn"),
    24: ("t+24h", "Dài hạn"),
}

# 5 thẻ dự báo hiển thị đồng thời tất cả mốc thời gian
fc_cols = st.columns(5)
for i, h in enumerate(horizons):
    val              = forecasts[h]
    fc_color, fc_label = get_aqi_info(val)
    tag, range_label = label_map[h]

    border_style = f"1px solid {fc_color}55"
    bg_style = f"rgba({int(fc_color[1:3],16)},{int(fc_color[3:5],16)},{int(fc_color[5:7],16)},0.04)"


    fc_cols[i].markdown(f"""
        <div class='forecast-card' style='border:{border_style};background:{bg_style};'>
            <div class='forecast-hour'>{tag}</div>
            <div class='forecast-value' style='color:{fc_color};'>{val}</div>
            <div class='forecast-unit'>μg/m³</div>
            <div class='forecast-label' style='color:{fc_color};'>{fc_label}</div>
            <div style='font-size:0.68rem;color:#94a3b8;margin-top:4px;'>{range_label}</div>
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
    line=dict(color="#6366f1", width=2.5, shape="spline"),
    fill="tozeroy",
    fillcolor="rgba(99,102,241,0.06)",
    name="Lịch sử (24h)",
    hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>PM2.5: %{y:.1f} μg/m³<extra></extra>",
))

# Trace dự báo: đường xanh nét đứt + điểm tròn màu AQI
fig_fc.add_trace(go.Scatter(
    x=future_times, y=future_vals,
    mode="lines+markers",
    line=dict(color="#3b82f6", width=2.5, dash="dot", shape="spline"),
    marker=dict(
        size=10,
        color=[get_aqi_info(v)[0] for v in future_vals],
        line=dict(color="white", width=2)
    ),
    name="Dự báo",
    hovertemplate="<b>%{x|%H:%M %d/%m}</b><br>PM2.5 (dự báo): %{y:.1f} μg/m³<extra></extra>",
))

# Đường dọc phân chia "Hiện tại" và "Tương lai"
fig_fc.add_vline(
    x=future_base,
    line_dash="dash", line_color="#94a3b8", line_width=1.5,
)
fig_fc.add_annotation(
    x=future_base,
    y=max(all_pm25 + future_vals) * 1.05,
    text="Hiện tại", showarrow=False,
    font=dict(color="#64748b", size=11),
    bgcolor="rgba(255,255,255,0)",
)

# Các đường ngưỡng AQI nằm ngang tham chiếu cho dễ đọc
for thr, lbl, clr in [(12, "Tốt", "#34d399"), (35.4, "TB", "#fbbf24"), (55.4, "Không tốt", "#f97316")]:
    fig_fc.add_hline(y=thr, line_dash="dot", line_color=clr, line_width=1, opacity=0.4)

# Layout biểu đồ dự báo: cùng phong cách dark mode
fig_fc.update_layout(
    height=320,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#475569", family="Inter, sans-serif"),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
        font=dict(size=11, color="#475569"), bgcolor="rgba(255,255,255,0)",
    ),
    margin=dict(l=0, r=0, t=28, b=0),
    hovermode="x unified",
    yaxis_title="PM2.5 (μg/m³)",
)
fig_fc.update_xaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11, color="#64748b"))
fig_fc.update_yaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11, color="#64748b"), zeroline=False)

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
                color:#94a3b8;
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
                font-size:0.72rem;color:#94a3b8;'>
        Air Quality Forecasting Demo &nbsp;·&nbsp; Đồ án 2 - HK7 &nbsp;·&nbsp;
        Dữ liệu từ mạng lưới quan trắc quốc gia &nbsp;·&nbsp;
    </div>
""", unsafe_allow_html=True)
