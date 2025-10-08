import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import re # Thư viện để xử lý Regex cho việc định dạng văn bản AI

# Import types for configuration (cần thiết cho system_instruction trong chat)
from google.genai import types

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="TRỢ LÝ PHÂN TÍCH BÁO CÁO TÀI CHÍNH CỦA AGRIBANK",
    layout="wide"
)

# --- Hiển thị Logo và Tiêu đề Cập nhật ---
col_logo, col_title = st.columns([1, 6])

# Logo Agribank (Sử dụng đường dẫn hình ảnh có thể truy cập công khai)
# Lưu ý: URL bạn cung cấp không phải là URL ảnh trực tiếp, tôi sử dụng một placeholder có thể truy cập được hoặc bạn cần thay thế bằng URL trực tiếp của logo.
# Để minh họa, tôi sử dụng URL của Agribank từ trang chủ có thể truy cập công khai.
logo_url = "https://firebasestorage.googleapis.com/v0/b/chat-with-pdf-2.appspot.com/o/agribank_logo_placeholder.png?alt=media&token=48029d20-b467-4286-98ec-69339e08398e"

with col_logo:
    st.image(logo_url, width=100)

with col_title:
    st.markdown(
        "<h1 style='color: #8B0000; font-weight: 900;'>TRỢ LÝ PHÂN TÍCH BÁO CÁO TÀI CHÍNH CỦA AGRIBANK</h1>", 
        unsafe_allow_html=True
    )

# --- Dòng cảnh báo ---
st.markdown(
    """
    <p style='color: #8B0000; font-weight: bold; font-style: italic;'>
    ⚠️ AI có thể mắc sai sót. Luôn kiểm chứng lại thông tin.
    </p>
    """,
    unsafe_allow_html=True
)

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN XỬ LÝ CHIA CHO 0 *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN XỬ LÝ CHIA CHO 0 KẾT THÚC *******************************
    
    return df

# --- Hàm định dạng kết quả AI ---
def format_ai_result(text):
    """
    Chuyển văn bản nhận xét từ AI thành định dạng bullet point có icon.
    Giả định mỗi đoạn văn mới là một luận điểm chính.
    """
    paragraphs = text.strip().split('\n\n')
    
    formatted_output = ""
    icons = ["✅", "📈", "💰", "⚙️", "⚠️", "📊"] # Các icon phù hợp với phân tích tài chính
    
    for i, p in enumerate(paragraphs):
        # Loại bỏ các dấu gạch đầu dòng Markdown có thể có từ AI
        clean_p = re.sub(r'^\s*[-*]\s*', '', p.strip(), flags=re.MULTILINE)
        
        # Chọn icon xoay vòng
        icon = icons[i % len(icons)]
        
        # Tạo bullet point HTML/Markdown
        formatted_output += f"**{icon} {icon} {icon}** {clean_p}\n\n"
        
    return formatted_output

# --- Hàm gọi API Gemini cho phân tích Báo cáo Tài chính ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn, mỗi đoạn là một luận điểm chính) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return format_ai_result(response.text) # Gọi hàm định dạng kết quả

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Khởi tạo Client Gemini duy nhất bằng @st.cache_resource ---
@st.cache_resource
def get_gemini_client(api_key):
    """Khởi tạo và lưu trữ Gemini Client. Chỉ chạy MỘT LẦN."""
    if not api_key:
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        return None

# ******************************* PHẦN BỔ SUNG CHAT GEMINI BẮT ĐẦU *******************************

# --- Thiết lập Sidebar Chat ---
with st.sidebar:
    st.subheader("Trò chuyện với Gemini 💬")
    st.info("Sử dụng Gemini để hỏi thêm về các thuật ngữ tài chính hoặc kiến thức chung.")
    
    # Lấy API Key và khởi tạo Client
    api_key = st.secrets.get("GEMINI_API_KEY")
    client = get_gemini_client(api_key)
    
    # 1. Khởi tạo session state cho lịch sử chat và session
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    # 2. Khởi tạo Chat Session nếu Client đã sẵn sàng
    if client:
        # Tạo Chat Session nếu chưa có (Chỉ chạy 1 lần)
        if "chat_session" not in st.session_state:
            try:
                # Khởi tạo cấu hình cho System Instruction
                system_instruction = "Bạn là một chuyên gia tài chính và AI trợ giúp, hãy trả lời các câu hỏi một cách chính xác và chuyên nghiệp bằng Tiếng Việt. Tuyệt đối không tiết lộ vai trò AI của bạn, chỉ tập trung vào việc cung cấp thông tin tài chính hữu ích."
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
                
                # Tạo Chat Session
                st.session_state["chat_session"] = client.chats.create(
                    model='gemini-2.5-flash',
                    config=config 
                )
                
            except APIError as e:
                st.error(f"Lỗi tạo Chat Session: Kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
            except Exception as e:
                st.error(f"Lỗi không xác định khi tạo Chat Session: {e}")

        # 3. Logic Chat chính (Chỉ chạy khi chat_session đã được tạo)
        if "chat_session" in st.session_state:
            
            # Hiển thị lịch sử chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input cho người dùng
            if prompt := st.chat_input("Hỏi Gemini một câu hỏi...", key="sidebar_chat_input"):
                
                # Thêm tin nhắn người dùng vào lịch sử và hiển thị
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Gửi câu hỏi đến Gemini và hiển thị phản hồi
                with st.chat_message("assistant"):
                    with st.spinner("Đang gửi và chờ câu trả lời..."):
                        try:
                            # Sử dụng st.session_state["chat_session"] để gửi tin nhắn
                            response = st.session_state["chat_session"].send_message(prompt)
                            st.markdown(response.text)
                            # Thêm phản hồi của AI vào lịch sử
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except APIError as e:
                            error_msg = f"Lỗi Gemini API: Không thể gửi tin nhắn. Chi tiết: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        except Exception as e:
                            # Bắt lỗi "client has been closed" nếu nó vẫn xảy ra ở đây (ít có khả năng)
                            error_msg = f"Lỗi không xác định khi gửi tin nhắn: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    else:
        # Hiển thị lỗi nếu không có API key hoặc client không thể khởi tạo
        if not api_key:
            st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'.")

# ******************************* PHẦN BỔ SUNG CHAT GEMINI KẾT THÚC *******************************

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Thêm CSS để định dạng chỉ số màu đỏ booc-đô và in đậm
            st.markdown(
                """
                <style>
                .stMetric {
                    color: #8B0000; /* Màu đỏ booc-đô */
                    font-weight: bold;
                }
                .stMetric > div > div:nth-child(2) > div:nth-child(1) {
                    font-weight: bold; /* In đậm giá trị */
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn (Dùng giá trị giả định hoặc lọc từ file nếu có)
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="**Chỉ số Thanh toán Hiện hành (Năm trước)**",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "Không xác định"
                    )
                with col2:
                    st.metric(
                        label="**Chỉ số Thanh toán Hiện hành (Năm sau)**",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "Không xác định",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf') else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key_analysis = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key_analysis:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key_analysis)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        
                        # Hiển thị kết quả AI đã được định dạng
                        st.markdown(ai_result, unsafe_allow_html=True)
                        
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
