import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from disease_info import DISEASE_INFO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import tempfile
import time
from collections import defaultdict

# ================================
# 페이지 설정
# ================================
st.set_page_config(
    page_title="딸기 병 분류 AI 플랫폼",
    page_icon="🍓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# 커스텀 CSS
# ================================
st.markdown("""
<style>
    /* 메인 배경 그라데이션 */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* 카드 스타일 */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    }
    
    /* 헤더 스타일 */
    .big-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 통계 박스 */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* 프로그레스 바 */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 세션 상태 초기화
# ================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'disease_stats' not in st.session_state:
    st.session_state.disease_stats = {}
if 'video_stats' not in st.session_state:
    st.session_state.video_stats = {}

# ================================
# YOLOv8 모델 로드
# ================================
@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        
        model_path = os.path.join(os.path.dirname(__file__), "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")
        
        model = YOLO(model_path)
        
        class_names = [
            "0",
            "Angular leaf spot",
            "Anthracnose",
            "Fusarium wilt",
            "Gray mold",
            "Leaf spot",
            "Powdery mildew",
            "stawberry",
            "stawberry_1"
        ]
        
        return model, class_names, model_path
    except Exception as e:
        st.error(f"❌ 모델 로드 실패: {e}")
        return None, None, None

# ================================
# 이미지 분석 함수
# ================================
def analyze_image(image, model, class_names, confidence_threshold=50):
    img_array = np.array(image)
    results = model(img_array, conf=confidence_threshold/100.0)
    
    predictions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            predictions.append({
                'x': float(x_center),
                'y': float(y_center),
                'width': float(width),
                'height': float(height),
                'confidence': conf * 100,
                'class': class_names[cls] if cls < len(class_names) else str(cls)
            })
    
    return {'predictions': predictions}

# ================================
# 결과 시각화
# ================================
def draw_predictions(image, predictions):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    colors = {
        "Angular leaf spot": (255, 107, 107),
        "Anthracnose": (255, 159, 67),
        "Fusarium wilt": (196, 119, 210),
        "Gray mold": (130, 130, 130),
        "Leaf spot": (255, 193, 7),
        "Powdery mildew": (129, 212, 250),
    }
    
    for pred in predictions:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        w = int(pred['width'])
        h = int(pred['height'])
        
        color = colors.get(pred['class'], (0, 255, 0))
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 4)
        
        label = f"{pred['class']} {pred['confidence']:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(img_bgr, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(img_bgr, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
    
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ================================
# 동영상 프레임 분석
# ================================
def process_video(video_path, model, class_names, confidence, progress_callback=None):
    """동영상 분석 및 프레임별 통계"""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_stats = []
    disease_counts = defaultdict(int)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 분석
        result = analyze_image(pil_image, model, class_names, confidence)
        predictions = result['predictions']
        
        # 통계 수집
        detected_diseases = [p['class'] for p in predictions]
        for disease in detected_diseases:
            disease_counts[disease] += 1
        
        frame_stats.append({
            'frame': frame_idx,
            'time': frame_idx / fps,
            'detections': len(predictions),
            'diseases': detected_diseases
        })
        
        frame_idx += 1
        
        # 진행률 업데이트
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx / total_frames)
    
    cap.release()
    
    return frame_stats, disease_counts, total_frames, fps

# ================================
# 메인 헤더
# ================================
st.markdown('<h1 class="big-title">🍓 AI 딸기 병 분류 플랫폼</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">YOLOv8 딥러닝으로 딸기 질병을 자동 진단하고 치료제를 추천받으세요</p>', unsafe_allow_html=True)

# ================================
# 사이드바
# ================================
with st.sidebar:
    st.markdown("## ⚙️ 설정")
    
    confidence = st.slider(
        "🎯 검출 신뢰도 임계값 (%)", 
        0, 100, 50, 5,
        help="낮을수록 더 많은 객체를 검출하지만 정확도가 떨어질 수 있습니다."
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 분석 통계")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">총 분석</div>
            <div class="stat-number">{st.session_state.total_analyzed}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">히스토리</div>
            <div class="stat-number">{len(st.session_state.history)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("🗑️ 히스토리 초기화"):
        st.session_state.history = []
        st.session_state.total_analyzed = 0
        st.session_state.disease_stats = {}
        st.session_state.video_stats = {}
        st.success("초기화 완료!")
        st.rerun()

# ================================
# 모델 로드
# ================================
if st.session_state.model is None:
    with st.spinner("🔄 AI 모델을 불러오는 중..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        
        model, class_names, model_location = load_model()
        
        if model is not None:
            st.session_state.model = model
            st.session_state.class_names = class_names
            st.session_state.model_location = model_location
            st.success("✅ AI 모델 준비 완료!")
            st.balloons()
        else:
            st.error("❌ 모델을 불러올 수 없습니다.")
            st.stop()

# ================================
# 탭 메뉴
# ================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 단일 이미지 진단", 
    "📸 다중 이미지 분석", 
    "🎥 동영상 분석", 
    "📊 통계 대시보드", 
    "ℹ️ 사용 가이드"
])

# ================================
# TAB 1: 단일 이미지 진단
# ================================
with tab1:
    st.markdown("### 📤 이미지 업로드")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "딸기 이미지를 선택하세요",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 원본 이미지")
            st.image(image, use_column_width=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("🔍 AI 진단 시작", type="primary", key="analyze_single"):
                with st.spinner("🧠 AI가 분석 중..."):
                    try:
                        result = analyze_image(
                            image, 
                            st.session_state.model, 
                            st.session_state.class_names,
                            confidence
                        )
                        predictions = result.get('predictions', [])
                        
                        if len(predictions) == 0:
                            st.warning("⚠️ 검출된 병징이 없습니다.")
                        else:
                            result_image = draw_predictions(image.copy(), predictions)
                            
                            with col2:
                                st.markdown("#### 🎯 분석 결과")
                                st.image(result_image, use_column_width=True)
                            
                            detected_diseases = set([pred['class'] for pred in predictions])
                            
                            st.session_state.total_analyzed += 1
                            for disease in detected_diseases:
                                st.session_state.disease_stats[disease] = \
                                    st.session_state.disease_stats.get(disease, 0) + 1
                            
                            st.success("✅ 분석 완료!")
                            st.balloons()
                    
                    except Exception as e:
                        st.error(f"❌ 오류: {e}")

# ================================
# TAB 2: 다중 이미지 분석
# ================================
with tab2:
    st.markdown("### 📸 다중 이미지 일괄 분석")
    st.info("💡 여러 이미지를 한번에 업로드하여 일괄 분석할 수 있습니다!")
    
    uploaded_files = st.file_uploader(
        "여러 개의 이미지를 선택하세요 (최대 10개)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="multi_upload"
    )
    
    if uploaded_files:
        st.markdown(f"**업로드된 이미지: {len(uploaded_files)}개**")
        
        if len(uploaded_files) > 10:
            st.warning("⚠️ 한번에 최대 10개까지만 분석할 수 있습니다.")
            uploaded_files = uploaded_files[:10]
        
        if st.button("🚀 전체 이미지 분석 시작", type="primary", key="analyze_multi"):
            st.markdown("---")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_summary = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"분석 중... ({idx+1}/{len(uploaded_files)})")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                image = Image.open(uploaded_file)
                
                result = analyze_image(
                    image,
                    st.session_state.model,
                    st.session_state.class_names,
                    confidence
                )
                predictions = result.get('predictions', [])
                
                detected_diseases = set([pred['class'] for pred in predictions])
                
                # 통계 업데이트
                st.session_state.total_analyzed += 1
                for disease in detected_diseases:
                    st.session_state.disease_stats[disease] = \
                        st.session_state.disease_stats.get(disease, 0) + 1
                
                results_summary.append({
                    'image': image,
                    'filename': uploaded_file.name,
                    'predictions': predictions,
                    'detected_count': len(predictions),
                    'diseases': list(detected_diseases)
                })
            
            status_text.text("✅ 분석 완료!")
            st.success(f"🎉 {len(uploaded_files)}개 이미지 분석 완료!")
            
            # 결과 표시
            st.markdown("---")
            st.markdown("### 📊 분석 결과 요약")
            
            # 요약 통계
            total_detections = sum([r['detected_count'] for r in results_summary])
            all_diseases = set()
            for r in results_summary:
                all_diseases.update(r['diseases'])
            
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("총 검출 객체", total_detections)
            with summary_cols[1]:
                st.metric("검출된 질병 종류", len(all_diseases))
            with summary_cols[2]:
                avg_per_image = total_detections / len(uploaded_files)
                st.metric("이미지당 평균 검출", f"{avg_per_image:.1f}")
            
            # 개별 결과
            st.markdown("### 🖼️ 개별 이미지 결과")
            
            for idx, result in enumerate(results_summary):
                with st.expander(f"📷 {result['filename']} - 검출: {result['detected_count']}개", expanded=(idx==0)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.image(result['image'], caption="원본", use_column_width=True)
                    
                    with col2:
                        if result['predictions']:
                            result_img = draw_predictions(result['image'].copy(), result['predictions'])
                            st.image(result_img, caption="분석 결과", use_column_width=True)
                        else:
                            st.info("검출된 객체 없음")
                    
                    if result['diseases']:
                        st.markdown("**검출된 질병:**")
                        for disease in result['diseases']:
                            disease_info = DISEASE_INFO.get(disease, {})
                            if disease_info:
                                st.markdown(f"- 🦠 {disease_info.get('name_kr', disease)}")

# ================================
# TAB 3: 동영상 분석
# ================================
with tab3:
    st.markdown("### 🎥 동영상 실시간 분석")
    st.info("💡 동영상을 업로드하면 프레임별로 질병을 검출하고 통계를 제공합니다!")
    
    uploaded_video = st.file_uploader(
        "동영상 파일을 선택하세요 (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )
    
    if uploaded_video is not None:
        # 임시 파일로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("🎬 동영상 분석 시작", type="primary", key="analyze_video"):
            st.markdown("---")
            st.markdown("### 📊 분석 진행 중...")
            
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_placeholder.text(f"분석 중... {int(progress*100)}%")
            
            # 동영상 분석
            with st.spinner("🎥 프레임별 분석 중..."):
                frame_stats, disease_counts, total_frames, fps = process_video(
                    video_path,
                    st.session_state.model,
                    st.session_state.class_names,
                    confidence,
                    update_progress
                )
            
            status_placeholder.text("✅ 분석 완료!")
            st.success(f"🎉 총 {total_frames} 프레임 분석 완료!")
            
            # 통계 저장
            st.session_state.video_stats = {
                'frame_stats': frame_stats,
                'disease_counts': disease_counts,
                'total_frames': total_frames,
                'fps': fps
            }
            
            # 결과 표시
            st.markdown("---")
            st.markdown("### 📈 분석 결과")
            
            # 요약 통계
            total_detections = sum([fs['detections'] for fs in frame_stats])
            frames_with_detection = len([fs for fs in frame_stats if fs['detections'] > 0])
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("총 프레임", total_frames)
            with metric_cols[1]:
                st.metric("FPS", fps)
            with metric_cols[2]:
                st.metric("총 검출", total_detections)
            with metric_cols[3]:
                detection_rate = (frames_with_detection / total_frames) * 100
                st.metric("검출 프레임 비율", f"{detection_rate:.1f}%")
            
            # 프레임별 검출 그래프
            st.markdown("### 📊 프레임별 검출 그래프")
            
            df_frames = pd.DataFrame([
                {'프레임': fs['frame'], '시간(초)': fs['time'], '검출 수': fs['detections']}
                for fs in frame_stats
            ])
            
            fig1 = px.line(
                df_frames,
                x='프레임',
                y='검출 수',
                title='프레임별 질병 검출 수',
                labels={'프레임': '프레임 번호', '검출 수': '검출된 객체 수'}
            )
            fig1.update_traces(line_color='#667eea', line_width=3)
            st.plotly_chart(fig1, use_container_width=True)
            
            # 질병별 검출 횟수
            if disease_counts:
                st.markdown("### 🦠 질병별 검출 통계")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 파이 차트
                    fig2 = go.Figure(data=[go.Pie(
                        labels=list(disease_counts.keys()),
                        values=list(disease_counts.values()),
                        hole=0.4
                    )])
                    fig2.update_layout(title="질병 분포")
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    # 바 차트
                    df_disease = pd.DataFrame({
                        '질병': list(disease_counts.keys()),
                        '검출 횟수': list(disease_counts.values())
                    })
                    fig3 = px.bar(
                        df_disease,
                        x='질병',
                        y='검출 횟수',
                        title="질병별 검출 횟수",
                        color='검출 횟수',
                        color_continuous_scale=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # 질병 정보 테이블
                st.markdown("### 📋 검출된 질병 상세")
                for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
                    disease_info = DISEASE_INFO.get(disease, {})
                    if disease_info:
                        with st.expander(f"🦠 {disease_info['name_kr']} - {count}회 검출"):
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.markdown(f"**증상:** {disease_info['symptoms']}")
                                st.markdown(f"**원인:** {disease_info['cause']}")
                            with col_info2:
                                st.markdown(f"**예방법:** {disease_info['prevention']}")
        
        # 임시 파일 정리
        try:
            os.unlink(video_path)
        except:
            pass

# ================================
# TAB 4: 통계 대시보드
# ================================
with tab4:
    st.markdown("### 📊 전체 분석 통계")
    
    if st.session_state.disease_stats:
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.disease_stats.keys()),
            values=list(st.session_state.disease_stats.values()),
            hole=0.4
        )])
        fig.update_layout(title="누적 질병 검출 통계")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 아직 분석된 데이터가 없습니다!")

# ================================
# TAB 5: 사용 가이드
# ================================
with tab5:
    st.markdown("### 📖 사용 가이드")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔬 단일 이미지 진단
        - 1장의 이미지를 정밀 분석
        - 즉시 결과 확인 및 치료제 추천
        
        #### 📸 다중 이미지 분석
        - 최대 10장까지 일괄 분석
        - 여러 샘플 동시 비교
        - 통계적 분석 가능
        """)
    
    with col2:
        st.markdown("""
        #### 🎥 동영상 분석
        - 실시간 프레임별 검출
        - 시간에 따른 질병 추이 분석
        - 검출 빈도 통계 제공
        
        #### ⚙️ 설정 팁
        - 신뢰도 50-70%: 균형잡힌 검출
        - 신뢰도 70% 이상: 정확도 우선
        - 신뢰도 50% 미만: 민감도 우선
        """)

# ================================
# 푸터
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem;'>
    <h3>🍓 딸기 병 분류 AI 플랫폼 v3.0</h3>
    <p style='font-size: 1.1rem;'>
        Powered by YOLOv8 & Streamlit | 단일/다중 이미지 + 동영상 분석 지원
    </p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>
        ⚠️ 본 진단은 참고용이며, 정확한 진단은 전문가와 상담하세요.
    </p>
</div>
""", unsafe_allow_html=True)
