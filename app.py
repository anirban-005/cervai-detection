# app.py — AI Cervical Cancer Detection System
# Save this file as app.py in the same folder as your models/

import streamlit as st
import numpy as np
import cv2
import os
import io
import json
import smtplib
import tempfile
from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import Image as RLImage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')


import gdown
import os

def download_models_if_needed():
    os.makedirs('models', exist_ok=True)

    models_info = {
        'VGG16'      : '1yKv7Sg5CyRNEZMOEcx_URSX72d3ELwh5',
        'ResNet50'   : '1Ps2tsZcHh1E69cmx40ukeItqv6kPQ5yP',
        'InceptionV3': '1g358xYITced26AohwtzNK_inNiFmGW0l'
    }

    for name, file_id in models_info.items():
        path = f'models/{name}.weights.h5'
        if not os.path.exists(path):
            st.warning(f'Downloading {name} model... this may take a few minutes')
            try:
                # Method 1 - gdown direct
                url = f'https://drive.google.com/uc?id={file_id}&export=download&confirm=t'
                gdown.download(url, path, quiet=False, fuzzy=True)

                if os.path.exists(path) and os.path.getsize(path) > 1000:
                    st.success(f'{name} downloaded successfully!')
                else:
                    # Method 2 - gdown with different format
                    os.remove(path) if os.path.exists(path) else None
                    gdown.download(
                        id      = file_id,
                        output  = path,
                        quiet   = False
                    )
                    st.success(f'{name} ready!')

            except Exception as e:
                st.error(f'Failed to download {name}: {str(e)}')

        else:
            size = os.path.getsize(path) / (1024*1024)
            st.sidebar.success(f'{name} ready ({size:.1f} MB)')

download_models_if_needed()




# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title    = "CervAI — Cancer Detection System",
    page_icon     = "🔬",
    layout        = "wide",
    initial_sidebar_state = "expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --primary    : #00d4aa;
    --secondary  : #0a0f1e;
    --accent     : #ff4b6e;
    --card-bg    : #0d1526;
    --text-light : #e0e6f0;
    --text-muted : #8892a4;
    --success    : #00d4aa;
    --warning    : #ffa733;
    --danger     : #ff4b6e;
    --border     : rgba(0, 212, 170, 0.2);
}

* { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2e 50%, #0a1628 100%);
    color: var(--text-light);
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

/* Animated background particles */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(circle at 20% 80%, rgba(0,212,170,0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(255,75,110,0.05) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(0,100,255,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* Hero Header */
.hero-header {
    text-align: center;
    padding: 3rem 2rem 2rem;
    position: relative;
}

.hero-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4aa, #00a8ff, #ff4b6e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -2px;
    line-height: 1.1;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    from { filter: drop-shadow(0 0 20px rgba(0,212,170,0.3)); }
    to   { filter: drop-shadow(0 0 40px rgba(0,212,170,0.6)); }
}

.hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,168,255,0.15));
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 0.4rem 1.5rem;
    font-size: 0.85rem;
    color: var(--primary);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1rem;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,170,0.3); }
    50%       { box-shadow: 0 0 0 10px rgba(0,212,170,0); }
}

/* Cards */
.glass-card {
    background: rgba(13, 21, 38, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary), transparent);
}

.glass-card:hover {
    border-color: rgba(0,212,170,0.4);
    transform: translateY(-2px);
    box-shadow: 0 20px 60px rgba(0,212,170,0.1);
}

/* Section Headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
    margin-left: 1rem;
}

/* Result Cards */
.result-card {
    background: linear-gradient(135deg, rgba(0,212,170,0.1), rgba(0,168,255,0.05));
    border: 1px solid rgba(0,212,170,0.3);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.result-card:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 40px rgba(0,212,170,0.2);
}

.result-value {
    font-size: 2.5rem;
    font-weight: 900;
    color: var(--primary);
    font-family: 'JetBrains Mono', monospace;
}

.result-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.3rem;
}

/* Risk Badge */
.risk-low      { background: rgba(0,212,170,0.2);  color: #00d4aa; border: 1px solid #00d4aa; }
.risk-mild     { background: rgba(255,167,51,0.2); color: #ffa733; border: 1px solid #ffa733; }
.risk-moderate { background: rgba(255,167,51,0.3); color: #ff8c00; border: 1px solid #ff8c00; }
.risk-high     { background: rgba(255,75,110,0.2); color: #ff4b6e; border: 1px solid #ff4b6e; }
.risk-critical { background: rgba(255,0,0,0.3);    color: #ff0000; border: 1px solid #ff0000; }

.risk-badge {
    display: inline-block;
    padding: 0.6rem 2rem;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    animation: riskPulse 2s ease-in-out infinite;
}

@keyframes riskPulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.8; }
}

/* Progress Bars */
.prob-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 50px;
    height: 8px;
    overflow: hidden;
    margin: 0.3rem 0;
}

.prob-bar {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, #00d4aa, #00a8ff);
    transition: width 1s ease;
    animation: barLoad 1s ease-out;
}

@keyframes barLoad {
    from { width: 0%; }
}

/* Upload Area */
.upload-area {
    border: 2px dashed rgba(0,212,170,0.3);
    border-radius: 20px;
    padding: 3rem;
    text-align: center;
    transition: all 0.3s ease;
    background: rgba(0,212,170,0.02);
}

.upload-area:hover {
    border-color: var(--primary);
    background: rgba(0,212,170,0.05);
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: rgba(10,15,30,0.95) !important;
    border-right: 1px solid var(--border) !important;
}

/* Input Fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select {
    background: rgba(13,21,38,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-light) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #00a8ff) !important;
    color: #0a0f1e !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(0,212,170,0.3) !important;
}

/* Metric Cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem;
    text-align: center;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.scanning {
    animation: spin 2s linear infinite;
    display: inline-block;
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
}

/* Status indicators */
.status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--primary);
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 8px;
}

.stSelectbox label, .stTextInput label,
.stNumberInput label, .stFileUploader label {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stRadio label { color: var(--text-light) !important; }
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ─────────────────────────────────────────────────
CLASS_NAMES = [
    'im_Dyskeratotic',
    'im_Koilocytotic',
    'im_Metaplastic',
    'im_Parabasal',
    'im_Superficial-Intermediate'
]
CLASS_DISPLAY = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate'
]
STAGE_MAP = {
    'im_Dyskeratotic'            : 'Stage 3 - Severe Dysplasia (CIN III)',
    'im_Koilocytotic'            : 'Stage 2 - Moderate Dysplasia (CIN II)',
    'im_Metaplastic'             : 'Stage 1 - Mild Dysplasia (CIN I)',
    'im_Parabasal'               : 'Stage 4 - Possible Malignancy',
    'im_Superficial-Intermediate': 'Stage 0 - Normal'
}
RISK_MAP = {
    'im_Superficial-Intermediate': ('LOW RISK',      'risk-low',      '#00d4aa'),
    'im_Metaplastic'             : ('MILD RISK',     'risk-mild',     '#ffa733'),
    'im_Koilocytotic'            : ('MODERATE RISK', 'risk-moderate', '#ff8c00'),
    'im_Dyskeratotic'            : ('HIGH RISK',     'risk-high',     '#ff4b6e'),
    'im_Parabasal'               : ('CRITICAL RISK', 'risk-critical', '#ff0000')
}
ACCURACY_MAP = {
    'VGG16'      : 96.88,
    'ResNet50'   : 95.39,
    'InceptionV3': 92.93,
    'Ensemble'   : 97.20
}
MODEL_NAMES = ['VGG16', 'ResNet50', 'InceptionV3']
LAST_CONV_LAYERS = {
    'VGG16'      : 'block5_conv3',
    'ResNet50'   : 'conv5_block3_out',
    'InceptionV3': 'mixed10'
}


# ─── LOAD MODELS ───────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
    from tensorflow.keras.layers import (Dense, Dropout,
                                          BatchNormalization,
                                          GlobalAveragePooling2D)
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    def build_and_load(model_name, weights_path):
        if model_name == 'VGG16':
            base = VGG16(weights=None, include_top=False,
                         input_shape=(224,224,3), pooling='avg')
        elif model_name == 'ResNet50':
            base = ResNet50(weights=None, include_top=False,
                            input_shape=(224,224,3), pooling='avg')
        elif model_name == 'InceptionV3':
            base = InceptionV3(weights=None, include_top=False,
                               input_shape=(299,299,3), pooling='avg')

        x      = base.output
        x      = BatchNormalization()(x)
        x      = Dense(512, activation='relu',
                       kernel_regularizer=l2(0.001))(x)
        x      = Dropout(0.5)(x)
        x      = Dense(256, activation='relu',
                       kernel_regularizer=l2(0.001))(x)
        x      = Dropout(0.3)(x)
        output = Dense(5, activation='softmax')(x)
        model  = Model(inputs=base.input, outputs=output)

        model.load_weights(weights_path)
        model.compile(
            optimizer = tf.keras.optimizers.Adam(1e-4),
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy']
        )
        return model

    models = {}
    for name in MODEL_NAMES:
        weights_path = f'models/{name}.weights.h5'
        if os.path.exists(weights_path):
            try:
                models[name] = build_and_load(name, weights_path)
                print(f"{name} loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"{name} failed: {str(e)[:80]}")
        else:
            st.sidebar.warning(f"{name} weights not found!")
    return models
@st.cache_resource
def get_preprocess_fn(model_name):
    from tensorflow.keras.applications.vgg16       import preprocess_input as vgg_pre
    from tensorflow.keras.applications.resnet50    import preprocess_input as res_pre
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre
    fn_map   = {'VGG16': vgg_pre, 'ResNet50': res_pre, 'InceptionV3': inc_pre}
    size_map = {'VGG16': (224,224), 'ResNet50': (224,224), 'InceptionV3': (299,299)}
    return fn_map[model_name], size_map[model_name]


# ─── PREDICTION ────────────────────────────────────────────────
def predict_image(image_pil, model_choice, trained_models):
    all_probs = []
    models_to_use = MODEL_NAMES if model_choice == 'Ensemble' else [model_choice]

    for model_name in models_to_use:
        if model_name not in trained_models:
            continue
        model           = trained_models[model_name]
        preprocess_fn, img_size = get_preprocess_fn(model_name)

        img     = np.array(image_pil.convert('RGB'))
        img     = cv2.resize(img, img_size)
        img_pre = preprocess_fn(img.astype(np.float32))
        img_exp = np.expand_dims(img_pre, 0)

        probs = model.predict(img_exp, verbose=0)[0]
        all_probs.append(probs)

    if len(all_probs) == 0:
        st.error("No models loaded! Check model files.")
        return None

    final_probs = np.mean(all_probs, axis=0)
    pred_idx    = int(np.argmax(final_probs))
    cls_name    = CLASS_NAMES[pred_idx]
    confidence  = float(final_probs[pred_idx]) * 100

    return {
    'class_name' : cls_name,
    'class_disp' : CLASS_DISPLAY[pred_idx],
    'stage'      : STAGE_MAP[cls_name],
    'risk'       : RISK_MAP[cls_name][0],
    'risk_class' : RISK_MAP[cls_name][1],
    'risk_color' : RISK_MAP[cls_name][2],
    'confidence' : confidence,
    'all_probs'  : final_probs,
    'pred_idx'   : int(pred_idx)
}


# ─── GRAD-CAM ──────────────────────────────────────────────────
def generate_gradcam(image_pil, model_name, trained_models, pred_idx):
    if model_name not in trained_models:
        return None

    from tensorflow.keras.models import Model
    model           = trained_models[model_name]
    preprocess_fn, img_size = get_preprocess_fn(model_name)
    last_conv       = LAST_CONV_LAYERS[model_name]

    img     = np.array(image_pil.convert('RGB'))
    img_res = cv2.resize(img, img_size)
    img_pre = preprocess_fn(img_res.astype(np.float32))
    img_exp = np.expand_dims(img_pre, 0)

    try:
        grad_model = Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv).output, model.output]
        )
        import tensorflow as tf
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_exp)
            class_ch        = preds[:, pred_idx]

        grads        = tape.gradient(class_ch, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out     = conv_out[0]
        heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (
                       tf.math.reduce_max(heatmap) + 1e-8)
        heatmap      = heatmap.numpy()

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(
            cv2.resize(heatmap_uint8, img_size), cv2.COLORMAP_JET
        )
        heatmap_rgb  = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_res, 0.6, heatmap_rgb, 0.4, 0)
        return img_res, heatmap, superimposed

    except Exception as e:
        return None


# ─── PDF GENERATION ────────────────────────────────────────────
def generate_pdf_report(patient_info, prediction, gradcam_img,
                         original_img, model_choice):
    buffer    = io.BytesIO()
    doc       = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm,   bottomMargin=1.5*cm
    )
    styles    = getSampleStyleSheet()
    story     = []
    W, H      = A4

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'],
        fontSize=22, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#00d4aa'),
        alignment=TA_CENTER, spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'SubStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#8892a4'),
        alignment=TA_CENTER, spaceAfter=4
    )
    section_style = ParagraphStyle(
        'SectionStyle', parent=styles['Heading2'],
        fontSize=13, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#00d4aa'),
        spaceBefore=12, spaceAfter=6
    )
    normal_style = ParagraphStyle(
        'NormalStyle', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#2c3e50'),
        spaceAfter=4
    )
    bold_style = ParagraphStyle(
        'BoldStyle', parent=styles['Normal'],
        fontSize=10, fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1a1a2e'), spaceAfter=4
    )

    # ── HEADER ──
    story.append(Paragraph("CervAI Medical Report", title_style))
    story.append(Paragraph(
        "AI-Assisted Cervical Cancer Detection System", subtitle_style
    ))
    story.append(Paragraph(
        f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        subtitle_style
    ))
    story.append(HRFlowable(
        width="100%", thickness=2,
        color=colors.HexColor('#00d4aa'), spaceAfter=12
    ))

    # ── PATIENT INFORMATION ──
    story.append(Paragraph("Patient Information", section_style))
    patient_data = [
        ['Field', 'Details'],
        ['Patient Name',    patient_info['name']],
        ['Age',             f"{patient_info['age']} years"],
        ['Height',          f"{patient_info['height']} cm"],
        ['Weight',          f"{patient_info['weight']} kg"],
        ['BMI',             f"{patient_info['bmi']:.1f}"],
        ['WhatsApp',        patient_info['whatsapp']],
        ['Email',           patient_info['email']],
        ['Report Date',     datetime.now().strftime('%d/%m/%Y')],
        ['Report ID',       f"CERVAI-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
    ]
    patient_table = Table(patient_data, colWidths=[5*cm, 11*cm])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#00d4aa')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  11),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',    (0,1), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,1), (-1,-1), 10),
        ('BACKGROUND',  (0,1), (-1,-1), colors.HexColor('#f8fffe')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8fffe'), colors.HexColor('#e8fdf9')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#00d4aa')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ROWHEIGHT',   (0,0), (-1,-1), 20),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # ── AI PREDICTION RESULTS ──
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#e0e0e0'), spaceAfter=8
    ))
    story.append(Paragraph("AI Prediction Results", section_style))

    risk_color_map = {
        'LOW RISK'     : '#00d4aa',
        'MILD RISK'    : '#ffa733',
        'MODERATE RISK': '#ff8c00',
        'HIGH RISK'    : '#ff4b6e',
        'CRITICAL RISK': '#ff0000'
    }
    risk_hex = risk_color_map.get(prediction['risk'], '#00d4aa')

    result_data = [
        ['Parameter',      'Result'],
        ['Cell Type',       prediction['class_disp']],
        ['Cancer Stage',    prediction['stage']],
        ['Risk Level',      prediction['risk']],
        ['Confidence',      f"{prediction['confidence']:.2f}%"],
        ['Model Used',      model_choice],
        ['Model Accuracy',  f"{ACCURACY_MAP.get(model_choice, 0):.2f}%"],
    ]
    result_table = Table(result_data, colWidths=[5*cm, 11*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,0),  11),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME',    (0,1), (0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',    (0,1), (-1,-1), 10),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.HexColor('#f8fffe'), colors.HexColor('#e8fdf9')]),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING',     (0,0), (-1,-1), 8),
        ('ROWHEIGHT',   (0,0), (-1,-1), 20),
        ('TEXTCOLOR',   (1,3), (1,3),   colors.HexColor(risk_hex)),
        ('FONTNAME',    (1,3), (1,3),   'Helvetica-Bold'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 12))

    # ── PROBABILITY TABLE ──
    story.append(Paragraph("Class Probability Breakdown", section_style))
    prob_data = [['Cell Class', 'Probability', 'Stage']]
    for i, (cls_n, cls_d) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY)):
        prob = prediction['all_probs'][i] * 100
        stage = STAGE_MAP[cls_n].split(' - ')[0]
        marker = ' (PREDICTED)' if i == prediction['pred_idx'] else ''
        prob_data.append([f"{cls_d}{marker}", f"{prob:.2f}%", stage])

    prob_table = Table(prob_data, colWidths=[6*cm, 4*cm, 6*cm])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0),  colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',   (0,0), (-1,0),  colors.white),
        ('FONTNAME',    (0,0), (-1,0),  'Helvetica-Bold'),
        ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
        ('ALIGN',       (1,0), (1,-1),  'CENTER'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID',        (0,0), (-1,-1), 0.3, colors.HexColor('#dddddd')),
        ('PADDING',     (0,0), (-1,-1), 6),
        ('FONTNAME',    (0, prediction['pred_idx']+1),
                        (-1, prediction['pred_idx']+1), 'Helvetica-Bold'),
        ('BACKGROUND',  (0, prediction['pred_idx']+1),
                        (-1, prediction['pred_idx']+1),
                        colors.HexColor('#e8fdf9')),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 12))

    # ── GRAD-CAM IMAGES ──
    if gradcam_img is not None:
        story.append(HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor('#e0e0e0'), spaceAfter=8
        ))
        story.append(Paragraph("Grad-CAM Explainability Analysis", section_style))
        story.append(Paragraph(
            "The heatmap highlights the cellular regions that influenced "
            "the AI prediction. Red/warm areas indicate high activation zones.",
            normal_style
        ))
        story.append(Spacer(1, 8))

        orig_buf = io.BytesIO()
        Image.fromarray(gradcam_img[0]).save(orig_buf, format='PNG')
        orig_buf.seek(0)

        heat_buf = io.BytesIO()
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * gradcam_img[1]), cv2.COLORMAP_JET
        )
        Image.fromarray(
            cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        ).save(heat_buf, format='PNG')
        heat_buf.seek(0)

        over_buf = io.BytesIO()
        Image.fromarray(gradcam_img[2]).save(over_buf, format='PNG')
        over_buf.seek(0)

        img_table = Table([[
            RLImage(orig_buf, width=5*cm, height=5*cm),
            RLImage(heat_buf, width=5*cm, height=5*cm),
            RLImage(over_buf, width=5*cm, height=5*cm),
        ],[
            Paragraph("Original Image",  normal_style),
            Paragraph("Grad-CAM Heatmap", normal_style),
            Paragraph("Overlay",          normal_style),
        ]])
        img_table.setStyle(TableStyle([
            ('ALIGN',   (0,0), (-1,-1), 'CENTER'),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 12))

    # ── RECOMMENDATIONS ──
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#e0e0e0'), spaceAfter=8
    ))
    story.append(Paragraph("Clinical Recommendations", section_style))

    recommendations = {
        'LOW RISK'     : "No immediate action required. Continue routine annual Pap smear screening. Maintain regular gynecological check-ups.",
        'MILD RISK'    : "Mild cellular changes detected. Follow-up colposcopy recommended within 6 months. Monitor with repeat Pap smear.",
        'MODERATE RISK': "Moderate dysplasia detected. Immediate colposcopy and biopsy recommended. Consult a gynecologist within 2-4 weeks.",
        'HIGH RISK'    : "Severe dysplasia detected. Urgent specialist referral required. LEEP or cone biopsy may be necessary. Consult immediately.",
        'CRITICAL RISK': "Critical findings detected. IMMEDIATE specialist consultation required. Do not delay seeking medical attention."
    }
    story.append(Paragraph(
        recommendations.get(prediction['risk'], ''),
        normal_style
    ))
    story.append(Spacer(1, 8))

    # Disclaimer
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor('#e0e0e0'), spaceAfter=8
    ))
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#999999'),
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI system and is intended "
        "for screening purposes only. It does not constitute a medical diagnosis. "
        "Please consult a qualified medical professional for proper diagnosis and treatment.",
        disclaimer_style
    ))
    story.append(Paragraph(
        "CervAI Detection System | Powered by Deep Learning | "
        f"Report ID: CERVAI-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ─── EMAIL SENDING ─────────────────────────────────────────────
def send_email(receiver_email, patient_name, prediction, pdf_buffer):
    try:
        sender_email    = st.secrets.get("EMAIL_ADDRESS", "")
        sender_password = st.secrets.get("EMAIL_PASSWORD", "")

        if not sender_email or not sender_password:
            return False, "Email credentials not configured in secrets"

        msg             = MIMEMultipart()
        msg['From']     = sender_email
        msg['To']       = receiver_email
        msg['Subject']  = f"CervAI Medical Report - {patient_name}"

        body = f"""
Dear {patient_name},

Your cervical cancer screening report is ready.

RESULTS SUMMARY:
- Cell Type    : {prediction['class_disp']}
- Cancer Stage : {prediction['stage']}
- Risk Level   : {prediction['risk']}
- Confidence   : {prediction['confidence']:.2f}%

Please find your detailed PDF report attached.

IMPORTANT: This is an AI-assisted screening report.
Please consult your doctor for proper diagnosis.

Regards,
CervAI Detection System
        """

        msg.attach(MIMEText(body, 'plain'))

        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_buffer.getvalue())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename="CervAI_Report_{patient_name}.pdf"'
        )
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"

    except Exception as e:
        return False, str(e)


# ─── WHATSAPP SENDING ──────────────────────────────────────────
def send_whatsapp(whatsapp_number, patient_name, prediction):
    try:
        import urllib.parse

        # Exact same content as email
        message_text = (
            f"CervAI Medical Report\n\n"
            f"Dear {patient_name},\n\n"
            f"Your cervical cancer screening report is ready.\n\n"
            f"RESULTS SUMMARY:\n"
            f"Patient Name   : {patient_name}\n"
            f"Cell Type      : {prediction['class_disp']}\n"
            f"Cancer Stage   : {prediction['stage']}\n"
            f"Risk Level     : {prediction['risk']}\n"
            f"Confidence     : {prediction['confidence']:.2f}%\n\n"
            f"CLASS PROBABILITIES:\n"
            f"Dyskeratotic              : {prediction['all_probs'][0]*100:.2f}%\n"
            f"Koilocytotic              : {prediction['all_probs'][1]*100:.2f}%\n"
            f"Metaplastic               : {prediction['all_probs'][2]*100:.2f}%\n"
            f"Parabasal                 : {prediction['all_probs'][3]*100:.2f}%\n"
            f"Superficial-Intermediate  : {prediction['all_probs'][4]*100:.2f}%\n\n"
            f"RECOMMENDATION:\n"
            f"{get_recommendation(prediction['risk'])}\n\n"
            f"Report Date    : {datetime.now().strftime('%d/%m/%Y %I:%M %p')}\n"
            f"Report ID      : CERVAI-{datetime.now().strftime('%Y%m%d%H%M%S')}\n\n"
            f"IMPORTANT: This is an AI-assisted screening report.\n"
            f"Please consult your doctor for proper diagnosis.\n\n"
            f"Regards,\n"
            f"CervAI Detection System"
        )

        # Properly encode message
        encoded_message = urllib.parse.quote(message_text)

        # Clean phone number
        phone   = whatsapp_number.strip().replace('+','').replace(' ','').replace('-','')

        # Create WhatsApp link
        wa_link = f"https://wa.me/{phone}?text={encoded_message}"

        return True, wa_link

    except Exception as e:
        return False, str(e)


def get_recommendation(risk):
    recommendations = {
        'LOW RISK'     : "No immediate action required. Continue routine annual Pap smear screening.",
        'MILD RISK'    : "Mild cellular changes detected. Follow-up colposcopy recommended within 6 months.",
        'MODERATE RISK': "Moderate dysplasia detected. Immediate colposcopy and biopsy recommended.",
        'HIGH RISK'    : "Severe dysplasia detected. Urgent specialist referral required. Consult immediately.",
        'CRITICAL RISK': "Critical findings detected. IMMEDIATE specialist consultation required."
    }
    return recommendations.get(risk, '')

# ─── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
        <div style='font-size:3rem;'>🔬</div>
        <div style='font-size:1.3rem; font-weight:700;
                    color:#00d4aa; letter-spacing:2px;'>CervAI</div>
        <div style='font-size:0.75rem; color:#8892a4;
                    letter-spacing:1px;'>DETECTION SYSTEM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='color:#8892a4; font-size:0.8rem;"
        "text-transform:uppercase; letter-spacing:1px;'>"
        "Model Selection</div>",
        unsafe_allow_html=True
    )

    model_choice = st.radio(
        "Select Model",
        options = ['Ensemble (Recommended)', 'VGG16', 'ResNet50', 'InceptionV3'],
        label_visibility = 'collapsed'
    )
    model_key = model_choice.replace(' (Recommended)', '')

    st.markdown("---")

    # Model accuracy display
    st.markdown(
        "<div style='color:#8892a4; font-size:0.8rem;"
        "text-transform:uppercase; letter-spacing:1px;"
        "margin-bottom:0.5rem;'>Model Performance</div>",
        unsafe_allow_html=True
    )
    for m, acc in ACCURACY_MAP.items():
        color = '#00d4aa' if m == model_key else '#8892a4'
        bold  = 'bold'   if m == model_key else 'normal'
        st.markdown(
            f"<div style='display:flex; justify-content:space-between;"
            f"padding:0.3rem 0; color:{color}; font-weight:{bold};'>"
            f"<span>{m}</span><span>{acc:.2f}%</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        "<div style='color:#8892a4; font-size:0.75rem;"
        "text-align:center; line-height:1.6;'>"
        "AI-powered cervical cancer<br>detection using Transfer Learning<br>"
        "VGG16 + ResNet50 + InceptionV3<br><br>"
        "<span style='color:#00d4aa;'>v1.0.0</span></div>",
        unsafe_allow_html=True
    )


# ─── MAIN CONTENT ──────────────────────────────────────────────
trained_models = load_all_models()

# Hero Header
st.markdown("""
<div class='hero-header fade-in'>
    <div class='hero-badge'>
        <span class='status-dot'></span>AI System Online
    </div>
    <div class='hero-title'>CervAI</div>
    <div class='hero-subtitle'>
        Cervical Cancer Detection & Stage Prediction
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)
stats = [
    (col1, "96.88%", "Best Accuracy",  "#00d4aa"),
    (col2, "0.9975", "Best AUC Score", "#00a8ff"),
    (col3, "5",      "Cell Classes",   "#ffa733"),
    (col4, "3",      "CNN Models",     "#ff4b6e"),
]
for col, val, label, color in stats:
    with col:
        st.markdown(f"""
        <div class='result-card fade-in'>
            <div class='result-value' style='color:{color};'>{val}</div>
            <div class='result-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs([
    "🔬 Detection",
    "📊 Model Info",
    "📖 About"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ══════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: Patient Form + Upload ──
    with col_left:
        st.markdown(
            "<div class='section-header'>👤 Patient Information</div>",
            unsafe_allow_html=True
        )

        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                patient_name = st.text_input(
                    "Full Name", placeholder="John Doe"
                )
            with c2:
                patient_age = st.number_input(
                    "Age", min_value=1, max_value=120, value=30
                )

            c3, c4 = st.columns(2)
            with c3:
                patient_height = st.number_input(
                    "Height (cm)", min_value=50, max_value=250, value=160
                )
            with c4:
                patient_weight = st.number_input(
                    "Weight (kg)", min_value=10, max_value=300, value=60
                )

            bmi = patient_weight / ((patient_height/100) ** 2)
            st.markdown(
                f"<div style='color:#00d4aa; font-size:0.9rem;"
                f"margin:-0.5rem 0 0.5rem;'>"
                f"BMI: <strong>{bmi:.1f}</strong></div>",
                unsafe_allow_html=True
            )

            c5, c6 = st.columns(2)
            with c5:
                patient_whatsapp = st.text_input(
                    "WhatsApp Number",
                    placeholder="919876543210"
                )
            with c6:
                patient_email = st.text_input(
                    "Email Address",
                    placeholder="patient@email.com"
                )

        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>🖼 Upload Pap Smear Image</div>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload Image",
            type         = ['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility = 'collapsed'
        )

        if uploaded_file:
            image_pil = Image.open(uploaded_file)
            st.image(
                image_pil,
                caption      = "Uploaded Pap Smear Image",
                use_column_width = True
            )

        # Delivery options
        st.markdown("<div class='custom-divider'></div>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>📤 Report Delivery</div>",
            unsafe_allow_html=True
        )
        delivery = st.multiselect(
            "Send Report Via",
            options      = ['Email', 'WhatsApp'],
            default      = ['Email'],
            label_visibility = 'collapsed'
        )

        # Analyze button
        analyze_btn = st.button(
            "🔬 ANALYZE IMAGE",
            use_container_width = True
        )

    # ── RIGHT: Results ──
    with col_right:
        st.markdown(
            "<div class='section-header'>📊 Analysis Results</div>",
            unsafe_allow_html=True
        )

        if analyze_btn and uploaded_file:
            if not patient_name:
                st.error("Please enter patient name!")
            else:
                with st.spinner("Analyzing image..."):
                    image_pil  = Image.open(uploaded_file)
                    prediction = predict_image(
                        image_pil, model_key, trained_models
                    )

                    gradcam = None
                    if prediction is not None:
                        gc_model = model_key if model_key != 'Ensemble' else 'VGG16'
                        gradcam  = generate_gradcam(
                            image_pil, gc_model,
                            trained_models, int(prediction['pred_idx'])
                        )

                if prediction is None:
                    st.error("Models not loaded yet. Please wait and try again.")
                    st.stop()
                
                # Results display
                st.markdown(f"""
                <div class='glass-card fade-in'>
                    <div style='text-align:center; margin-bottom:1rem;'>
                        <div style='font-size:0.85rem; color:#8892a4;
                                    text-transform:uppercase;
                                    letter-spacing:2px;'>Predicted Cell Type</div>
                        <div style='font-size:2rem; font-weight:900;
                                    color:#00d4aa; margin:0.3rem 0;'>
                            {prediction['class_disp']}
                        </div>
                        <div style='font-size:0.9rem; color:#8892a4;'>
                            {prediction['stage']}
                        </div>
                    </div>
                    <div style='text-align:center; margin:1rem 0;'>
                        <span class='risk-badge {prediction["risk_class"]}'>
                            {prediction['risk']}
                        </span>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:0.8rem; color:#8892a4;
                                    text-transform:uppercase; letter-spacing:1px;'>
                            Confidence
                        </div>
                        <div style='font-size:2.5rem; font-weight:900;
                                    color:{prediction["risk_color"]};
                                    font-family:JetBrains Mono, monospace;'>
                            {prediction['confidence']:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown(
                    "<div style='color:#8892a4; font-size:0.8rem;"
                    "text-transform:uppercase; letter-spacing:1px;"
                    "margin:1rem 0 0.5rem;'>Class Probabilities</div>",
                    unsafe_allow_html=True
                )
                for i, (cls_d, prob) in enumerate(
                    zip(CLASS_DISPLAY, prediction['all_probs'])
                ):
                    pct   = prob * 100
                    color = '#00d4aa' if i == prediction['pred_idx'] else '#8892a4'
                    st.markdown(f"""
                    <div style='margin:0.4rem 0;'>
                        <div style='display:flex; justify-content:space-between;
                                    margin-bottom:3px;'>
                            <span style='font-size:0.85rem; color:{color};
                                         font-weight:{"700" if i == prediction["pred_idx"] else "400"};'>
                                {cls_d}
                            </span>
                            <span style='font-size:0.85rem; color:{color};
                                         font-family:JetBrains Mono,monospace;'>
                                {pct:.1f}%
                            </span>
                        </div>
                        <div class='prob-bar-container'>
                            <div class='prob-bar'
                                 style='width:{pct}%;
                                        background:{"linear-gradient(90deg,#00d4aa,#00a8ff)" if i == prediction["pred_idx"] else "rgba(136,146,164,0.3)"};'>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Grad-CAM
                if gradcam:
                    st.markdown(
                        "<div class='custom-divider'></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        "<div style='color:#8892a4; font-size:0.8rem;"
                        "text-transform:uppercase; letter-spacing:1px;"
                        "margin-bottom:0.5rem;'>Grad-CAM Explainability</div>",
                        unsafe_allow_html=True
                    )
                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        st.image(gradcam[0], caption="Original",  use_column_width=True)
                    with gc2:
                        heatmap_vis = cv2.applyColorMap(
                            np.uint8(255 * gradcam[1]), cv2.COLORMAP_JET
                        )
                        st.image(
                            cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB),
                            caption = "Heatmap",
                            use_column_width = True
                        )
                    with gc3:
                        st.image(gradcam[2], caption="Overlay", use_column_width=True)

                # Generate PDF
                patient_info = {
                    'name'    : patient_name,
                    'age'     : patient_age,
                    'height'  : patient_height,
                    'weight'  : patient_weight,
                    'bmi'     : bmi,
                    'whatsapp': patient_whatsapp,
                    'email'   : patient_email
                }

                pdf_buffer = generate_pdf_report(
                    patient_info, prediction,
                    gradcam, image_pil, model_key
                )

                # Download PDF
                st.markdown(
                    "<div class='custom-divider'></div>",
                    unsafe_allow_html=True
                )
                st.download_button(
                    label            = "📥 DOWNLOAD PDF REPORT",
                    data             = pdf_buffer,
                    file_name        = f"CervAI_Report_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime             = "application/pdf",
                    use_container_width = True
                )

                # Send reports
                if delivery:
                    st.markdown(
                        "<div style='color:#8892a4; font-size:0.8rem;"
                        "text-transform:uppercase; letter-spacing:1px;"
                        "margin:0.5rem 0;'>Sending Report...</div>",
                        unsafe_allow_html=True
                    )

                if 'Email' in delivery and patient_email:
                    pdf_buffer.seek(0)
                    success, msg = send_email(
                        patient_email, patient_name,
                        prediction, pdf_buffer
                    )
                    if success:
                        st.success(f"Email sent to {patient_email}")
                    else:
                        st.warning(f"Email: {msg}")

                if 'WhatsApp' in delivery and patient_whatsapp:
                    success, wa_link = send_whatsapp(
                        patient_whatsapp, patient_name, prediction
                    )
                    if success:
                        st.markdown(
                            f"""
                            <a href='{wa_link}' target='_blank'
                               style='text-decoration:none;'>
                                <div style='
                                    background: linear-gradient(135deg, #25D366, #128C7E);
                                    color: white;
                                    padding: 1rem 2rem;
                                    border-radius: 12px;
                                    font-size: 1rem;
                                    font-weight: 700;
                                    cursor: pointer;
                                    width: 100%;
                                    text-align: center;
                                    margin: 0.5rem 0;
                                    letter-spacing: 1px;
                                    box-shadow: 0 4px 15px rgba(37,211,102,0.3);
                                '>
                                    📱 SEND WHATSAPP MESSAGE
                                </div>
                            </a>
                            <div style='
                                color: #8892a4;
                                font-size: 0.8rem;
                                text-align: center;
                                margin-top: 0.3rem;
                            '>
                                Click above — WhatsApp opens with full report ready to send
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"WhatsApp error: {wa_link}")

        elif analyze_btn and not uploaded_file:
            st.warning("Please upload a Pap smear image first!")
        else:
            st.markdown("""
            <div class='glass-card' style='text-align:center; padding:4rem 2rem;'>
                <div style='font-size:4rem; margin-bottom:1rem;'>🔬</div>
                <div style='font-size:1.2rem; color:#8892a4;
                            font-weight:300; line-height:1.8;'>
                    Fill in patient details<br>
                    Upload a Pap smear image<br>
                    Click Analyze to get results
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 2 — MODEL INFO
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<div class='section-header'>Model Performance Comparison</div>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, (name, acc) in zip(
        [m1, m2, m3, m4],
        ACCURACY_MAP.items()
    ):
        with col:
            color = '#00d4aa' if name == 'Ensemble' else '#00a8ff'
            st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size:0.8rem; color:#8892a4;
                            text-transform:uppercase;
                            letter-spacing:1px;'>{name}</div>
                <div style='font-size:2.2rem; font-weight:900;
                            color:{color};
                            font-family:JetBrains Mono,monospace;'>
                    {acc:.2f}%
                </div>
                <div style='font-size:0.75rem; color:#8892a4;'>Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # Cancer Stage Table
    st.markdown(
        "<div class='section-header'>Cancer Stage Mapping</div>",
        unsafe_allow_html=True
    )
    stage_colors = {
        'Stage 0': '#00d4aa',
        'Stage 1': '#ffa733',
        'Stage 2': '#ff8c00',
        'Stage 3': '#ff4b6e',
        'Stage 4': '#ff0000'
    }
    for cls_n, cls_d in zip(CLASS_NAMES, CLASS_DISPLAY):
        stage = STAGE_MAP[cls_n]
        risk  = RISK_MAP[cls_n][0]
        color = RISK_MAP[cls_n][2]
        st.markdown(f"""
        <div class='glass-card' style='padding:1rem 1.5rem;
             margin:0.4rem 0; display:flex;
             justify-content:space-between; align-items:center;'>
            <div>
                <span style='font-weight:700; color:#e0e6f0;'>{cls_d}</span>
                <span style='color:#8892a4; font-size:0.85rem;
                             margin-left:1rem;'>{stage}</span>
            </div>
            <span class='risk-badge' style='color:{color};
                         border:1px solid {color};
                         background:rgba(255,255,255,0.05);
                         padding:0.3rem 1rem;
                         font-size:0.8rem;'>{risk}</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='glass-card fade-in'>
        <div class='section-header'>About CervAI</div>
        <p style='color:#e0e6f0; line-height:1.8; font-size:1rem;'>
        CervAI is an AI-powered cervical cancer detection system that uses
        deep learning and transfer learning to analyze Pap smear images.
        The system classifies cells into 5 categories and predicts the
        corresponding cancer stage with high accuracy.
        </p>

        <div class='section-header' style='margin-top:1.5rem;'>
        Technology Stack
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr;
                    gap:1rem; margin-top:0.5rem;'>
            <div style='background:rgba(0,212,170,0.05);
                        border:1px solid rgba(0,212,170,0.2);
                        border-radius:12px; padding:1rem;'>
                <div style='color:#00d4aa; font-weight:700;
                            margin-bottom:0.5rem;'>Models</div>
                <div style='color:#8892a4; font-size:0.9rem;
                            line-height:2;'>
                    VGG16 (96.88%)<br>
                    ResNet50 (95.39%)<br>
                    InceptionV3 (92.93%)<br>
                    Ensemble (Best)
                </div>
            </div>
            <div style='background:rgba(0,168,255,0.05);
                        border:1px solid rgba(0,168,255,0.2);
                        border-radius:12px; padding:1rem;'>
                <div style='color:#00a8ff; font-weight:700;
                            margin-bottom:0.5rem;'>Features</div>
                <div style='color:#8892a4; font-size:0.9rem;
                            line-height:2;'>
                    Transfer Learning<br>
                    Grad-CAM Explainability<br>
                    PDF Report Generation<br>
                    WhatsApp + Email Delivery
                </div>
            </div>
        </div>

        <div class='section-header' style='margin-top:1.5rem;'>
        Disclaimer
        </div>
        <p style='color:#8892a4; font-size:0.9rem; line-height:1.8;'>
        This system is intended for screening and research purposes only.
        It does not replace professional medical diagnosis. Always consult
        a qualified healthcare professional for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)