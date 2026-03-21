import streamlit as st
import numpy as np
import cv2
import os
import io
import smtplib
import requests
from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tensorflow as tf
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import Image as RLImage
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CervAI - Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    'im_Dyskeratotic': 'Stage 3 - Severe Dysplasia (CIN III)',
    'im_Koilocytotic': 'Stage 2 - Moderate Dysplasia (CIN II)',
    'im_Metaplastic': 'Stage 1 - Mild Dysplasia (CIN I)',
    'im_Parabasal': 'Stage 4 - Possible Malignancy',
    'im_Superficial-Intermediate': 'Stage 0 - Normal'
}
RISK_MAP = {
    'im_Superficial-Intermediate': ('LOW RISK', 'risk-low', '#00d4aa'),
    'im_Metaplastic': ('MILD RISK', 'risk-mild', '#ffa733'),
    'im_Koilocytotic': ('MODERATE RISK', 'risk-moderate', '#ff8c00'),
    'im_Dyskeratotic': ('HIGH RISK', 'risk-high', '#ff4b6e'),
    'im_Parabasal': ('CRITICAL RISK', 'risk-critical', '#ff0000')
}
ACCURACY_MAP = {
    'VGG16': 96.88,
    'ResNet50': 95.39,
    'InceptionV3': 92.93,
    'Ensemble': 97.20
}
MODEL_NAMES = ['VGG16', 'ResNet50', 'InceptionV3']
LAST_CONV_LAYERS = {
    'VGG16': 'block5_conv3',
    'ResNet50': 'conv5_block3_out',
    'InceptionV3': 'mixed10'
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
    --primary: #00d4aa;
    --card-bg: #0d1526;
    --text-light: #e0e6f0;
    --text-muted: #8892a4;
    --border: rgba(0,212,170,0.2);
}
* { font-family: 'Outfit', sans-serif; }
.stApp { background: linear-gradient(135deg,#0a0f1e 0%,#0d1a2e 50%,#0a1628 100%); color: var(--text-light); }
#MainMenu, footer, header { visibility: hidden; }
.hero-title { font-size:4rem; font-weight:900; background:linear-gradient(135deg,#00d4aa,#00a8ff,#ff4b6e); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; letter-spacing:-2px; line-height:1.1; animation:titleGlow 3s ease-in-out infinite alternate; }
@keyframes titleGlow { from{filter:drop-shadow(0 0 20px rgba(0,212,170,0.3));} to{filter:drop-shadow(0 0 40px rgba(0,212,170,0.6));} }
.hero-subtitle { font-size:1.2rem; color:var(--text-muted); font-weight:300; letter-spacing:3px; text-transform:uppercase; margin-top:0.5rem; }
.hero-badge { display:inline-block; background:linear-gradient(135deg,rgba(0,212,170,0.15),rgba(0,168,255,0.15)); border:1px solid var(--border); border-radius:50px; padding:0.4rem 1.5rem; font-size:0.85rem; color:var(--primary); letter-spacing:2px; text-transform:uppercase; margin-bottom:1rem; animation:pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,170,0.3);} 50%{box-shadow:0 0 0 10px rgba(0,212,170,0);} }
.glass-card { background:rgba(13,21,38,0.8); backdrop-filter:blur(20px); border:1px solid var(--border); border-radius:20px; padding:2rem; margin:1rem 0; transition:all 0.3s ease; position:relative; overflow:hidden; }
.glass-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--primary),transparent); }
.glass-card:hover { border-color:rgba(0,212,170,0.4); transform:translateY(-2px); box-shadow:0 20px 60px rgba(0,212,170,0.1); }
.section-header { font-size:1.4rem; font-weight:700; color:var(--primary); margin-bottom:1.5rem; display:flex; align-items:center; gap:0.5rem; }
.result-card { background:linear-gradient(135deg,rgba(0,212,170,0.1),rgba(0,168,255,0.05)); border:1px solid rgba(0,212,170,0.3); border-radius:16px; padding:1.5rem; text-align:center; transition:all 0.3s ease; }
.result-value { font-size:2.5rem; font-weight:900; color:var(--primary); font-family:'JetBrains Mono',monospace; }
.result-label { font-size:0.85rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:2px; margin-top:0.3rem; }
.risk-low { background:rgba(0,212,170,0.2); color:#00d4aa; border:1px solid #00d4aa; }
.risk-mild { background:rgba(255,167,51,0.2); color:#ffa733; border:1px solid #ffa733; }
.risk-moderate { background:rgba(255,167,51,0.3); color:#ff8c00; border:1px solid #ff8c00; }
.risk-high { background:rgba(255,75,110,0.2); color:#ff4b6e; border:1px solid #ff4b6e; }
.risk-critical { background:rgba(255,0,0,0.3); color:#ff0000; border:1px solid #ff0000; }
.risk-badge { display:inline-block; padding:0.6rem 2rem; border-radius:50px; font-size:1.1rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; animation:riskPulse 2s ease-in-out infinite; }
@keyframes riskPulse { 0%,100%{opacity:1;} 50%{opacity:0.8;} }
.prob-bar-container { background:rgba(255,255,255,0.05); border-radius:50px; height:8px; overflow:hidden; margin:0.3rem 0; }
.prob-bar { height:100%; border-radius:50px; background:linear-gradient(90deg,#00d4aa,#00a8ff); transition:width 1s ease; }
.custom-divider { height:1px; background:linear-gradient(90deg,transparent,var(--border),transparent); margin:2rem 0; }
.status-dot { width:10px; height:10px; border-radius:50%; background:var(--primary); display:inline-block; animation:pulse 2s ease-in-out infinite; margin-right:8px; }
.stButton > button { background:linear-gradient(135deg,#00d4aa,#00a8ff) !important; color:#0a0f1e !important; font-weight:700 !important; border:none !important; border-radius:12px !important; padding:0.7rem 2rem !important; font-size:1rem !important; letter-spacing:1px !important; text-transform:uppercase !important; }
@keyframes fadeInUp { from{opacity:0;transform:translateY(30px);} to{opacity:1;transform:translateY(0);} }
.fade-in { animation:fadeInUp 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────

def download_models_if_needed():
    os.makedirs('models', exist_ok=True)

    # Remove obviously corrupted (too small) files first
    for name in MODEL_NAMES:
        path = 'models/' + name + '.weights.h5'
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb < 40:
                os.remove(path)
                st.sidebar.warning(name + ' corrupted file deleted – re-downloading')

    models_info = {
        'VGG16':       '1MJFdFH1ulb2Oz26_1hYiD_OPeu1w4RcC',
        'ResNet50':    '1m8NV9yrHdsVoOq51VlkQ3yYJLY7odXf4',
        'InceptionV3': '1JDGfbMxODBPVlSeL7qxjafrqzEuui2Ip'
    }

    def download_from_gdrive(file_id, dest):
        session = requests.Session()
        url = "https://drive.google.com/uc?export=download&id=" + file_id
        response = session.get(url, stream=True)
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        if token:
            url = ("https://drive.google.com/uc?export=download&confirm="
                   + token + "&id=" + file_id)
            response = session.get(url, stream=True)
        if response.status_code != 200 or 'text/html' in response.headers.get('Content-Type', ''):
            url = ("https://drive.google.com/uc?export=download&confirm=t&id=" + file_id)
            response = session.get(url, stream=True)
        total = 0
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
        return total

    for name, file_id in models_info.items():
        path = 'models/' + name + '.weights.h5'
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            try:
                with open(path, 'rb') as f:
                    header = f.read(8)
                is_valid_hdf5 = header.startswith(b'\x89HDF\r\n\x1a\n')
                if is_valid_hdf5 and size_mb > 40:
                    st.sidebar.success(name + ': ' + str(round(size_mb, 1)) + ' MB ready')
                    continue
                else:
                    st.sidebar.warning(name + ' file invalid – re-downloading')
                    os.remove(path)
            except Exception:
                st.sidebar.warning(name + ' file check failed – re-downloading')
                if os.path.exists(path):
                    os.remove(path)

        msg = st.empty()
        msg.info('Downloading ' + name + ' model… please wait')
        try:
            total_bytes = download_from_gdrive(file_id, path)
            size_mb = total_bytes / (1024 * 1024)
            with open(path, 'rb') as f:
                header = f.read(8)
            is_valid = header.startswith(b'\x89HDF\r\n\x1a\n')
            if is_valid and size_mb > 40:
                msg.success(name + ' downloaded! (' + str(round(size_mb, 1)) + ' MB)')
            else:
                msg.error(name + ' download failed – invalid file')
                if os.path.exists(path):
                    os.remove(path)
        except Exception as e:
            msg.error(name + ' download error: ' + str(e)[:100])


# ─────────────────────────────────────────────
# BUILD MODEL ARCHITECTURE + LOAD WEIGHTS
# ─────────────────────────────────────────────

def build_and_load(model_name, weights_path):
    """
    Re-create the exact same architecture that was used during training,
    then load the saved weights into it.
    """
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3

    num_classes = len(CLASS_NAMES)  # 5

    if model_name == 'VGG16':
        base = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        img_size = (224, 224)
    elif model_name == 'ResNet50':
        base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        img_size = (224, 224)
    elif model_name == 'InceptionV3':
        base = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
        img_size = (299, 299)
    else:
        raise ValueError('Unknown model: ' + model_name)

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.load_weights(weights_path)
    return model


# ─────────────────────────────────────────────
# LOAD ALL THREE MODELS  (cached so Streamlit
# only builds/loads them once per session)
# ─────────────────────────────────────────────

@st.cache_resource
def load_all_models():
    models = {}
    for name in MODEL_NAMES:
        weights_path = 'models/' + name + '.weights.h5'
        if not os.path.exists(weights_path):
            st.sidebar.error('❌ ' + name + ' weight file not found')
            continue
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        st.sidebar.info(name + ' file: ' + str(round(size_mb, 1)) + ' MB')
        try:
            models[name] = build_and_load(name, weights_path)
            st.sidebar.success('✅ ' + name + ' loaded')
        except Exception as e:
            st.sidebar.error('❌ ' + name + ' failed: ' + str(e)[:200])
            # Remove corrupt weight file so it re-downloads next run
            try:
                os.remove(weights_path)
                st.sidebar.warning(name + ' removed – will re-download next run')
            except Exception:
                pass
    st.sidebar.info('Loaded models: ' + str(list(models.keys())))
    return models


# ─────────────────────────────────────────────
# PRE-PROCESSING HELPERS
# ─────────────────────────────────────────────

@st.cache_resource
def get_preprocess_fn(model_name):
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
    from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
    from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre
    fn_map   = {'VGG16': vgg_pre,      'ResNet50': res_pre,  'InceptionV3': inc_pre}
    size_map = {'VGG16': (224, 224),   'ResNet50': (224, 224), 'InceptionV3': (299, 299)}
    return fn_map[model_name], size_map[model_name]


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_image(image_pil, model_choice, trained_models):
    if not trained_models:
        st.error("❌ No models loaded properly. Check the sidebar for errors and restart the app.")
        st.stop()

    all_probs = []
    models_to_use = MODEL_NAMES if model_choice == 'Ensemble' else [model_choice]

    for model_name in models_to_use:
        if model_name not in trained_models:
            st.warning(f"{model_name} not available – skipping")
            continue
        try:
            model = trained_models[model_name]
            preprocess_fn, img_size = get_preprocess_fn(model_name)
            img = np.array(image_pil.convert('RGB'))
            img = cv2.resize(img, img_size)
            img_pre = preprocess_fn(img.astype(np.float32))
            img_exp = np.expand_dims(img_pre, 0)
            probs = model.predict(img_exp, verbose=0)
            probs = np.squeeze(probs).flatten()
            all_probs.append(probs.astype(np.float32))
        except Exception as e:
            st.warning(f"{model_name} prediction failed: {str(e)[:80]}")

    if not all_probs:
        st.error("All predictions failed!")
        return None

    final_probs = np.mean(all_probs, axis=0)
    pred_idx   = int(np.argmax(final_probs))
    cls_name   = CLASS_NAMES[pred_idx]
    confidence = float(final_probs[pred_idx]) * 100

    return {
        'class_name': cls_name,
        'class_disp': CLASS_DISPLAY[pred_idx],
        'stage':      STAGE_MAP[cls_name],
        'risk':       RISK_MAP[cls_name][0],
        'risk_class': RISK_MAP[cls_name][1],
        'risk_color': RISK_MAP[cls_name][2],
        'confidence': confidence,
        'all_probs':  final_probs,
        'pred_idx':   pred_idx
    }


# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

def generate_gradcam(image_pil, model_name, trained_models, pred_idx):
    if model_name not in trained_models:
        return None
    try:
        from tensorflow.keras.models import Model as KModel
        model = trained_models[model_name]
        preprocess_fn, img_size = get_preprocess_fn(model_name)
        last_conv = LAST_CONV_LAYERS[model_name]

        img     = np.array(image_pil.convert('RGB'))
        img_res = cv2.resize(img, img_size)
        img_pre = preprocess_fn(img_res.astype(np.float32))
        img_exp = np.expand_dims(img_pre, 0)

        grad_model = KModel(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_exp)
            class_ch = preds[:, pred_idx]
        grads       = tape.gradient(class_ch, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out    = conv_out[0]
        heatmap     = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap     = tf.squeeze(heatmap)
        heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap     = heatmap.numpy()

        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(cv2.resize(heatmap_uint8, img_size), cv2.COLORMAP_JET)
        heatmap_rgb   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed  = cv2.addWeighted(img_res, 0.6, heatmap_rgb, 0.4, 0)
        return img_res, heatmap, superimposed
    except Exception:
        return None


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────

def generate_pdf_report(patient_info, prediction, gradcam_img, model_choice):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Title'], fontSize=22,
                                 fontName='Helvetica-Bold',
                                 textColor=colors.HexColor('#00d4aa'),
                                 alignment=TA_CENTER, spaceAfter=6)
    sub_style   = ParagraphStyle('S', parent=styles['Normal'], fontSize=10,
                                 textColor=colors.HexColor('#8892a4'),
                                 alignment=TA_CENTER, spaceAfter=4)
    sec_style   = ParagraphStyle('H', parent=styles['Heading2'], fontSize=13,
                                 fontName='Helvetica-Bold',
                                 textColor=colors.HexColor('#00d4aa'),
                                 spaceBefore=12, spaceAfter=6)
    norm_style  = ParagraphStyle('N', parent=styles['Normal'], fontSize=10,
                                 textColor=colors.HexColor('#2c3e50'), spaceAfter=4)
    disc_style  = ParagraphStyle('D', parent=styles['Normal'], fontSize=8,
                                 textColor=colors.HexColor('#999999'),
                                 alignment=TA_CENTER)
    story = []
    story.append(Paragraph("CervAI Medical Report", title_style))
    story.append(Paragraph("AI-Assisted Cervical Cancer Detection System", sub_style))
    story.append(Paragraph("Report Generated: " + datetime.now().strftime('%B %d, %Y at %I:%M %p'), sub_style))
    story.append(HRFlowable(width="100%", thickness=2,
                            color=colors.HexColor('#00d4aa'), spaceAfter=12))
    story.append(Paragraph("Patient Information", sec_style))
    pt = Table([
        ['Field', 'Details'],
        ['Patient Name', patient_info['name']],
        ['Age',          str(patient_info['age']) + ' years'],
        ['Height',       str(patient_info['height']) + ' cm'],
        ['Weight',       str(patient_info['weight']) + ' kg'],
        ['BMI',          str(round(patient_info['bmi'], 1))],
        ['WhatsApp',     patient_info['whatsapp']],
        ['Email',        patient_info['email']],
        ['Report Date',  datetime.now().strftime('%d/%m/%Y')],
        ['Report ID',    'CERVAI-' + datetime.now().strftime('%Y%m%d%H%M%S')]
    ], colWidths=[5*cm, 11*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1,  0), colors.HexColor('#00d4aa')),
        ('TEXTCOLOR',    (0, 0), (-1,  0), colors.white),
        ('FONTNAME',     (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTNAME',     (0, 1), ( 0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 10),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8fffe'), colors.HexColor('#e8fdf9')]),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#00d4aa')),
        ('PADDING',      (0, 0), (-1, -1), 8)
    ]))
    story.append(pt)
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#e0e0e0'), spaceAfter=8))
    story.append(Paragraph("AI Prediction Results", sec_style))
    risk_colors_map = {
        'LOW RISK': '#00d4aa', 'MILD RISK': '#ffa733',
        'MODERATE RISK': '#ff8c00', 'HIGH RISK': '#ff4b6e', 'CRITICAL RISK': '#ff0000'
    }
    risk_hex = risk_colors_map.get(prediction['risk'], '#00d4aa')
    rt = Table([
        ['Parameter', 'Result'],
        ['Cell Type',      prediction['class_disp']],
        ['Cancer Stage',   prediction['stage']],
        ['Risk Level',     prediction['risk']],
        ['Confidence',     str(round(prediction['confidence'], 2)) + '%'],
        ['Model Used',     model_choice],
        ['Model Accuracy', str(ACCURACY_MAP.get(model_choice, 0)) + '%']
    ], colWidths=[5*cm, 11*cm])
    rt.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1,  0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR',    (0, 0), (-1,  0), colors.white),
        ('FONTNAME',     (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTNAME',     (0, 1), ( 0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 10),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8fffe'), colors.HexColor('#e8fdf9')]),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('PADDING',      (0, 0), (-1, -1), 8),
        ('TEXTCOLOR',    (1, 3), ( 1,  3), colors.HexColor(risk_hex)),
        ('FONTNAME',     (1, 3), ( 1,  3), 'Helvetica-Bold')
    ]))
    story.append(rt)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Class Probability Breakdown", sec_style))
    pb = [['Cell Class', 'Probability', 'Stage']]
    for i, (cn, cd) in enumerate(zip(CLASS_NAMES, CLASS_DISPLAY)):
        prob   = prediction['all_probs'][i] * 100
        stage  = STAGE_MAP[cn].split(' - ')[0]
        marker = ' (PREDICTED)' if i == prediction['pred_idx'] else ''
        pb.append([cd + marker, str(round(prob, 2)) + '%', stage])
    pbt = Table(pb, colWidths=[6*cm, 4*cm, 6*cm])
    pbt.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('FONTNAME',     (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',     (0, 0), (-1, -1), 9),
        ('ALIGN',        (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN',        (1, 0), ( 1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ('GRID',         (0, 0), (-1, -1), 0.3, colors.HexColor('#dddddd')),
        ('PADDING',      (0, 0), (-1, -1), 6),
        ('FONTNAME',     (0, prediction['pred_idx']+1), (-1, prediction['pred_idx']+1), 'Helvetica-Bold'),
        ('BACKGROUND',   (0, prediction['pred_idx']+1), (-1, prediction['pred_idx']+1),
         colors.HexColor('#e8fdf9'))
    ]))
    story.append(pbt)
    story.append(Spacer(1, 12))
    if gradcam_img is not None:
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor('#e0e0e0'), spaceAfter=8))
        story.append(Paragraph("Grad-CAM Explainability Analysis", sec_style))
        story.append(Paragraph(
            "Red/warm areas indicate high activation zones influencing the AI prediction.",
            norm_style))
        story.append(Spacer(1, 8))
        orig_buf = io.BytesIO()
        Image.fromarray(gradcam_img[0]).save(orig_buf, 'PNG')
        orig_buf.seek(0)
        heat_buf = io.BytesIO()
        hc = cv2.applyColorMap(np.uint8(255 * gradcam_img[1]), cv2.COLORMAP_JET)
        Image.fromarray(cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)).save(heat_buf, 'PNG')
        heat_buf.seek(0)
        over_buf = io.BytesIO()
        Image.fromarray(gradcam_img[2]).save(over_buf, 'PNG')
        over_buf.seek(0)
        it = Table([[
            RLImage(orig_buf, 5*cm, 5*cm),
            RLImage(heat_buf, 5*cm, 5*cm),
            RLImage(over_buf, 5*cm, 5*cm)
        ], [
            Paragraph("Original",  norm_style),
            Paragraph("Heatmap",   norm_style),
            Paragraph("Overlay",   norm_style)
        ]])
        it.setStyle(TableStyle([
            ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
            ('PADDING', (0, 0), (-1, -1), 5)
        ]))
        story.append(it)
        story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#e0e0e0'), spaceAfter=8))
    story.append(Paragraph("Clinical Recommendations", sec_style))
    recs = {
        'LOW RISK':      "No immediate action required. Continue routine annual Pap smear screening.",
        'MILD RISK':     "Mild cellular changes detected. Follow-up colposcopy recommended within 6 months.",
        'MODERATE RISK': "Moderate dysplasia detected. Immediate colposcopy and biopsy recommended within 2-4 weeks.",
        'HIGH RISK':     "Severe dysplasia detected. Urgent specialist referral required. Consult immediately.",
        'CRITICAL RISK': "Critical findings detected. IMMEDIATE specialist consultation required."
    }
    story.append(Paragraph(recs.get(prediction['risk'], ''), norm_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor('#e0e0e0'), spaceAfter=8))
    story.append(Paragraph(
        "DISCLAIMER: This AI report is for screening only and does not constitute a medical diagnosis. "
        "Please consult a qualified medical professional.", disc_style))
    story.append(Paragraph(
        "CervAI Detection System | Powered by Deep Learning | Report ID: CERVAI-"
        + datetime.now().strftime('%Y%m%d%H%M%S'), disc_style))
    doc.build(story)
    buffer.seek(0)
    return buffer


# ─────────────────────────────────────────────
# EMAIL / WHATSAPP
# ─────────────────────────────────────────────

def send_email(receiver_email, patient_name, prediction, pdf_buffer):
    try:
        sender_email    = st.secrets.get("EMAIL_ADDRESS", "")
        sender_password = st.secrets.get("EMAIL_PASSWORD", "")
        if not sender_email or not sender_password:
            return False, "Email credentials not configured in st.secrets"
        msg = MIMEMultipart()
        msg['From']    = sender_email
        msg['To']      = receiver_email
        msg['Subject'] = "CervAI Medical Report - " + patient_name
        body = (
            "Dear " + patient_name + ",\n\n"
            "Your cervical cancer screening report is ready.\n\n"
            "RESULTS SUMMARY:\n"
            "- Cell Type    : " + prediction['class_disp'] + "\n"
            "- Cancer Stage : " + prediction['stage'] + "\n"
            "- Risk Level   : " + prediction['risk'] + "\n"
            "- Confidence   : " + str(round(prediction['confidence'], 2)) + "%\n\n"
            "Please find your detailed PDF report attached.\n\n"
            "IMPORTANT: This is an AI-assisted screening report.\n"
            "Please consult your doctor for proper diagnosis.\n\n"
            "Regards,\nCervAI Detection System"
        )
        msg.attach(MIMEText(body, 'plain'))
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_buffer.getvalue())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment; filename="CervAI_Report_' + patient_name + '.pdf"')
        msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, str(e)


def get_recommendation(risk):
    recs = {
        'LOW RISK':      "No immediate action required. Continue routine annual Pap smear screening.",
        'MILD RISK':     "Mild cellular changes. Follow-up colposcopy recommended within 6 months.",
        'MODERATE RISK': "Moderate dysplasia. Immediate colposcopy and biopsy recommended.",
        'HIGH RISK':     "Severe dysplasia. Urgent specialist referral required. Consult immediately.",
        'CRITICAL RISK': "Critical findings. IMMEDIATE specialist consultation required."
    }
    return recs.get(risk, '')


def send_whatsapp(whatsapp_number, patient_name, prediction):
    try:
        import urllib.parse
        probs = prediction['all_probs']
        msg_text = (
            "CervAI Medical Report\n\n"
            "Dear " + patient_name + ",\n\n"
            "Your cervical cancer screening report is ready.\n\n"
            "RESULTS SUMMARY:\n"
            "Patient Name   : " + patient_name + "\n"
            "Cell Type      : " + prediction['class_disp'] + "\n"
            "Cancer Stage   : " + prediction['stage'] + "\n"
            "Risk Level     : " + prediction['risk'] + "\n"
            "Confidence     : " + str(round(prediction['confidence'], 2)) + "%\n\n"
            "CLASS PROBABILITIES:\n"
            "Dyskeratotic             : " + str(round(float(probs[0])*100, 2)) + "%\n"
            "Koilocytotic             : " + str(round(float(probs[1])*100, 2)) + "%\n"
            "Metaplastic              : " + str(round(float(probs[2])*100, 2)) + "%\n"
            "Parabasal                : " + str(round(float(probs[3])*100, 2)) + "%\n"
            "Superficial-Intermediate : " + str(round(float(probs[4])*100, 2)) + "%\n\n"
            "RECOMMENDATION:\n" + get_recommendation(prediction['risk']) + "\n\n"
            "Report Date : " + datetime.now().strftime('%d/%m/%Y %I:%M %p') + "\n"
            "Report ID   : CERVAI-" + datetime.now().strftime('%Y%m%d%H%M%S') + "\n\n"
            "IMPORTANT: AI-assisted screening. Please consult your doctor.\n\n"
            "Regards,\nCervAI Detection System"
        )
        phone   = whatsapp_number.strip().replace('+', '').replace(' ', '').replace('-', '')
        wa_link = "https://wa.me/" + phone + "?text=" + urllib.parse.quote(msg_text)
        return True, wa_link
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0;'>"
        "<div style='font-size:3rem;'>🔬</div>"
        "<div style='font-size:1.3rem;font-weight:700;color:#00d4aa;letter-spacing:2px;'>CervAI</div>"
        "<div style='font-size:0.75rem;color:#8892a4;letter-spacing:1px;'>DETECTION SYSTEM</div>"
        "</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='color:#8892a4;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;'>Model Selection</div>",
                unsafe_allow_html=True)
    model_choice = st.radio(
        "Model",
        ['Ensemble (Recommended)', 'VGG16', 'ResNet50', 'InceptionV3'],
        label_visibility='collapsed'
    )
    model_key = model_choice.replace(' (Recommended)', '')
    st.markdown("---")
    st.markdown("<div style='color:#8892a4;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>Model Performance</div>",
                unsafe_allow_html=True)
    for m, acc in ACCURACY_MAP.items():
        color = '#00d4aa' if m == model_key else '#8892a4'
        bold  = 'bold'    if m == model_key else 'normal'
        st.markdown(
            "<div style='display:flex;justify-content:space-between;padding:0.3rem 0;"
            "color:" + color + ";font-weight:" + bold + ";'>"
            "<span>" + m + "</span><span>" + str(acc) + "%</span></div>",
            unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "<div style='color:#8892a4;font-size:0.75rem;text-align:center;line-height:1.6;'>"
        "AI-powered cervical cancer<br>detection using Transfer Learning<br>"
        "VGG16 + ResNet50 + InceptionV3<br><br>"
        "<span style='color:#00d4aa;'>v1.0.0</span></div>",
        unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BOOT-SEQUENCE  (download → load)
# ─────────────────────────────────────────────

download_models_if_needed()
trained_models = load_all_models()


# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center;padding:3rem 2rem 2rem;'>"
    "<div class='hero-badge'><span class='status-dot'></span>AI System Online</div>"
    "<div class='hero-title'>CervAI</div>"
    "<div class='hero-subtitle'>Cervical Cancer Detection and Stage Prediction</div>"
    "</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
for col, val, label, color in [
    (col1, "96.88%", "Best Accuracy",   "#00d4aa"),
    (col2, "0.9975",  "Best AUC Score",  "#00a8ff"),
    (col3, "5",       "Cell Classes",    "#ffa733"),
    (col4, "3",       "CNN Models",      "#ff4b6e")
]:
    with col:
        st.markdown(
            "<div class='result-card fade-in'>"
            "<div class='result-value' style='color:" + color + ";'>" + val + "</div>"
            "<div class='result-label'>" + label + "</div>"
            "</div>", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔬 Detection", "📊 Model Info", "📖 About"])

# ── Tab 1: Detection ──────────────────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-header'>👤 Patient Information</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            patient_name = st.text_input("Full Name", placeholder="John Doe")
        with c2:
            patient_age = st.number_input("Age", min_value=1, max_value=120, value=30)
        c3, c4 = st.columns(2)
        with c3:
            patient_height = st.number_input("Height (cm)", min_value=50, max_value=250, value=160)
        with c4:
            patient_weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=60)
        bmi = patient_weight / ((patient_height / 100) ** 2)
        st.markdown(
            "<div style='color:#00d4aa;font-size:0.9rem;margin:-0.5rem 0 0.5rem;'>"
            "BMI: <strong>" + str(round(bmi, 1)) + "</strong></div>",
            unsafe_allow_html=True)
        c5, c6 = st.columns(2)
        with c5:
            patient_whatsapp = st.text_input("WhatsApp Number", placeholder="919876543210")
        with c6:
            patient_email = st.text_input("Email Address", placeholder="patient@email.com")
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🖼 Upload Pap Smear Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload", type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility='collapsed')
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded Image",
                     use_container_width=True)
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📤 Report Delivery</div>", unsafe_allow_html=True)
        delivery    = st.multiselect("Send Via", ['Email', 'WhatsApp'],
                                     default=['Email'], label_visibility='collapsed')
        analyze_btn = st.button("🔬 ANALYZE IMAGE", use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>📊 Analysis Results</div>", unsafe_allow_html=True)

        if analyze_btn and uploaded_file:
            if not patient_name:
                st.error("Please enter patient name!")
            else:
                with st.spinner("Analyzing image…"):
                    image_pil  = Image.open(uploaded_file)
                    prediction = predict_image(image_pil, model_key, trained_models)
                    gradcam    = None
                    if prediction is not None:
                        gc_model = model_key if model_key != 'Ensemble' else 'VGG16'
                        gradcam  = generate_gradcam(image_pil, gc_model,
                                                    trained_models, prediction['pred_idx'])

                if prediction is None:
                    st.error("Prediction failed. Please try again.")
                    st.stop()

                # Result card
                st.markdown(
                    "<div class='glass-card fade-in'>"
                    "<div style='text-align:center;margin-bottom:1rem;'>"
                    "<div style='font-size:0.85rem;color:#8892a4;text-transform:uppercase;letter-spacing:2px;'>Predicted Cell Type</div>"
                    "<div style='font-size:2rem;font-weight:900;color:#00d4aa;margin:0.3rem 0;'>" + prediction['class_disp'] + "</div>"
                    "<div style='font-size:0.9rem;color:#8892a4;'>" + prediction['stage'] + "</div>"
                    "</div>"
                    "<div style='text-align:center;margin:1rem 0;'>"
                    "<span class='risk-badge " + prediction['risk_class'] + "'>" + prediction['risk'] + "</span>"
                    "</div>"
                    "<div style='text-align:center;'>"
                    "<div style='font-size:0.8rem;color:#8892a4;text-transform:uppercase;letter-spacing:1px;'>Confidence</div>"
                    "<div style='font-size:2.5rem;font-weight:900;color:" + prediction['risk_color'] + ";font-family:JetBrains Mono,monospace;'>"
                    + str(round(prediction['confidence'], 1)) + "%</div>"
                    "</div></div>", unsafe_allow_html=True)

                # Probability bars
                st.markdown("<div style='color:#8892a4;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin:1rem 0 0.5rem;'>Class Probabilities</div>",
                            unsafe_allow_html=True)
                for i, (cls_d, prob) in enumerate(zip(CLASS_DISPLAY, prediction['all_probs'])):
                    pct   = float(prob) * 100
                    color = '#00d4aa' if i == prediction['pred_idx'] else '#8892a4'
                    fw    = '700'     if i == prediction['pred_idx'] else '400'
                    bg    = ('linear-gradient(90deg,#00d4aa,#00a8ff)'
                             if i == prediction['pred_idx'] else 'rgba(136,146,164,0.3)')
                    st.markdown(
                        "<div style='margin:0.4rem 0;'>"
                        "<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
                        "<span style='font-size:0.85rem;color:" + color + ";font-weight:" + fw + ";'>" + cls_d + "</span>"
                        "<span style='font-size:0.85rem;color:" + color + ";font-family:JetBrains Mono,monospace;'>" + str(round(pct, 1)) + "%</span>"
                        "</div>"
                        "<div class='prob-bar-container'>"
                        "<div class='prob-bar' style='width:" + str(pct) + "%;background:" + bg + ";'></div>"
                        "</div></div>", unsafe_allow_html=True)

                # Grad-CAM
                if gradcam:
                    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                    st.markdown("<div style='color:#8892a4;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.5rem;'>Grad-CAM Explainability</div>",
                                unsafe_allow_html=True)
                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        st.image(gradcam[0], caption="Original",  use_container_width=True)
                    with gc2:
                        hv = cv2.applyColorMap(np.uint8(255 * gradcam[1]), cv2.COLORMAP_JET)
                        st.image(cv2.cvtColor(hv, cv2.COLOR_BGR2RGB), caption="Heatmap",
                                 use_container_width=True)
                    with gc3:
                        st.image(gradcam[2], caption="Overlay", use_container_width=True)

                # PDF
                patient_info = {
                    'name': patient_name, 'age': patient_age,
                    'height': patient_height, 'weight': patient_weight,
                    'bmi': bmi, 'whatsapp': patient_whatsapp, 'email': patient_email
                }
                pdf_buffer = generate_pdf_report(patient_info, prediction, gradcam, model_key)
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 DOWNLOAD PDF REPORT",
                    data=pdf_buffer,
                    file_name="CervAI_Report_" + patient_name + "_" + datetime.now().strftime('%Y%m%d') + ".pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

                # Delivery
                if delivery:
                    st.markdown("<div style='color:#8892a4;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin:0.5rem 0;'>Sending Report…</div>",
                                unsafe_allow_html=True)
                if 'Email' in delivery and patient_email:
                    pdf_buffer.seek(0)
                    success, msg = send_email(patient_email, patient_name, prediction, pdf_buffer)
                    if success:
                        st.success("Email sent to " + patient_email)
                    else:
                        st.warning("Email: " + msg)
                if 'WhatsApp' in delivery and patient_whatsapp:
                    success, wa_link = send_whatsapp(patient_whatsapp, patient_name, prediction)
                    if success:
                        st.markdown(
                            "<a href='" + wa_link + "' target='_blank' style='text-decoration:none;'>"
                            "<div style='background:linear-gradient(135deg,#25D366,#128C7E);color:white;"
                            "padding:1rem 2rem;border-radius:12px;font-size:1rem;font-weight:700;"
                            "cursor:pointer;width:100%;text-align:center;margin:0.5rem 0;"
                            "letter-spacing:1px;box-shadow:0 4px 15px rgba(37,211,102,0.3);'>"
                            "📱 SEND WHATSAPP MESSAGE</div></a>"
                            "<div style='color:#8892a4;font-size:0.8rem;text-align:center;margin-top:0.3rem;'>"
                            "Click above to open WhatsApp with full report</div>",
                            unsafe_allow_html=True)
                    else:
                        st.warning("WhatsApp error: " + wa_link)

        elif analyze_btn and not uploaded_file:
            st.warning("Please upload a Pap smear image first!")
        else:
            st.markdown(
                "<div class='glass-card' style='text-align:center;padding:4rem 2rem;'>"
                "<div style='font-size:4rem;margin-bottom:1rem;'>🔬</div>"
                "<div style='font-size:1.2rem;color:#8892a4;font-weight:300;line-height:1.8;'>"
                "Fill in patient details<br>Upload a Pap smear image<br>Click Analyze to get results"
                "</div></div>", unsafe_allow_html=True)

# ── Tab 2: Model Info ─────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, (name, acc) in zip([m1, m2, m3, m4], ACCURACY_MAP.items()):
        with col:
            color = '#00d4aa' if name == 'Ensemble' else '#00a8ff'
            st.markdown(
                "<div class='glass-card' style='text-align:center;'>"
                "<div style='font-size:0.8rem;color:#8892a4;text-transform:uppercase;letter-spacing:1px;'>" + name + "</div>"
                "<div style='font-size:2.2rem;font-weight:900;color:" + color + ";font-family:JetBrains Mono,monospace;'>" + str(acc) + "%</div>"
                "<div style='font-size:0.75rem;color:#8892a4;'>Accuracy</div>"
                "</div>", unsafe_allow_html=True)
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Cancer Stage Mapping</div>", unsafe_allow_html=True)
    for cls_n, cls_d in zip(CLASS_NAMES, CLASS_DISPLAY):
        stage = STAGE_MAP[cls_n]
        risk  = RISK_MAP[cls_n][0]
        color = RISK_MAP[cls_n][2]
        st.markdown(
            "<div class='glass-card' style='padding:1rem 1.5rem;margin:0.4rem 0;"
            "display:flex;justify-content:space-between;align-items:center;'>"
            "<div><span style='font-weight:700;color:#e0e6f0;'>" + cls_d + "</span>"
            "<span style='color:#8892a4;font-size:0.85rem;margin-left:1rem;'>" + stage + "</span></div>"
            "<span class='risk-badge' style='color:" + color + ";border:1px solid " + color + ";"
            "background:rgba(255,255,255,0.05);padding:0.3rem 1rem;font-size:0.8rem;'>" + risk + "</span>"
            "</div>", unsafe_allow_html=True)

# ── Tab 3: About ──────────────────────────────
with tab3:
    st.markdown(
        "<div class='glass-card fade-in'>"
        "<div class='section-header'>About CervAI</div>"
        "<p style='color:#e0e6f0;line-height:1.8;font-size:1rem;'>"
        "CervAI is an AI-powered cervical cancer detection system that uses deep learning and transfer "
        "learning to analyse Pap smear images. The system classifies cells into 5 categories and predicts "
        "the corresponding cancer stage with high accuracy.</p>"
        "<div class='section-header' style='margin-top:1.5rem;'>Technology Stack</div>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:0.5rem;'>"
        "<div style='background:rgba(0,212,170,0.05);border:1px solid rgba(0,212,170,0.2);"
        "border-radius:12px;padding:1rem;'>"
        "<div style='color:#00d4aa;font-weight:700;margin-bottom:0.5rem;'>Models</div>"
        "<div style='color:#8892a4;font-size:0.9rem;line-height:2;'>"
        "VGG16 (96.88%)<br>ResNet50 (95.39%)<br>InceptionV3 (92.93%)<br>Ensemble (Best)</div></div>"
        "<div style='background:rgba(0,168,255,0.05);border:1px solid rgba(0,168,255,0.2);"
        "border-radius:12px;padding:1rem;'>"
        "<div style='color:#00a8ff;font-weight:700;margin-bottom:0.5rem;'>Features</div>"
        "<div style='color:#8892a4;font-size:0.9rem;line-height:2;'>"
        "Transfer Learning<br>Grad-CAM Explainability<br>PDF Report Generation<br>"
        "WhatsApp + Email Delivery</div></div></div>"
        "<div class='section-header' style='margin-top:1.5rem;'>Disclaimer</div>"
        "<p style='color:#8892a4;font-size:0.9rem;line-height:1.8;'>"
        "This system is intended for screening and research purposes only. It does not replace "
        "professional medical diagnosis. Always consult a qualified healthcare professional.</p>"
        "</div>", unsafe_allow_html=True)