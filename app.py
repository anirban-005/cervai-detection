from email.mime import base

import streamlit as st
import numpy as np
import cv2
import os
import io
import smtplib
import urllib.parse
import warnings
warnings.filterwarnings('ignore')

from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import tensorflow as tf
from tensorflow.keras.applications.vgg16        import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50     import preprocess_input as res_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre
from tensorflow.keras.models import Model

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

EMAIL_ADDRESS  = "anirbanbck2004@gmail.com"
EMAIL_PASSWORD = "irvkbpsgxvqbgjpu"

st.set_page_config(
    page_title="CervAI — Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --teal:#00d4aa; --blue:#0ea5e9; --purple:#8b5cf6;
    --rose:#f43f5e; --amber:#f59e0b; --green:#10b981;
    --bg:#08101e; --card:#0e1a2e; --card2:#122038;
    --text:#e2e8f0; --muted:#64748b; --border:rgba(0,212,170,0.15);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* Animated gradient bg */
.stApp {
    background: linear-gradient(135deg, #06090f 0%, #08101e 40%, #0a1628 100%) !important;
    animation: bgShift 20s ease-in-out infinite alternate;
}
@keyframes bgShift {
    0%   { background-position: 0% 0%; }
    100% { background-position: 100% 100%; }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080e1c 0%, #0a1422 100%) !important;
    border-right: 1px solid rgba(0,212,170,0.15) !important;
}
section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--blue), var(--purple), var(--teal));
    background-size: 300% 100%;
    animation: rainbowMove 4s linear infinite;
    z-index: 10;
}
@keyframes rainbowMove { 0%{background-position:0%} 100%{background-position:300%} }

/* Floating microscope icon */
.brand-icon {
    font-size: 3rem;
    display: block;
    text-align: center;
    animation: iconFloat 3s ease-in-out infinite;
    filter: drop-shadow(0 0 12px rgba(0,212,170,0.7));
}
@keyframes iconFloat {
    0%,100% { transform: translateY(0px); filter: drop-shadow(0 0 12px rgba(0,212,170,0.7)); }
    50%     { transform: translateY(-8px); filter: drop-shadow(0 0 24px rgba(0,212,170,1)); }
}

.brand-name {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #00d4aa, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 0.08em;
    margin: 0.3rem 0 0.1rem;
    animation: nameGlow 3s ease-in-out infinite alternate;
}
@keyframes nameGlow {
    from { filter: drop-shadow(0 0 6px rgba(0,212,170,0.4)); }
    to   { filter: drop-shadow(0 0 18px rgba(0,212,170,0.8)); }
}

.brand-sub {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.72rem;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.online-pill {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.3);
    color: var(--green);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0.8rem 0 1.2rem;
    animation: pillPulse 2s ease-in-out infinite;
}
@keyframes pillPulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.3); }
    50%     { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}
.online-dot {
    width: 7px; height: 7px;
    background: var(--green); border-radius: 50%;
    box-shadow: 0 0 8px rgba(16,185,129,0.8);
    animation: dotBlink 1.5s ease-in-out infinite;
}
@keyframes dotBlink {
    0%,100% { opacity:1; } 50% { opacity:0.3; }
}

.section-label {
    color: var(--muted);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.6rem;
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    border-bottom: 1px solid rgba(0,212,170,0.1);
    padding-bottom: 0.3rem;
}

.perf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0.7rem;
    border-radius: 8px;
    margin: 0.2rem 0;
    border: 1px solid transparent;
    transition: all 0.3s;
    font-size: 0.83rem;
}
.perf-row:hover { background: rgba(0,212,170,0.05); border-color: rgba(0,212,170,0.15); }
.perf-row.best {
    background: rgba(0,212,170,0.08);
    border-color: rgba(0,212,170,0.25);
    animation: bestGlow 3s ease-in-out infinite;
}
@keyframes bestGlow {
    0%,100% { box-shadow: 0 0 0 0 rgba(0,212,170,0.2); }
    50%     { box-shadow: 0 0 15px 2px rgba(0,212,170,0.12); }
}

.credit-section {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(0,212,170,0.1);
}
.credit-title {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    margin-bottom: 0.7rem;
}
.credit-name {
    font-family: 'Playfair Display', serif;
    font-size: 0.78rem;
    color: rgba(0,212,170,0.75);
    text-align: center;
    padding: 0.18rem 0;
    letter-spacing: 0.04em;
    transition: all 0.3s;
}
.credit-name:hover { color: #00ffcc; text-shadow: 0 0 10px rgba(0,212,170,0.5); }

/* Buttons glowing */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0ea5e9) !important;
    color: #06090f !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 0.8rem 2rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.4), 0 0 40px rgba(0,212,170,0.15) !important;
    animation: btnGlow 3s ease-in-out infinite !important;
    font-family: 'Inter', sans-serif !important;
}
@keyframes btnGlow {
    0%,100% { box-shadow: 0 4px 20px rgba(0,212,170,0.4), 0 0 40px rgba(0,212,170,0.15); }
    50%     { box-shadow: 0 4px 30px rgba(0,212,170,0.65), 0 0 60px rgba(0,212,170,0.25); }
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    filter: brightness(1.1) !important;
    animation: none !important;
    box-shadow: 0 8px 40px rgba(0,212,170,0.65) !important;
}

[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #8b5cf6, #0ea5e9) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.5rem !important;
    animation: purpleGlow 3s ease-in-out infinite !important;
}
@keyframes purpleGlow {
    0%,100% { box-shadow: 0 4px 20px rgba(139,92,246,0.35); }
    50%     { box-shadow: 0 4px 35px rgba(139,92,246,0.6); }
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    animation: none !important;
}

/* Hero */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3rem, 8vw, 5.5rem);
    font-weight: 900;
    background: linear-gradient(135deg, #00d4aa 0%, #00ffcc 30%, #0ea5e9 60%, #8b5cf6 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 0.04em;
    line-height: 1;
    animation: heroGrad 5s ease-in-out infinite;
    filter: drop-shadow(0 0 30px rgba(0,212,170,0.3));
}
@keyframes heroGrad {
    0%,100% { background-position: 0% 50%; }
    50%     { background-position: 100% 50%; }
}

.hero-sub {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin: 0.5rem 0 2rem;
}

/* Stat cards */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.4s ease;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--c, #00d4aa), transparent);
    animation: topLine 3s ease-in-out infinite;
}
@keyframes topLine { 0%,100%{opacity:0.5;} 50%{opacity:1;} }
.stat-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.4), 0 0 30px rgba(0,212,170,0.12);
    border-color: rgba(0,212,170,0.3);
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.4rem;
    animation: numGlow 3s ease-in-out infinite alternate;
}
@keyframes numGlow {
    from { filter: drop-shadow(0 0 4px currentColor); }
    to   { filter: drop-shadow(0 0 14px currentColor); }
}
.stat-lbl {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Glass card */
.g-card {
    background: rgba(14,26,46,0.92);
    border: 1px solid rgba(0,212,170,0.12);
    border-radius: 18px;
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(15px);
    transition: all 0.4s ease;
}
.g-card::before {
    content: '';
    position: absolute; top: 0; left: -200%; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4aa, #0ea5e9, transparent);
    animation: shimmer 4s ease-in-out infinite;
}
@keyframes shimmer { 0%{left:-200%;} 100%{left:200%;} }
.g-card:hover {
    border-color: rgba(0,212,170,0.25);
    transform: translateY(-3px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.35), 0 0 25px rgba(0,212,170,0.07);
}
.g-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #00d4aa;
    margin: 0 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.g-card-title::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,212,170,0.4), transparent);
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(6,9,15,0.8) !important;
    border: 1px solid rgba(90,106,130,0.3) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: all 0.3s !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(0,212,170,0.6) !important;
    box-shadow: 0 0 0 3px rgba(0,212,170,0.1), 0 0 20px rgba(0,212,170,0.1) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}
.stRadio label { color: var(--text) !important; }
[data-testid="stFileUploader"] {
    background: rgba(6,9,15,0.6) !important;
    border: 2px dashed rgba(0,212,170,0.2) !important;
    border-radius: 14px !important;
    transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,212,170,0.5) !important;
    background: rgba(0,212,170,0.03) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.3s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,212,170,0.18), rgba(14,165,233,0.12)) !important;
    color: #00d4aa !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    animation: tabGlow 2.5s ease-in-out infinite !important;
}
@keyframes tabGlow {
    0%,100% { box-shadow: 0 0 15px rgba(0,212,170,0.1); }
    50%     { box-shadow: 0 0 25px rgba(0,212,170,0.22); }
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Result display */
.result-box {
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    animation: resultIn 0.6s cubic-bezier(0.4,0,0.2,1);
}
@keyframes resultIn {
    from { opacity:0; transform:scale(0.95) translateY(15px); filter:blur(8px); }
    to   { opacity:1; transform:scale(1) translateY(0); filter:blur(0); }
}
.result-box.normal   { background:rgba(0,212,170,0.08); border:1px solid rgba(0,212,170,0.3); animation:resultIn 0.6s ease, normalGlow 3s 0.6s ease-in-out infinite; }
.result-box.moderate { background:rgba(14,165,233,0.08); border:1px solid rgba(14,165,233,0.3); }
.result-box.high     { background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.3); animation:resultIn 0.6s ease, highGlow 3s 0.6s ease-in-out infinite; }
.result-box.critical { background:rgba(244,63,94,0.08); border:1px solid rgba(244,63,94,0.3); animation:resultIn 0.6s ease, critGlow 3s 0.6s ease-in-out infinite; }
@keyframes normalGlow { 0%,100%{box-shadow:0 0 30px rgba(0,212,170,0.08);} 50%{box-shadow:0 0 50px rgba(0,212,170,0.2);} }
@keyframes highGlow   { 0%,100%{box-shadow:0 0 30px rgba(245,158,11,0.08);} 50%{box-shadow:0 0 50px rgba(245,158,11,0.2);} }
@keyframes critGlow   { 0%,100%{box-shadow:0 0 30px rgba(244,63,94,0.08);} 50%{box-shadow:0 0 50px rgba(244,63,94,0.2);} }

.result-cell {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0.3rem 0;
    animation: cellSlide 0.5s ease-out 0.3s both;
}
@keyframes cellSlide { from{opacity:0;transform:translateX(-20px);} to{opacity:1;transform:translateX(0);} }

.conf-num {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1;
    animation: confPop 0.8s cubic-bezier(0.4,0,0.2,1) 0.5s both;
}
@keyframes confPop { from{opacity:0;transform:scale(0.7);} to{opacity:1;transform:scale(1);} }

/* Prob bars */
.prob-row { margin: 0.5rem 0; }
.prob-meta { display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:4px; }
.prob-track { background:rgba(255,255,255,0.05); border-radius:50px; height:8px; overflow:hidden; }
.prob-fill {
    height:100%; border-radius:50px;
    position:relative; overflow:hidden;
    transition: width 1.5s cubic-bezier(0.4,0,0.2,1);
}
.prob-fill::after {
    content:''; position:absolute; top:0; left:-100%; width:100%; height:100%;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.4),transparent);
    animation:shine 2.5s ease-in-out infinite;
}
@keyframes shine { 0%{left:-100%;} 100%{left:100%;} }

/* Grad-CAM labels */
.gc-lbl { text-align:center; font-family:'Cormorant Garamond',serif; font-style:italic; font-size:0.72rem; color:var(--muted); margin-top:0.3rem; letter-spacing:0.1em; }

/* BMI pill */
.bmi-pill { display:inline-flex; align-items:center; gap:0.5rem; border-radius:50px; padding:0.4rem 1rem; font-size:0.85rem; font-weight:600; margin-top:0.5rem; border:1px solid; animation:bmiGlow 3s ease-in-out infinite; }
@keyframes bmiGlow { 0%,100%{opacity:0.8;} 50%{opacity:1; box-shadow:0 0 12px currentColor;} }

/* WA button */
.wa-btn { display:inline-flex; align-items:center; justify-content:center; gap:0.6rem; background:linear-gradient(135deg,#25d366,#128c7e); color:white !important; text-decoration:none !important; border-radius:12px; padding:0.75rem 1.5rem; font-weight:700; font-size:0.88rem; width:100%; transition:all 0.3s; animation:waGlow 3s ease-in-out infinite; }
@keyframes waGlow { 0%,100%{box-shadow:0 4px 15px rgba(37,211,102,0.3);} 50%{box-shadow:0 4px 30px rgba(37,211,102,0.55);} }
.wa-btn:hover { transform:translateY(-2px); animation:none; box-shadow:0 8px 30px rgba(37,211,102,0.5) !important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(0,212,170,0.3); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
CLASS_NAMES = ['Dyskeratotic','Koilocytotic','Metaplastic','Parabasal','Superficial-Intermediate']
CLASS_INFO = {
    'Dyskeratotic':            {'stage':'Stage 3 - Severe Dysplasia (CIN III)',  'risk':'HIGH RISK',     'rc':'high',     'color':'#f59e0b'},
    'Koilocytotic':            {'stage':'Stage 2 - Moderate Dysplasia (CIN II)', 'risk':'HIGH RISK',     'rc':'high',     'color':'#f59e0b'},
    'Metaplastic':             {'stage':'Stage 1 - Mild Dysplasia (CIN I)',      'risk':'MODERATE RISK', 'rc':'moderate', 'color':'#0ea5e9'},
    'Parabasal':               {'stage':'Stage 4 - Possible Malignancy',         'risk':'CRITICAL RISK', 'rc':'critical', 'color':'#f43f5e'},
    'Superficial-Intermediate':{'stage':'Stage 0 - Normal',                      'risk':'LOW RISK',      'rc':'normal',   'color':'#00d4aa'},
}
CI_FULL = {
    'Dyskeratotic':            {'desc':'Abnormal keratinization associated with HPV infection and precancerous transformation.', 'rec':'Colposcopy and biopsy required within 2 weeks. Refer to gynecologic oncologist.'},
    'Koilocytotic':            {'desc':'Active HPV infection with characteristic perinuclear halos. Strong indicator of CIN 2.', 'rec':'Colposcopy required. HPV genotyping. Repeat Pap smear in 6 months.'},
    'Metaplastic':             {'desc':'Cells at transformation zone — usually benign squamous metaplasia requiring monitoring.', 'rec':'Routine follow-up Pap smear in 12 months. HPV screening recommended.'},
    'Parabasal':               {'desc':'High numbers may indicate severe atrophy or invasive carcinoma. Urgent evaluation needed.', 'rec':'IMMEDIATE specialist consultation. Biopsy and imaging required urgently.'},
    'Superficial-Intermediate':{'desc':'Normal mature squamous epithelium. No abnormality detected. Healthy cervical cytology.', 'rec':'Continue routine annual Pap smear screening. No immediate action required.'},
}
MODEL_PERF     = {'VGG16':96.88,'ResNet50':95.39,'InceptionV3':92.93,'Ensemble':97.20}
PREPROCESS_MAP = {'VGG16':vgg_pre,'ResNet50':res_pre,'InceptionV3':inc_pre}
SIZE_MAP       = {'VGG16':(224,224),'ResNet50':(224,224),'InceptionV3':(299,299)}
LAST_CONV      = {'VGG16':'block5_conv3','ResNet50':'conv5_block3_out','InceptionV3':'mixed10'}
TEAM           = ["Anirban Bhattacharya","Sneha Priya","Aritra Maity","Archisman Das","Surajit Maji"]

# ── Load Models ────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.regularizers import l2
    def build(name, wp):
        if name=='VGG16':    base=VGG16(weights=None,include_top=False,input_shape=(224,224,3),pooling='avg')
        elif name=='ResNet50':base=ResNet50(weights=None,include_top=False,input_shape=(224,224,3),pooling='avg')
        else:                base=InceptionV3(weights=None,include_top=False,input_shape=(299,299,3),pooling='avg')
        x=BatchNormalization()(base.output)
        x=Dense(512,activation='relu',kernel_regularizer=l2(0.001))(x)
        x=Dropout(0.5)(x)
        x=Dense(256,activation='relu',kernel_regularizer=l2(0.001))(x)
        x=Dropout(0.3)(x)
        out=Dense(5,activation='softmax')(x)
        m=Model(inputs=base.input,outputs=out)
        m.load_weights(wp)
        return m
    loaded={}
    for name in ['VGG16','ResNet50','InceptionV3']:
        wp='models/'+name+'.weights.h5'
        if os.path.exists(wp):
            try:
                loaded[name]=build(name,wp)
                st.sidebar.success(name+' loaded')
            except Exception as e:
                st.sidebar.error(name+': '+str(e)[:50])
        else:
            st.sidebar.warning(name+' not found')
    return loaded

def predict_image(image_pil, model_choice, trained_models):
    models_to_use=['VGG16','ResNet50','InceptionV3'] if model_choice=='Ensemble' else [model_choice]
    all_probs=[]
    for mn in models_to_use:
        if mn not in trained_models: continue
        img=np.array(image_pil.convert('RGB'))
        img=cv2.resize(img,SIZE_MAP[mn])
        inp=np.expand_dims(PREPROCESS_MAP[mn](img.astype(np.float32)),0)
        all_probs.append(np.squeeze(trained_models[mn].predict(inp,verbose=0)).astype(np.float32))
    if not all_probs: return None,None
    fp=np.mean(all_probs,axis=0).flatten()
    pi=int(np.argmax(fp))
    return CLASS_NAMES[pi],fp

def generate_gradcam(image_pil, model_name, trained_models, pred_idx):
    if model_name not in trained_models: return None
    try:
        model=trained_models[model_name]
        img=np.array(image_pil.convert('RGB'))
        img_r=cv2.resize(img,SIZE_MAP[model_name])
        inp=np.expand_dims(PREPROCESS_MAP[model_name](img_r.astype(np.float32)),0)
        gm=Model(inputs=model.inputs,outputs=[model.get_layer(LAST_CONV[model_name]).output,model.output])
        with tf.GradientTape() as tape:
            co,preds=gm(inp); cc=preds[:,pred_idx]
        grads=tape.gradient(cc,co); pg=tf.reduce_mean(grads,axis=(0,1,2))
        hm=tf.squeeze(co[0]@pg[...,tf.newaxis])
        hm=tf.maximum(hm,0)/(tf.math.reduce_max(hm)+1e-8); hm=hm.numpy()
        hc=cv2.applyColorMap(cv2.resize(np.uint8(255*hm),SIZE_MAP[model_name]),cv2.COLORMAP_JET)
        hr=cv2.cvtColor(hc,cv2.COLOR_BGR2RGB)
        return img_r,hm,cv2.addWeighted(img_r,0.6,hr,0.4,0),hr
    except: return None

def generate_pdf(name,age,height,weight,bmi,bmi_cat,whatsapp,email_addr,pred_class,probs,model_used,gcr,report_date,report_id):
    W, H = A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=20*mm)

# Light Theme Colors
    BG    = colors.HexColor("#F4F6FB")   # Soft blue-grey background
    CARD  = colors.HexColor("#FFFFFF")   # Pure white cards
    CARD2 = colors.HexColor("#EEF2FF")   # Light lavender secondary card

    TEAL  = colors.HexColor("#F7F9FC")   # Deep blue (primary accent)
    BLUE  = colors.HexColor("#3B82F6")   # Medium blue (secondary accent)

    MUTED = colors.HexColor("#6B7280")   # Neutral grey for muted text
    TEXT  = colors.HexColor("#111827")   # Near-black for body text
    BORD  = colors.HexColor("#CBD5E1")   # Light grey-blue border

    info  = CLASS_INFO[pred_class]
    ACNT  = colors.HexColor(info["color"])

    base = getSampleStyleSheet()
    def ps(n, **k): return ParagraphStyle(n, parent=base["Normal"], **k)
    story = []

    # Header — centered
    hd=Table([[
        Paragraph('<para alignment="center"><b><font size="22" color="#00d4aa">CervAI</font></b></para>',
                  ps("h1",fontSize=22,textColor=TEAL,alignment=TA_CENTER,leading=28)),
    ],[
        Paragraph('<para alignment="center"><font size="10" color="#64748b">Cervical Cancer Diagnostic Report</font></para>',
                  ps("h2",fontSize=10,textColor=MUTED,alignment=TA_CENTER,leading=15)),
    ],[
        Paragraph(f'<para alignment="center"><font size="8" color="#64748b">Report ID: </font><font size="8" color="#00d4aa"><b>{report_id}</b></font><font size="8" color="#64748b">   |   Date: {report_date}</font></para>',
                  ps("h3",fontSize=8,textColor=MUTED,alignment=TA_CENTER,leading=13)),
    ]],colWidths=[174*mm])
    hd.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),BG),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("LINEBELOW",(0,-1),(-1,-1),2,TEAL),
    ]))
    story+=[hd,Spacer(1,12)]

    story.append(Paragraph('PATIENT INFORMATION',ps("s1",fontSize=9,fontName="Helvetica-Bold",textColor=TEAL,spaceAfter=5)))
    rows=[["PATIENT NAME",name or "-","AGE",f"{age} years"],
          ["HEIGHT",f"{height} cm","WEIGHT",f"{weight} kg"],
          ["BMI",f"{bmi:.1f} ({bmi_cat})","WHATSAPP",whatsapp or "-"],
          ["EMAIL",email_addr or "-","MODEL",model_used]]
    ptd=[[Paragraph(r[0],ps(f"l{i}",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED)),
          Paragraph(r[1],ps(f"v{i}",fontSize=9,fontName="Helvetica-Bold",textColor=TEXT)),
          Paragraph(r[2],ps(f"l2{i}",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED)),
          Paragraph(r[3],ps(f"v2{i}",fontSize=9,fontName="Helvetica-Bold",textColor=TEXT))]
         for i,r in enumerate(rows)]
    pt=Table(ptd,colWidths=[30*mm,58*mm,30*mm,56*mm])
    pt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD),("BACKGROUND",(0,0),(0,-1),CARD2),
        ("BACKGROUND",(2,0),(2,-1),CARD2),("GRID",(0,0),(-1,-1),0.5,BORD),
        ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8)]))
    story+=[pt,Spacer(1,12)]

    story.append(Paragraph('AI PREDICTION RESULTS',ps("s2",fontSize=9,fontName="Helvetica-Bold",textColor=TEAL,spaceAfter=5)))
    top_conf=probs[CLASS_NAMES.index(pred_class)]*100; acc=MODEL_PERF.get(model_used.split()[0],97.20)
    rl=Table([[Paragraph("PREDICTED CELL TYPE",ps("rl1",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED))],
              [Paragraph(f'<font color="{info["color"]}"><b>{pred_class}</b></font>',ps("rl2",fontSize=16,fontName="Helvetica-Bold",textColor=ACNT,leading=22))],
              [Paragraph(info["stage"],ps("rl3",fontSize=8,textColor=MUTED))],[Spacer(1,4)],
              [Paragraph(f'<font color="{info["color"]}"><b>{info["risk"]}</b></font>',ps("rl4",fontSize=11,fontName="Helvetica-Bold",textColor=ACNT))]],colWidths=[97*mm])
    rl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD),("LEFTPADDING",(0,0),(-1,-1),14),
        ("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
    rr=Table([[Paragraph("CONFIDENCE",ps("rr1",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED,alignment=TA_CENTER))],
              [Paragraph(f'<font color="{info["color"]}"><b>{top_conf:.1f}%</b></font>',ps("rr2",fontSize=26,fontName="Helvetica-Bold",textColor=ACNT,alignment=TA_CENTER,leading=32))],
              [Paragraph("MODEL ACCURACY",ps("rr3",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED,alignment=TA_CENTER))],
              [Paragraph(f'<b>{acc}%</b>',ps("rr4",fontSize=14,fontName="Helvetica-Bold",textColor=BLUE,alignment=TA_CENTER))],
              [Paragraph(model_used,ps("rr5",fontSize=8,textColor=MUTED,alignment=TA_CENTER))]],colWidths=[77*mm])
    rr.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD2),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
    rw=Table([[rl,rr]],colWidths=[99*mm,79*mm])
    rw.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,BORD),("BACKGROUND",(0,0),(-1,-1),CARD),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("LINEABOVE",(0,0),(-1,0),2,ACNT)]))
    story+=[rw,Spacer(1,12)]

    story.append(Paragraph('CLASS PROBABILITY BREAKDOWN',ps("s3",fontSize=9,fontName="Helvetica-Bold",textColor=TEAL,spaceAfter=5)))
    pb=[["CELL CLASS","PROBABILITY","STAGE","STATUS"]]
    for i,cls in enumerate(CLASS_NAMES):
        pct=probs[i]*100; is_p=cls==pred_class; ci=CLASS_INFO[cls]
        stg=ci["stage"].split("-")[0].strip() if "-" in ci["stage"] else ci["stage"][:20]
        pb.append([Paragraph(f'<b>{cls}</b>'if is_p else cls,ps(f"pn{i}",fontSize=9,fontName="Helvetica-Bold"if is_p else"Helvetica",textColor=colors.HexColor(ci["color"])if is_p else TEXT)),
                   Paragraph(f'<b>{pct:.2f}%</b>'if is_p else f'{pct:.2f}%',ps(f"pp{i}",fontSize=9,fontName="Helvetica-Bold"if is_p else"Helvetica",textColor=colors.HexColor(ci["color"])if is_p else TEXT)),
                   Paragraph(stg,ps(f"ps{i}",fontSize=8,textColor=MUTED)),
                   Paragraph("PREDICTED"if is_p else"-",ps(f"pst{i}",fontSize=7,fontName="Helvetica-Bold"if is_p else"Helvetica",textColor=ACNT if is_p else MUTED))])
    pbt=Table(pb,colWidths=[52*mm,32*mm,64*mm,26*mm])
    pr=CLASS_NAMES.index(pred_class)+1
    pbt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),CARD2),("BACKGROUND",(0,1),(-1,-1),CARD),
        ("GRID",(0,0),(-1,-1),0.5,BORD),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("BACKGROUND",(0,pr),(-1,pr),colors.HexColor("#1e2f50"))]))
    story+=[pbt,Spacer(1,12)]

    if gcr:
        story.append(Paragraph('GRAD-CAM EXPLAINABILITY',ps("s4",fontSize=9,fontName="Helvetica-Bold",textColor=TEAL,spaceAfter=4)))
        story.append(Paragraph("Red/warm areas indicate the cellular regions most influential to the AI prediction.",
                                ps("gd",fontSize=8,textColor=MUTED,spaceAfter=7,leading=13)))
        oa,_,sup,hr=gcr
        ob=io.BytesIO(); Image.fromarray(oa).save(ob,"PNG"); ob.seek(0)
        hb=io.BytesIO(); Image.fromarray(hr).save(hb,"PNG"); hb.seek(0)
        xb=io.BytesIO(); Image.fromarray(sup).save(xb,"PNG"); xb.seek(0)
        sz=52*mm
        gct=Table([[RLImage(ob,sz,sz),RLImage(hb,sz,sz),RLImage(xb,sz,sz)],
                   [Paragraph("ORIGINAL",ps("il1",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED,alignment=TA_CENTER)),
                    Paragraph("GRAD-CAM",ps("il2",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED,alignment=TA_CENTER)),
                    Paragraph("OVERLAY", ps("il3",fontSize=7,fontName="Helvetica-Bold",textColor=MUTED,alignment=TA_CENTER))]],
                  colWidths=[58*mm,58*mm,58*mm])
        gct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD),("GRID",(0,0),(-1,-1),0.5,BORD),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,0),"MIDDLE"),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8)]))
        story+=[gct,Spacer(1,12)]

    rec=CI_FULL[pred_class]["rec"]
    story.append(Paragraph('CLINICAL RECOMMENDATION',ps("s5",fontSize=9,fontName="Helvetica-Bold",textColor=TEAL,spaceAfter=5)))
    rct=Table([[Paragraph(f'<font color="{info["color"]}"><b>{info["risk"]}: </b></font><font color="#e2e8f0">{rec}</font>',
                          ps("rd",fontSize=9,textColor=TEXT,leading=15))]],colWidths=[174*mm])
    rct.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),CARD2),("LINEABOVE",(0,0),(-1,0),2,ACNT),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),14),("RIGHTPADDING",(0,0),(-1,-1),14)]))
    story+=[rct,Spacer(1,12)]

    story.append(HRFlowable(width="100%",thickness=1,color=BORD,spaceAfter=6))
    team_str=" | ".join(TEAM)
    ft=Table([[Paragraph('DISCLAIMER: AI-generated report for screening only. Not a clinical diagnosis. Confirm with a licensed pathologist.',
                         ps("disc",fontSize=7,fontName="Helvetica-Oblique",textColor=MUTED,leading=11)),
               Paragraph(f'CervAI Detection System<br/><font color="#00d4aa">{report_id}</font><br/><font color="#3a4a60">{team_str}</font>',
                         ps("fid",fontSize=6.5,textColor=MUTED,alignment=TA_RIGHT,leading=11))]],
             colWidths=[110*mm,64*mm])
    ft.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0)]))
    story.append(ft)

    def add_bg(canvas,doc):
        canvas.saveState(); canvas.setFillColor(colors.HexColor("#0a0e1a"))
        canvas.rect(0,0,W,H,fill=1,stroke=0); canvas.restoreState()
    doc.build(story,onFirstPage=add_bg,onLaterPages=add_bg)
    buf.seek(0); return buf.read()

def send_email_report(to_email,name,pred_class,pdf_bytes,report_date,report_id):
    try:
        info=CLASS_INFO[pred_class]; rec=CI_FULL[pred_class]["rec"]
        msg=MIMEMultipart()
        msg["From"]=EMAIL_ADDRESS; msg["To"]=to_email
        msg["Subject"]=f"CervAI Diagnostic Report - {name} | {report_date}"
        body=(f"Dear {name},\n\nPlease find attached your cervical cancer screening report from CervAI.\n\n"
              f"REPORT SUMMARY\n{'='*50}\n"
              f"Patient Name   : {name}\nReport ID      : {report_id}\nReport Date    : {report_date}\n{'='*50}\n\n"
              f"DIAGNOSIS RESULTS\nCell Type      : {pred_class}\nCancer Stage   : {info['stage']}\nRisk Level     : {info['risk']}\n\n"
              f"CLINICAL RECOMMENDATION\n{rec}\n\n{'='*50}\n"
              f"IMPORTANT NOTICE\nThis report has been generated by an AI-assisted screening system. "
              f"It does not constitute a formal medical diagnosis. Please consult a qualified pathologist or oncologist.\n\n"
              f"Regards,\nCervAI Detection System\nDeveloped by: {', '.join(TEAM)}\n{'='*50}")
        msg.attach(MIMEText(body,"plain"))
        part=MIMEBase("application","octet-stream"); part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition",f'attachment; filename="CervAI_{name.replace(" ","_")}_{report_id}.pdf"')
        msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(EMAIL_ADDRESS,EMAIL_PASSWORD); s.sendmail(EMAIL_ADDRESS,to_email,msg.as_string())
        return True
    except Exception as e: return str(e)

def build_wa_msg(name,pred_class,probs,bmi,bmi_cat,report_date,report_id):
    info=CLASS_INFO[pred_class]; rec=CI_FULL[pred_class]["rec"]
    tc=probs[CLASS_NAMES.index(pred_class)]*100
    lines=["CervAI Cervical Cancer Screening Report","",
           f"Report ID   : {report_id}","Date        : "+report_date,"",
           "Patient Information",f"Name        : {name}",f"BMI         : {bmi:.1f} ({bmi_cat})","",
           "Diagnosis Summary",f"Cell Type   : {pred_class}",f"Stage       : {info['stage']}",
           f"Risk Level  : {info['risk']}",f"Confidence  : {tc:.1f} percent","",
           "Class Probabilities"]
    for i,cls in enumerate(CLASS_NAMES):
        lines.append(f"{cls:<28}: {probs[i]*100:.1f} percent")
    lines+=["","Clinical Recommendation",rec,"",
            "Important Notice",
            "This report is generated by an AI-assisted screening system. "
            "It does not constitute a formal medical diagnosis. "
            "Please consult a qualified healthcare professional for clinical advice.","",
            "Regards","CervAI Detection System",f"Developed by: {', '.join(TEAM)}"]
    return "\n".join(lines)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <span class="brand-icon">🔬</span>
    <div class="brand-name">CervAI</div>
    <div class="brand-sub">Intelligent Cancer Detection</div>
    <div class="online-pill"><div class="online-dot"></div>AI System Online</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
    model_choice = st.radio("", ["Ensemble (Recommended)","VGG16","ResNet50","InceptionV3"], label_visibility="collapsed")

    st.markdown('<div class="section-label">Model Performance</div>', unsafe_allow_html=True)
    for m,acc in MODEL_PERF.items():
        is_best = m=="Ensemble"
        cls_    = "best" if is_best else ""
        color   = "#00d4aa" if is_best else "#94a3b8"
        st.markdown(f'<div class="perf-row {cls_}"><span style="font-size:0.83rem;color:{"#e2e8f0" if is_best else "#94a3b8"}">{m}</span><span style="font-family:JetBrains Mono,monospace;font-size:0.78rem;font-weight:{"700" if is_best else "400"};color:{color}">{acc}%</span></div>', unsafe_allow_html=True)

    credits_html = "".join([f'<div class="credit-name">{n}</div>' for n in TEAM])
    st.markdown(f'<div class="credit-section"><div class="credit-title">Developed by</div>{credits_html}</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:0.62rem;color:#3a4a60;margin-top:1rem;font-family:JetBrains Mono,monospace;">CervAI v1.0.0 | 2026</div>', unsafe_allow_html=True)

trained_models = load_all_models()
model_key      = model_choice.replace(" (Recommended)","")

tab1, tab2, tab3 = st.tabs(["🔬  Detection","📊  Model Info","ℹ️  About"])

# ── TAB 1 ──────────────────────────────────────────────────────
with tab1:
    st.markdown('<div style="text-align:center;padding:3rem 1rem 2rem">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.88rem;color:#00d4aa;letter-spacing:0.3em;text-transform:uppercase;margin-bottom:0.8rem">Advanced AI Screening System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">CervAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Cervical Cancer Detection &amp; Stage Prediction</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,num,lbl,clr in [(c1,"96.88%","Best Accuracy","#00d4aa"),(c2,"0.9975","Best AUC","#0ea5e9"),(c3,"5","Cell Classes","#f59e0b"),(c4,"3","CNN Models","#f43f5e")]:
        with col:
            st.markdown(f'<div class="stat-card" style="--c:{clr}"><div class="stat-num" style="color:{clr}">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cL, cR = st.columns([1.1,1], gap="large")

    with cL:
        st.markdown('<div class="g-card"><div class="g-card-title">👤 Patient Information</div>', unsafe_allow_html=True)
        a,b=st.columns(2)
        with a: name=st.text_input("Full Name",placeholder="e.g. Priya Sharma")
        with b: age=st.number_input("Age",1,120,30)
        c,d=st.columns(2)
        with c: height=st.number_input("Height (CM)",50.0,250.0,160.0,0.5)
        with d: weight=st.number_input("Weight (KG)",10.0,300.0,60.0,0.5)
        e,f=st.columns(2)
        with e: whatsapp=st.text_input("WhatsApp Number",placeholder="+91 9876543210")
        with f: email=st.text_input("Email Address",placeholder="patient@email.com")
        bmi=weight/((height/100)**2)
        bmi_cat="Underweight" if bmi<18.5 else "Normal" if bmi<25 else "Overweight" if bmi<30 else "Obese"
        bc="#00d4aa" if bmi_cat=="Normal" else ("#f59e0b" if bmi_cat in ["Underweight","Overweight"] else "#f43f5e")
        st.markdown(f'<div class="bmi-pill" style="color:{bc};border-color:{bc}">BMI {bmi:.1f} — {bmi_cat}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="g-card-title">🖼 Upload Pap Smear Image</div>', unsafe_allow_html=True)
        uploaded=st.file_uploader("",type=["jpg","jpeg","png","bmp"],label_visibility="collapsed")
        if uploaded:
            st.image(uploaded,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with cR:
        st.markdown('<div class="g-card"><div class="g-card-title">📊 Analysis Results</div>', unsafe_allow_html=True)
        if "result" not in st.session_state:
            st.markdown('<div style="text-align:center;padding:3.5rem 1rem"><div style="font-size:3.5rem;filter:drop-shadow(0 0 20px rgba(0,212,170,0.5));animation:iconFloat 3s ease-in-out infinite">🔬</div><div style="font-family:Playfair Display,serif;font-size:1.1rem;color:#2a3a50;margin-top:1rem">Awaiting Analysis</div><div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.85rem;color:#1e2e40;margin-top:0.4rem">Upload an image and click Analyse</div></div>', unsafe_allow_html=True)
        else:
            r=st.session_state["result"]
            pc=r["pred_class"]; probs=np.array(r["probs"])
            info=CLASS_INFO[pc]; tc=probs[CLASS_NAMES.index(pc)]*100
            st.markdown(f"""
            <div class="result-box {info['rc']}">
                <div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.7rem;color:#64748b;letter-spacing:0.2em;text-transform:uppercase;margin-bottom:0.3rem">Predicted Cell Type</div>
                <div class="result-cell" style="color:{info['color']}">{pc}</div>
                <div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.88rem;color:#64748b;margin:0.2rem 0 0.8rem">{info['stage']}</div>
                <span style="background:{info['color']}18;border:1px solid {info['color']}50;color:{info['color']};border-radius:50px;padding:0.35rem 1.1rem;font-size:0.75rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase">{info['risk']}</span>
            </div>
            <div style="text-align:center;margin:1rem 0">
                <div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.68rem;color:#5a6a82;letter-spacing:0.2em;text-transform:uppercase">Confidence Score</div>
                <div class="conf-num" style="color:{info['color']}">{tc:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.68rem;color:#5a6a82;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.7rem">Class Probabilities</div>', unsafe_allow_html=True)
            for i,cls in enumerate(CLASS_NAMES):
                pct=probs[i]*100; ci=CLASS_INFO[cls]; is_top=cls==pc
                bg=f"linear-gradient(90deg,{ci['color']},{ci['color']}99)" if is_top else "#1e2f46"
                st.markdown(f'<div class="prob-row"><div class="prob-meta"><span style="color:{"#e2e8f0" if is_top else "#7a8a9a"};font-weight:{"600" if is_top else "400"}">{cls}</span><span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;color:{ci["color"] if is_top else "#5a6a82"};font-weight:{"700" if is_top else "400"}">{pct:.1f}%</span></div><div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;background:{bg}"></div></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if "result" in st.session_state and "grad" in st.session_state:
            g=st.session_state["grad"]
            st.markdown('<div class="g-card"><div class="g-card-title">🔥 Grad-CAM Explainability</div>', unsafe_allow_html=True)
            g1,g2,g3=st.columns(3)
            with g1: st.image(g["orig"],use_container_width=True); st.markdown('<div class="gc-lbl">Original</div>',unsafe_allow_html=True)
            with g2: st.image(g["heat"],use_container_width=True); st.markdown('<div class="gc-lbl">Heatmap</div>',unsafe_allow_html=True)
            with g3: st.image(g["over"],use_container_width=True); st.markdown('<div class="gc-lbl">Overlay</div>',unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if "result" in st.session_state:
            pc_r=st.session_state["result"]["pred_class"]
            rec_r=CI_FULL[pc_r]["rec"]
            st.markdown(f'<div class="g-card"><div class="g-card-title">💊 Clinical Recommendation</div><p style="font-family:Cormorant Garamond,serif;font-style:italic;color:#c8d5e8;font-size:1rem;line-height:1.8;margin:0">{rec_r}</p></div>', unsafe_allow_html=True)

        if "result" in st.session_state:
            r2=st.session_state["result"]
            st.markdown('<div class="g-card"><div class="g-card-title">📤 Download &amp; Share</div>', unsafe_allow_html=True)
            d1,d2=st.columns(2)
            with d1:
                st.download_button("⬇  Download PDF",data=r2["pdf_bytes"],
                    file_name=f"CervAI_{name.replace(' ','_')}_{r2['report_id']}.pdf",
                    mime="application/pdf",use_container_width=True)
            with d2:
                if whatsapp:
                    probs_w=np.array(r2["probs"])
                    wa_msg=build_wa_msg(name,r2["pred_class"],probs_w,bmi,bmi_cat,r2["report_date"],r2["report_id"])
                    num=whatsapp.replace("+","").replace(" ","").replace("-","")
                    st.markdown(f'<a class="wa-btn" href="https://wa.me/{num}?text={urllib.parse.quote(wa_msg)}" target="_blank">💬  Send via WhatsApp</a>',unsafe_allow_html=True)
                else:
                    st.caption("Enter WhatsApp number above to enable sharing.")
            st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        _,bc,_=st.columns([1,2,1])
        with bc:
            if st.button("🔬  Analyse & Generate Report",use_container_width=True,key="analyse"):
                if not name:
                    st.warning("Please enter the patient name.")
                else:
                    image_pil=Image.open(io.BytesIO(uploaded.read()))
                    report_date=datetime.now().strftime("%d/%m/%Y %H:%M")
                    report_id="CERVAI-"+datetime.now().strftime('%Y%m%d%H%M%S')
                    with st.spinner("Running AI analysis..."):
                        pred_class,probs=predict_image(image_pil,model_key,trained_models)
                    if pred_class is None:
                        st.error("Prediction failed. Please check model files."); st.stop()
                    pred_idx=CLASS_NAMES.index(pred_class)
                    with st.spinner("Generating Grad-CAM..."):
                        gcm=model_key if model_key!='Ensemble' else 'VGG16'
                        gcr=generate_gradcam(image_pil,gcm,trained_models,pred_idx)
                    with st.spinner("Building PDF report..."):
                        pdf_bytes=generate_pdf(name,age,height,weight,bmi,bmi_cat,whatsapp,email,pred_class,probs,model_key,gcr,report_date,report_id)
                    if email:
                        with st.spinner(f"Sending report to {email}..."):
                            res=send_email_report(email,name,pred_class,pdf_bytes,report_date,report_id)
                        if res is True: st.success(f"Report emailed successfully to {email}")
                        else: st.error(f"Email failed: {res}")
                    if gcr:
                        oa,_,sup,hr=gcr
                        st.session_state["grad"]={"orig":oa,"heat":hr,"over":sup}
                    st.session_state["result"]={"pred_class":pred_class,"probs":probs.tolist(),"pdf_bytes":pdf_bytes,"report_date":report_date,"report_id":report_id}
                    st.rerun()

    st.markdown('<div style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.16);border-radius:10px;padding:0.8rem 1.2rem;font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.83rem;color:rgba(245,158,11,0.8);margin-top:2rem;line-height:1.6">Medical Disclaimer: CervAI is an AI-assisted screening tool for preliminary use by qualified medical professionals only. It does not constitute a clinical diagnosis. All results must be verified by a licensed pathologist before any clinical decision is made.</div>', unsafe_allow_html=True)

# ── TAB 2 ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="g-card"><div class="g-card-title">📊 Model Performance</div>', unsafe_allow_html=True)
    mc1,mc2,mc3,mc4=st.columns(4)
    for col,(m,acc) in zip([mc1,mc2,mc3,mc4],MODEL_PERF.items()):
        clr="#00d4aa" if m=="Ensemble" else "#0ea5e9"
        best=m=="Ensemble"
        bg_style="rgba(0,212,170,0.07)" if best else "rgba(14,26,46,0.9)"
        bd_style="rgba(0,212,170,0.3)" if best else "rgba(0,212,170,0.1)"
        best_badge='<div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.62rem;color:#00d4aa;letter-spacing:0.15em;margin-bottom:0.3rem">Best Model</div>' if best else ""
        anim="animation:bestGlow 3s ease-in-out infinite;" if best else ""
        with col:
            st.markdown('<div style="background:'+bg_style+';border:1px solid '+bd_style+';border-radius:14px;padding:1.3rem;text-align:center;'+anim+'">'+best_badge+'<div style="font-size:0.7rem;color:#64748b;letter-spacing:0.12em;text-transform:uppercase;font-family:Cormorant Garamond,serif;font-style:italic">'+m+'</div><div style="font-family:Playfair Display,serif;font-size:2rem;font-weight:700;color:'+clr+'">'+str(acc)+'%</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="g-card"><div class="g-card-title">🧫 Cell Class Reference</div>', unsafe_allow_html=True)
    for cls,ci in CLASS_INFO.items():
        rec=CI_FULL[cls]["rec"]; desc=CI_FULL[cls]["desc"]
        st.markdown(f'<div style="display:flex;align-items:flex-start;gap:1rem;padding:1rem 1.2rem;background:rgba(14,26,46,0.8);border-radius:12px;margin-bottom:0.7rem;border-left:3px solid {ci["color"]};transition:all 0.3s"><div style="flex:1"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem"><span style="font-family:Playfair Display,serif;font-weight:700;color:{ci["color"]};font-size:1rem">{cls}</span><span style="font-size:0.72rem;font-weight:700;color:{ci["color"]};background:{ci["color"]}18;border:1px solid {ci["color"]}40;border-radius:50px;padding:0.2rem 0.8rem">{ci["risk"]}</span></div><div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.82rem;color:#64748b;margin-bottom:0.3rem">{ci["stage"]}</div><div style="font-size:0.85rem;color:#8899aa;line-height:1.6">{desc}</div></div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3 ──────────────────────────────────────────────────────
with tab3:
    col_a,col_b=st.columns([1,1],gap="large")
    with col_a:
        st.markdown('<div class="g-card"><div class="g-card-title">🔬 About CervAI</div><p style="font-family:Cormorant Garamond,serif;font-style:italic;color:#c8d5e8;line-height:1.9;font-size:1rem">CervAI is an AI-powered cervical cancer detection system employing deep learning and transfer learning to classify cervical cell images into five diagnostic categories. Built on an ensemble of VGG16, ResNet50 and InceptionV3 trained on 4,049 SIPaKMeD images, achieving 97.20% ensemble accuracy.</p></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="g-card">
            <div class="g-card-title">⚙ Technical Specifications</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem">
        """ + "".join([f'<div style="background:rgba(6,9,15,0.6);border:1px solid rgba(0,212,170,0.1);border-radius:10px;padding:0.9rem"><div style="font-family:Playfair Display,serif;color:{clr};font-weight:700;font-size:0.9rem;margin-bottom:0.4rem">{title}</div><div style="font-family:Cormorant Garamond,serif;font-style:italic;color:#8899aa;font-size:0.85rem;line-height:1.6">{body}</div></div>'
                        for title,body,clr in [
                            ("Dataset","SIPaKMeD — 4,049 cervical cell images across 5 classes","#00d4aa"),
                            ("Architecture","Ensemble of VGG16, ResNet50 and InceptionV3","#0ea5e9"),
                            ("Explainability","Grad-CAM using final convolutional layer activations","#8b5cf6"),
                            ("Performance","97.20% ensemble accuracy, AUC score 0.9975","#f43f5e"),
                        ]]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        team_cards="".join([f'<div style="display:flex;align-items:center;gap:0.8rem;padding:0.9rem 1rem;background:rgba(6,9,15,0.6);border:1px solid rgba(0,212,170,0.1);border-radius:12px;margin-bottom:0.6rem;transition:all 0.3s"><div style="width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,rgba(0,212,170,0.2),rgba(14,165,233,0.2));border:1px solid rgba(0,212,170,0.3);display:flex;align-items:center;justify-content:center;font-size:1rem;flex-shrink:0">&#9888;</div><div><div style="font-family:Playfair Display,serif;font-size:0.92rem;font-weight:700;color:#c8d5e8">{name}</div><div style="font-family:Cormorant Garamond,serif;font-style:italic;font-size:0.72rem;color:#5a6a82">Researcher</div></div></div>' for name in TEAM])
        st.markdown(f'<div class="g-card"><div class="g-card-title">👨‍🔬 Research Team</div>{team_cards}</div>', unsafe_allow_html=True)

        st.markdown('<div class="g-card"><div class="g-card-title">⚠ Medical Disclaimer</div><p style="font-family:Cormorant Garamond,serif;font-style:italic;color:#8899aa;font-size:0.95rem;line-height:1.9;margin:0">CervAI is an AI-assisted screening tool intended exclusively for preliminary use by qualified medical professionals. The system does not constitute a clinical diagnosis and must not be used as a substitute for professional medical judgement. All results must be verified by a licensed pathologist prior to any clinical decision-making.</p></div>', unsafe_allow_html=True)