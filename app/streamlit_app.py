import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from fpdf import FPDF
import base64
import shap
import matplotlib.pyplot as plt

# --- Configuración de la Página ---
st.set_page_config(
    page_title="CDSS Huancayo - Sistema de Soporte al Diagnóstico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilo y Diseño Profesional (Tema Médico) ---
import streamlit as st

def set_custom_style():
    """Inyecta CSS personalizado para un tema médico profesional."""
    st.markdown(
        """
        <style>
            /* Main container and sidebar */
        .reportview-container {{
            background-color: {COLORS['sidebar_bg']}; /* Using sidebar_bg as general background for consistency */
            color: #333333; /* Default text color */
        }}
        .sidebar .sidebar-content {{
            background-color: {COLORS['sidebar_bg']};
        }}

        /* Header */
        .css-18e3th9 {{ /* This class might change, but targets the main header container */
            background-color: {COLORS['header_bg']};
            color: {COLORS['header_text']};
        }}
        .css-1d391kg {{ /* Streamlit header title */
            color: {COLORS['header_text']};
        }}
        h1, h2, h3 {{
            color: {COLORS['header_bg']}; /* Using primary blue for headers */
        }}

        /* Widgets */
        .Widget>label {{
            color: #333333; /* Default text color for labels */
        }}
        .stButton>button {{
            background-color: {COLORS['analyze_button_bg']};
            color: {COLORS['analyze_button_text']};
            box-shadow: {COLORS['analyze_button_shadow']};
            transition: all 0.3s ease;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
        }}
        .stButton>button:hover {{
            background-color: {COLORS['analyze_button_hover_bg']};
            box-shadow: {COLORS['analyze_button_hover_shadow']};
        }}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {{
            color: #333333;
            background-color: #FFFFFF; /* White for input surfaces */
            border: 1px solid #CCCCCC;
            border-radius: 5px;
        }}
        .stRadio>div>label {{
            color: #333333;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
            font-size: 1.2rem;
        }}
        .stTabs [data-baseweb="tab-list"] button {{
            background-color: #E0E0E0; /* Default tab background */
            color: #424242; /* Default tab text */
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
            padding: 10px 15px;
            transition: all 0.3s ease;
        }}
        .stTabs [data-baseweb="tab-list"] button:hover {{
            background-color: #BDBDBD;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: {COLORS['demographics_tab_color']}; /* Example, will be overridden by specific tab styles */
            color: {COLORS['header_text']};
            border-bottom: 3px solid {COLORS['demographics_tab_color']};
        }}

        /* Specific Tab Styles (using data-testid for more precise targeting if needed) */
        /* You might need to inspect Streamlit's generated HTML to get the exact data-testid for each tab */
        /* For now, using a generic approach, may need refinement */
        .stTabs [data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] {{ /* Demographics */
            background-color: {COLORS['demographics_tab_color']};
            border-bottom: 3px solid {COLORS['demographics_tab_color']};
        }}
        .stTabs [data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] {{ /* Vitals */
            background-color: {COLORS['vitals_tab_color']};
            border-bottom: 3px solid {COLORS['vitals_tab_color']};
        }}
        .stTabs [data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] {{ /* Lab */
            background-color: {COLORS['lab_tab_color']};
            border-bottom: 3px solid {COLORS['lab_tab_color']};
        }}
        .stTabs [data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"] {{ /* History */
            background-color: {COLORS['history_tab_color']};
            border-bottom: 3px solid {COLORS['history_tab_color']};
        }}
        .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] {{ /* Symptoms */
            background-color: {COLORS['symptoms_tab_color']};
            border-bottom: 3px solid {COLORS['symptoms_tab_color']};
        }}

        /* Tab Content Backgrounds */
        .st-emotion-cache-1ht1j8p {{ /* This class targets the tab content area */
            background-color: {COLORS['demographics_bg']}; /* Default, will be overridden by specific tab content */
            padding: 20px;
            border-radius: 0 5px 5px 5px;
        }}
        /* Specific tab content backgrounds - these might need more specific targeting if Streamlit doesn't apply them automatically */
        /* For now, we'll rely on the tab selection to change the overall tab content background if possible, or apply directly in Python */

        /* Validation Boxes */
        .stAlert.normal {{
            background-color: {COLORS['normal_bg']};
            color: {COLORS['normal_color']};
            border: {COLORS['normal_border']};
        }}
        .stAlert.warning {{
            background-color: {COLORS['warning_bg']};
            color: {COLORS['warning_color']};
            border: {COLORS['warning_border']};
        }}
        .stAlert.critical {{
            background-color: {COLORS['critical_bg']};
            color: {COLORS['critical_color']};
            border: {COLORS['critical_border']};
        }}

        /* Main Diagnosis Card */
        .main-diagnosis-card {{
            background: {COLORS['main_diagnosis_bg']};
            color: {COLORS['main_diagnosis_text']};
            padding: 20px;
            border-radius: 10px;
            box-shadow: {COLORS['main_diagnosis_shadow']};
            text-align: center;
            margin-bottom: 20px;
        }}

        /* Confidence Levels */
        .confidence-high {{
            color: {COLORS['high_confidence_color']};
            background-color: {COLORS['high_confidence_bg']};
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .confidence-medium {{
            color: {COLORS['medium_confidence_color']};
            background-color: {COLORS['medium_confidence_bg']};
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .confidence-low {{
            color: {COLORS['low_confidence_color']};
            background-color: {COLORS['low_confidence_bg']};
            padding: 5px 10px;
            border-radius: 5px;
        }}
        .stProgress > div > div > div > div {{
            background-color: {COLORS['high_confidence_progress_bar']}; /* Default, will be set dynamically */
        }}

        /* Differential Diagnoses */
        .differential-ira {{
            color: {COLORS['IRA_color']};
            background-color: {COLORS['IRA_bg']};
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .differential-eda {{
            color: {COLORS['EDA_color']};
            background-color: {COLORS['EDA_bg']};
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .differential-hta {{
            color: {COLORS['HTA_color']};
            background-color: {COLORS['HTA_bg']};
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .differential-dm2 {{
            color: {COLORS['DM2_color']};
            background-color: {COLORS['DM2_bg']};
            padding: 5px 10px;
            border-radius: 5px;
            margin: 5px 0;
        }}

        /* Download Button */
        .download-button > button {{
            background-color: {COLORS['download_button_bg']};
            color: {COLORS['download_button_text']};
        }}
        .download-button > button:hover {{
            background-color: {COLORS['download_button_hover_bg']};
        }}

        /* Clear Button */
        .clear-button > button {{
            background-color: {COLORS['clear_button_bg']};
            color: {COLORS['clear_button_text']};
        }}
        .clear-button > button:hover {{
            background-color: {COLORS['clear_button_hover_bg']};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Carga de Recursos ---
@st.cache_resource
def load_resources():
    """ Carga el modelo, scaler y otros recursos necesarios. """
    base_path = Path(__file__).resolve().parent.parent
    model_path = base_path / "models" / "final_model.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"
    X_train_path = base_path / "data" / "processed" / "X_train.csv"
    
    resources = {"model": None, "scaler": None, "explainer": None, "feature_names": None, "error": None}

    try:
        print(f"Cargando modelo desde: {model_path.resolve()}")
        resources["model"] = joblib.load(model_path)
    except FileNotFoundError:
        resources["error"] = f"Error: No se encontró el archivo del modelo en: {model_path}"
    except Exception as e:
        resources["error"] = f"Error al cargar el modelo: {e}"

    try:
        print(f"Cargando scaler desde: {scaler_path.resolve()}")
        resources["scaler"] = joblib.load(scaler_path)
    except FileNotFoundError:
        resources["error"] = f"Error: No se encontró el archivo del scaler en: {scaler_path}"
    except Exception as e:
        resources["error"] = f"Error al cargar el scaler: {e}"
        
    try:
        print(f"Cargando X_train para nombres de características SHAP desde: {X_train_path.resolve()}")
        X_train = pd.read_csv(X_train_path)
        resources["feature_names"] = X_train.columns.tolist()
        # Crear el explainer SHAP
        resources["explainer"] = shap.TreeExplainer(resources["model"].named_steps['classifier'])
    except FileNotFoundError:
        resources["error"] = f"Error: No se encontró el archivo X_train.csv en: {X_train_path}. Necesario para SHAP."
    except Exception as e:
        resources["error"] = f"Error al cargar X_train o crear explainer SHAP: {e}"
        
    return resources

# --- Mapeos y Definiciones ---
DIAGNOSTICO_MAP = {0: 'DM2', 1: 'EDA', 2: 'HTA', 3: 'IRA'}
SEXO_MAP = {'Femenino': 0, 'Masculino': 1}
AREA_MAP = {'Rural': 0, 'Urbano': 1}
SINO_MAP = {'No': 0, 'Sí': 1}

# Define your color palette
COLORS = {
    # Colores principales del módulo
    'header_bg': '#1E88E5',        # Azul médico principal
    'header_text': '#FFFFFF',       # Blanco
    'sidebar_bg': '#F5F7FA',        # Gris muy claro

    # Pestañas del formulario
    'demographics_tab_color': '#1E88E5',
    'demographics_bg': '#E3F2FD',
    'vitals_tab_color': '#43A047',
    'vitals_bg': '#E8F5E9',
    'lab_tab_color': '#9C27B0',
    'lab_bg': '#F3E5F5',
    'history_tab_color': '#FF9800',
    'history_bg': '#FFF3E0',
    'symptoms_tab_color': '#F44336',
    'symptoms_bg': '#FFEBEE',

    # Validaciones y Alertas
    'normal_color': '#4CAF50',
    'normal_bg': '#E8F5E9',
    'normal_border': '2px solid #4CAF50',
    'warning_color': '#FFC107',
    'warning_bg': '#FFF9C4',
    'warning_border': '2px solid #FFC107',
    'critical_color': '#F44336',
    'critical_bg': '#FFCDD2',
    'critical_border': '2px solid #F44336',

    # Resultados de Predicción
    'main_diagnosis_bg': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'main_diagnosis_text': '#FFFFFF',
    'main_diagnosis_shadow': '0 8px 32px rgba(102, 126, 234, 0.4)',

    'high_confidence_color': '#4CAF50',
    'high_confidence_bg': '#E8F5E9',
    'high_confidence_progress_bar': '#4CAF50',
    'medium_confidence_color': '#FF9800',
    'medium_confidence_bg': '#FFF3E0',
    'medium_confidence_progress_bar': '#FF9800',
    'low_confidence_color': '#F44336',
    'low_confidence_bg': '#FFEBEE',
    'low_confidence_progress_bar': '#F44336',

    # Diagnósticos Diferenciales
    'IRA_color': '#2196F3',
    'IRA_bg': '#E3F2FD',
    'EDA_color': '#FF9800',
    'EDA_bg': '#FFF3E0',
    'HTA_color': '#F44336',
    'HTA_bg': '#FFEBEE',
    'DM2_color': '#9C27B0',
    'DM2_bg': '#F3E5F5',

    # Botones
    'analyze_button_bg': '#1E88E5',
    'analyze_button_hover_bg': '#1565C0',
    'analyze_button_text': '#FFFFFF',
    'analyze_button_shadow': '0 4px 12px rgba(30, 136, 229, 0.4)',
    'analyze_button_hover_shadow': '0 6px 20px rgba(30, 136, 229, 0.6)',
    'clear_button_bg': '#757575',
    'clear_button_hover_bg': '#616161',
    'clear_button_text': '#FFFFFF',
    'download_button_bg': '#43A047',
    'download_button_hover_bg': '#2E7D32',
    'download_button_text': '#FFFFFF',

    # Gráficos de Probabilidad
    'probability_IRA': '#2196F3',
    'probability_EDA': '#FF9800',
    'probability_HTA': '#F44336',
    'probability_DM2': '#9C27B0',
}

# Rangos clínicos para validación (ejemplos, se pueden ajustar)
RANGOS_CLINICOS = {
    'pas': (90, 180), 'pad': (60, 120), 'fc': (60, 100), 'fr': (12, 20),
    'temp': (36.0, 38.5), 'spo2': (92, 100), 'glucosa': (70, 180),
    'hba1c': (4.0, 10.0), 'creatinina': (0.6, 1.3), 'colesterol': (125, 240),
    'leucocitos': (4000, 11000)
}

# Lista de síntomas para checkboxes
SINTOMAS_RESPIRATORIOS = ['sintoma_tos', 'sintoma_dificultad_respiratoria', 'sintoma_dolor_garganta', 'sintoma_congestion_nasal', 'sintoma_epistaxis']
SINTOMAS_DIGESTIVOS = ['sintoma_diarrea', 'sintoma_dolor_abdominal', 'sintoma_perdida_apetito', 'sintoma_nauseas', 'sintoma_vomitos']
SINTOMAS_METABOLICOS = ['sintoma_poliuria', 'sintoma_polidipsia', 'sintoma_perdida_peso', 'sintoma_polifagia', 'sintoma_heridas_lentas', 'sintoma_infecciones_frecuentes']
SINTOMAS_CARDIOVASCULARES = ['sintoma_palpitaciones', 'sintoma_dolor_pecho', 'sintoma_tinnitus']
SINTOMAS_GENERALES = ['sintoma_cefalea', 'sintoma_deshidratacion', 'sintoma_vision_borrosa', 'sintoma_fiebre', 'sintoma_escalofrios', 'sintoma_debilidad', 'sintoma_malestar_general', 'sintoma_mareo', 'sintoma_fatiga', 'sintoma_asintomatico']
TODOS_SINTOMAS = SINTOMAS_RESPIRATORIOS + SINTOMAS_DIGESTIVOS + SINTOMAS_METABOLICOS + SINTOMAS_CARDIOVASCULARES + SINTOMAS_GENERALES

DIAGNOSTICO_MAP = {0: 'DM2', 1: 'EDA', 2: 'HTA', 3: 'IRA'}
SEXO_MAP = {'Femenino': 0, 'Masculino': 1}
AREA_MAP = {'Rural': 0, 'Urbano': 1}
SINO_MAP = {'No': 0, 'Sí': 1}

# Define your color palette
COLORS = {
    # Colores principales del módulo
    'header_bg': '#1E88E5',        # Azul médico principal
    'header_text': '#FFFFFF',       # Blanco
    'sidebar_bg': '#F5F7FA',        # Gris muy claro

    # Pestañas del formulario
    'demographics_tab_color': '#1E88E5',
    'demographics_bg': '#E3F2FD',
    'vitals_tab_color': '#43A047',
    'vitals_bg': '#E8F5E9',
    'lab_tab_color': '#9C27B0',
    'lab_bg': '#F3E5F5',
    'history_tab_color': '#FF9800',
    'history_bg': '#FFF3E0',
    'symptoms_tab_color': '#F44336',
    'symptoms_bg': '#FFEBEE',

    # Validaciones y Alertas
    'normal_color': '#4CAF50',
    'normal_bg': '#E8F5E9',
    'normal_border': '2px solid #4CAF50',
    'warning_color': '#FFC107',
    'warning_bg': '#FFF9C4',
    'warning_border': '2px solid #FFC107',
    'critical_color': '#F44336',
    'critical_bg': '#FFCDD2',
    'critical_border': '2px solid #F44336',

    # Resultados de Predicción
    'main_diagnosis_bg': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'main_diagnosis_text': '#FFFFFF',
    'main_diagnosis_shadow': '0 8px 32px rgba(102, 126, 234, 0.4)',

    'high_confidence_color': '#4CAF50',
    'high_confidence_bg': '#E8F5E9',
    'high_confidence_progress_bar': '#4CAF50',
    'medium_confidence_color': '#FF9800',
    'medium_confidence_bg': '#FFF3E0',
    'medium_confidence_progress_bar': '#FF9800',
    'low_confidence_color': '#F44336',
    'low_confidence_bg': '#FFEBEE',
    'low_confidence_progress_bar': '#F44336',

    # Diagnósticos Diferenciales
    'IRA_color': '#2196F3',
    'IRA_bg': '#E3F2FD',
    'EDA_color': '#FF9800',
    'EDA_bg': '#FFF3E0',
    'HTA_color': '#F44336',
    'HTA_bg': '#FFEBEE',
    'DM2_color': '#9C27B0',
    'DM2_bg': '#F3E5F5',

    # Botones
    'analyze_button_bg': '#1E88E5',
    'analyze_button_hover_bg': '#1565C0',
    'analyze_button_text': '#FFFFFF',
    'analyze_button_shadow': '0 4px 12px rgba(30, 136, 229, 0.4)',
    'analyze_button_hover_shadow': '0 6px 20px rgba(30, 136, 229, 0.6)',
    'clear_button_bg': '#757575',
    'clear_button_hover_bg': '#616161',
    'clear_button_text': '#FFFFFF',
    'download_button_bg': '#43A047',
    'download_button_hover_bg': '#2E7D32',
    'download_button_text': '#FFFFFF',

    # Gráficos de Probabilidad
    'probability_IRA': '#2196F3',
    'probability_EDA': '#FF9800',
    'probability_HTA': '#F44336',
    'probability_DM2': '#9C27B0',
}

# Rangos clínicos para validación (ejemplos, se pueden ajustar)
RANGOS_CLINICOS = {
    'pas': (90, 180), 'pad': (60, 120), 'fc': (60, 100), 'fr': (12, 20),
    'temp': (36.0, 38.5), 'spo2': (92, 100), 'glucosa': (70, 180),
    'hba1c': (4.0, 10.0), 'creatinina': (0.6, 1.3), 'colesterol': (125, 240),
    'leucocitos': (4000, 11000)
}

# Lista de síntomas para checkboxes
SINTOMAS_RESPIRATORIOS = ['sintoma_tos', 'sintoma_dificultad_respiratoria', 'sintoma_dolor_garganta', 'sintoma_congestion_nasal', 'sintoma_epistaxis']
SINTOMAS_DIGESTIVOS = ['sintoma_diarrea', 'sintoma_dolor_abdominal', 'sintoma_perdida_apetito', 'sintoma_nauseas', 'sintoma_vomitos']
SINTOMAS_METABOLICOS = ['sintoma_poliuria', 'sintoma_polidipsia', 'sintoma_perdida_peso', 'sintoma_polifagia', 'sintoma_heridas_lentas', 'sintoma_infecciones_frecuentes']
SINTOMAS_CARDIOVASCULARES = ['sintoma_palpitaciones', 'sintoma_dolor_pecho', 'sintoma_tinnitus']
SINTOMAS_GENERALES = ['sintoma_cefalea', 'sintoma_deshidratacion', 'sintoma_vision_borrosa', 'sintoma_fiebre', 'sintoma_escalofrios', 'sintoma_debilidad', 'sintoma_malestar_general', 'sintoma_mareo', 'sintoma_fatiga', 'sintoma_asintomatico']
TODOS_SINTOMAS = SINTOMAS_RESPIRATORIOS + SINTOMAS_DIGESTIVOS + SINTOMAS_METABOLICOS + SINTOMAS_CARDIOVASCULARES + SINTOMAS_GENERALES

# --- Funciones de Interpretación y Recomendación ---

def generate_medical_explanation(shap_values_instance, feature_names, predicted_diagnosis, original_input):
    """
    Genera una explicación en lenguaje médico basada en los valores SHAP.
    """
    explanation = f"Este paciente tiene una alta probabilidad de **{predicted_diagnosis}** debido a los siguientes factores:\n\n"
    
    # Crear un DataFrame para facilitar el manejo
    shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values_instance})
    shap_df['abs_shap'] = np.abs(shap_df['shap_value'])
    
    # Ordenar por la magnitud de la contribución SHAP
    shap_df = shap_df.sort_values(by='abs_shap', ascending=False).head(10) # Top 10 factores

    for _, row in shap_df.iterrows():
        feature = row['feature']
        shap_value = row['shap_value']
        original_value = original_input[feature] if feature in original_input else "N/A" # Get original value

        if shap_value > 0:
            impact = "aumenta"
            sign = "+"
        else:
            impact = "disminuye"
            sign = "" # No need for explicit minus sign as shap_value already has it

        # Traducir nombres de características a lenguaje más amigable
        # Esto es un ejemplo, se puede expandir con un mapeo más completo
        feature_display_name = feature.replace('sintoma_', '').replace('_', ' ').capitalize()
        
        explanation += f"- **{feature_display_name}** (valor: {original_value}) {impact} la probabilidad de {predicted_diagnosis} en {shap_value:.2f} unidades.\n"
        
    return explanation

def get_clinical_recommendations(predicted_diagnosis):
    """
    Proporciona recomendaciones clínicas basadas en el diagnóstico principal.
    """
    recommendations = {
        "IRA": """
        **Recomendaciones para Infección Respiratoria Aguda (IRA):**
        - **Exámenes Complementarios:** Radiografía de tórax (si hay disnea o hipoxemia), hemograma completo, PCR, panel viral respiratorio.
        - **Criterios de Derivación:** Dificultad respiratoria severa, SpO2 < 92%, alteración del estado de conciencia, comorbilidades descompensadas.
        - **Seguimiento:** Control en 24-48 horas, educación sobre signos de alarma, hidratación y reposo.
        """,
        "EDA": """
        **Recomendaciones para Enfermedad Diarreica Aguda (EDA):**
        - **Exámenes Complementarios:** Coprocultivo (si diarrea sanguinolenta o persistente), electrolitos (en casos severos).
        - **Criterios de Derivación:** Deshidratación severa, intolerancia a la vía oral, fiebre alta persistente, signos de abdomen agudo.
        - **Seguimiento:** Rehidratación oral, dieta blanda, control de signos de deshidratación.
        """,
        "HTA": """
        **Recomendaciones para Hipertensión Arterial (HTA):**
        - **Exámenes Complementarios:** Perfil lipídico, glucosa en ayunas y postprandial, creatinina, examen de orina, electrocardiograma.
        - **Criterios de Derivación:** Crisis hipertensiva, daño a órgano blanco (renal, cardíaco, cerebral), HTA secundaria sospechada.
        - **Seguimiento:** Monitoreo ambulatorio de presión arterial, cambios en estilo de vida (dieta, ejercicio), adherencia a medicación.
        """,
        "DM2": """
        **Recomendaciones para Diabetes Mellitus Tipo 2 (DM2):**
        - **Exámenes Complementarios:** HbA1c, glucosa en ayunas y postprandial, perfil lipídico, función renal, examen de orina, fondo de ojo.
        - **Criterios de Derivación:** Cetoacidosis diabética, estado hiperosmolar hiperglucémico, complicaciones micro/macrovasculares.
        - **Seguimiento:** Educación diabetológica, plan de alimentación, ejercicio, control glucémico regular, medicación según necesidad.
        """
    }
    return recommendations.get(predicted_diagnosis, "No hay recomendaciones específicas disponibles para este diagnóstico.")

# --- Funciones de Módulos ---

def display_inicio():
    """ Muestra la página de inicio/bienvenida. """
    st.title("Sistema de Soporte al Diagnóstico Clínico - Huancayo")
    st.markdown("---")
    st.subheader("Bienvenido al sistema de ayuda para el diagnóstico diferencial de IRA, EDA, Hipertensión y Diabetes.")
    st.write(
        "Utilice el menú de la izquierda para navegar entre las diferentes secciones:"
        "\n- **Predicción de Diagnóstico:** Ingrese los datos de un paciente para obtener una predicción."
        "\n- **Análisis de Resultados:** Explore la interpretabilidad del modelo (SHAP)."
        "\n- **Dashboard de Métricas:** Visualice el rendimiento histórico del modelo."
    )

def display_prediccion(resources):
    st.header("Módulo de Predicción de Diagnóstico")
    
    # Columnas para el formulario
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Formulario de Datos del Paciente")
        
        # Pestañas para organizar el formulario
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "👤 Demográficos", "❤️ Signos Vitales", "🔬 Laboratorio", 
            "📜 Antecedentes", "🩺 Síntomas"
        ])

        inputs = {}
        with tab1:
            inputs['edad'] = st.number_input("Edad (años)", min_value=1, max_value=100, value=45)
            inputs['sexo'] = st.selectbox("Sexo", options=list(SEXO_MAP.keys()))
            inputs['area'] = st.selectbox("Área de Residencia", options=list(AREA_MAP.keys()))
            inputs['distrito'] = st.number_input("Distrito (código numérico)", min_value=0, max_value=20, value=5, help="Valor numérico de 0 a 9 según el notebook.")
            inputs['ocupacion'] = st.number_input("Ocupación (código numérico)", min_value=0, max_value=20, value=8, help="Valor numérico de 0 a 14 según el notebook.")
            inputs['tiempo_enfermedad'] = st.number_input("Tiempo de Enfermedad (días)", min_value=0, max_value=365, value=7)
            st.write("---")
            st.subheader("Cálculo de IMC")
            peso = st.number_input("Peso (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.5)
            altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.65, step=0.01)
            if altura > 0:
                inputs['imc'] = round(peso / (altura ** 2), 2)
                st.metric(label="IMC Calculado", value=inputs['imc'])
            else:
                inputs['imc'] = 25.0


        with tab2:
            c1, c2, c3 = st.columns(3)
            inputs['pas'] = c1.number_input("Presión Arterial Sistólica (PAS)", value=120)
            inputs['pad'] = c2.number_input("Presión Arterial Diastólica (PAD)", value=80)
            inputs['fc'] = c3.number_input("Frecuencia Cardíaca (FC)", value=80)
            inputs['fr'] = c1.number_input("Frecuencia Respiratoria (FR)", value=18)
            inputs['temp'] = c2.number_input("Temperatura (°C)", value=37.0, format="%.1f")
            inputs['spo2'] = c3.number_input("Saturación de Oxígeno (SpO2 %)", value=98)

        with tab3:
            c1, c2, c3 = st.columns(3)
            inputs['glucosa'] = c1.number_input("Glucosa (mg/dL)", value=100)
            inputs['hba1c'] = c2.number_input("Hemoglobina Glicosilada (HbA1c %)", value=5.7, format="%.1f")
            inputs['creatinina'] = c1.number_input("Creatinina (mg/dL)", value=1.0, format="%.2f")
            inputs['colesterol'] = c2.number_input("Colesterol Total (mg/dL)", value=180)
            inputs['leucocitos'] = c1.number_input("Leucocitos (células/µL)", value=7500)

        with tab4:
            c1, c2, c3 = st.columns(3)
            inputs['tabaquismo'] = c1.radio("Tabaquismo", options=list(SINO_MAP.keys()), horizontal=True)
            inputs['alcoholismo'] = c2.radio("Alcoholismo", options=list(SINO_MAP.keys()), horizontal=True)
            inputs['sedentarismo'] = c3.radio("Sedentarismo", options=list(SINO_MAP.keys()), horizontal=True)
            inputs['ant_familiar_dm'] = c1.radio("Antecedente Familiar de Diabetes", options=list(SINO_MAP.keys()), horizontal=True)
            inputs['ant_familiar_hta'] = c2.radio("Antecedente Familiar de Hipertensión", options=list(SINO_MAP.keys()), horizontal=True)

        with tab5:
            st.write("Marque los síntomas que presenta el paciente:")
            for sintoma in TODOS_SINTOMAS:
                inputs[sintoma] = st.checkbox(sintoma.replace("sintoma_", " ").replace("_", " ").capitalize())

    # Columna para validaciones y resultados
    with col2:
        st.subheader("Validación y Alertas")
        alertas = []
        for key, (min_val, max_val) in RANGOS_CLINICOS.items():
            if key in inputs and not (min_val <= inputs[key] <= max_val):
                alerta = f"⚠️ **{key.upper()}** ({inputs[key]}) fuera del rango normal ({min_val} - {max_val})."
                st.markdown(f"<div style='background-color: {COLORS['critical_bg']}; color: {COLORS['critical_color']}; border: {COLORS['critical_border']}; padding: 10px; border-radius: 5px;'>🚨 {alerta}</div>", unsafe_allow_html=True)
                alertas.append(alerta)
        if not alertas:
            st.markdown(f"<div style='background-color: {COLORS['normal_bg']}; color: {COLORS['normal_color']}; border: {COLORS['normal_border']}; padding: 10px; border-radius: 5px;'>✅ Todos los valores clínicos están dentro de los rangos de referencia.</div>", unsafe_allow_html=True)

        st.write("---")
        
        if st.button("Analizar Caso Clínico", use_container_width=True, type="primary"):
            # --- Lógica de Predicción ---
            with st.spinner("Procesando datos y ejecutando modelo..."):
                # 1. Crear DataFrame de entrada
                input_data = {}
                
                # Mapear valores categóricos y binarios
                input_data['sexo'] = SEXO_MAP[inputs['sexo']]
                input_data['area'] = AREA_MAP[inputs['area']]
                input_data['tabaquismo'] = SINO_MAP[inputs['tabaquismo']]
                input_data['alcoholismo'] = SINO_MAP[inputs['alcoholismo']]
                input_data['sedentarismo'] = SINO_MAP[inputs['sedentarismo']]
                input_data['ant_familiar_dm'] = SINO_MAP[inputs['ant_familiar_dm']]
                input_data['ant_familiar_hta'] = SINO_MAP[inputs['ant_familiar_hta']]
                
                # Añadir síntomas
                for sintoma in TODOS_SINTOMAS:
                    input_data[sintoma] = 1 if inputs[sintoma] else 0
                
                # Añadir resto de valores numéricos
                for key in ['edad', 'distrito', 'ocupacion', 'imc', 'pas', 'pad', 'fc', 'fr', 'temp', 'spo2', 'glucosa', 'hba1c', 'creatinina', 'colesterol', 'leucocitos', 'tiempo_enfermedad']:
                    input_data[key] = inputs[key]

                # 2. Feature Engineering
                input_data['presion_pulso'] = input_data['pas'] - input_data['pad']
                
                imc_bins = [0, 18.5, 24.9, 29.9, np.inf]
                imc_labels = [0, 1, 2, 3]
                input_data['imc_categoria'] = pd.cut([input_data['imc']], bins=imc_bins, labels=imc_labels, right=False)[0]

                # 3. Ordenar y escalar
                feature_order = resources['model'].feature_names_in_
                df_input = pd.DataFrame([input_data])[feature_order]
                
                numerical_cols = ['edad', 'imc', 'pas', 'pad', 'fc', 'fr', 'temp', 'spo2', 'glucosa', 'hba1c', 'creatinina', 'colesterol', 'leucocitos', 'tiempo_enfermedad', 'presion_pulso']
                df_input[numerical_cols] = resources['scaler'].transform(df_input[numerical_cols])

                # 4. Predicción
                pred_proba = resources['model'].predict_proba(df_input)[0]
                
                # Calcular valores SHAP para la predicción actual
                if resources["explainer"] and resources["feature_names"]:
                    # Ensure df_input has the same columns as the explainer expects
                    # This might involve reordering or adding missing columns with default values
                    # For TreeExplainer, it's usually fine if the order is consistent
                    shap_values_raw = resources["explainer"].shap_values(df_input)
                    # shap_values_raw will be a list of arrays, one for each class
                    # We need to store all of them to allow analysis for any class
                    st.session_state['shap_values'] = shap_values_raw
                    st.session_state['feature_names'] = resources["feature_names"]
                    st.session_state['df_input_processed'] = df_input # Store processed input for waterfall plot
                else:
                    st.warning("SHAP explainer o nombres de características no disponibles. La interpretabilidad no se mostrará.")

                # 5. Procesar Resultados
                top_3_indices = pred_proba.argsort()[-3:][::-1]
                top_3_diagnosticos = [DIAGNOSTICO_MAP[i] for i in top_3_indices]
                top_3_confianzas = [pred_proba[i] for i in top_3_indices]

                diagnostico_principal = top_3_diagnosticos[0]
                confianza_principal = top_3_confianzas[0]

                if confianza_principal >= 0.8:
                    nivel_confianza = "Alta"
                elif confianza_principal >= 0.6:
                    nivel_confianza = "Media"
                else:
                    nivel_confianza = "Baja"

                # Guardar resultados en session_state para el PDF
                st.session_state['results'] = {
                    "diagnostico_principal": diagnostico_principal,
                    "confianza_principal": confianza_principal,
                    "nivel_confianza": nivel_confianza,
                    "top_3_diagnosticos": top_3_diagnosticos,
                    "top_3_confianzas": top_3_confianzas,
                    "alertas": alertas,
                    "inputs": inputs
                }

            st.subheader("Resultado del Análisis")
            st.markdown(f"""
                <div class="main-diagnosis-card">
                    <h3>Diagnóstico Principal: {diagnostico_principal}</h3>
                    <p style="font-size: 1.2rem;">
                        Con una confianza del <strong>{confianza_principal:.2%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if nivel_confianza == "Alta":
                st.markdown(f"<div class='confidence-high'>🟢 ALTA CONFIANZA</div>", unsafe_allow_html=True)
            elif nivel_confianza == "Media":
                st.markdown(f"<div class='confidence-medium'>🟡 CONFIANZA MEDIA</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='confidence-low'>🔴 BAJA CONFIANZA</div>", unsafe_allow_html=True)
            
            with st.expander("Ver Diagnósticos Diferenciales"):
                for i in range(1, len(top_3_diagnosticos)):
                    diag = top_3_diagnosticos[i]
                    conf = top_3_confianzas[i]
                    color_key = f"{diag}_color"
                    bg_key = f"{diag}_bg"
                    icon = ""
                    if diag == "IRA":
                        icon = "🫁"
                    elif diag == "EDA":
                        icon = "🤢"
                    elif diag == "HTA":
                        icon = "💔"
                    elif diag == "DM2":
                        icon = "🩸"
                    st.markdown(f"<div style='color: {COLORS.get(color_key, '#333333')}; background-color: {COLORS.get(bg_key, '#F0F2F6')}; padding: 5px 10px; border-radius: 5px; margin: 5px 0;'>{icon} {i+1}. {diag} ({conf:.2%})</div>", unsafe_allow_html=True)

    # --- Descarga de PDF ---
    if 'results' in st.session_state:
        st.write("---")
        st.subheader("Descargar Reporte")
        
        pdf_bytes = generate_pdf(st.session_state['results'])
        st.markdown(f"""
            <div class="download-button">
                <a href="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" download="reporte_diagnostico_{st.session_state['results']['inputs']['edad']}.pdf">
                    <button style="background-color: {COLORS['download_button_bg']}; color: {COLORS['download_button_text']}; border-radius: 10px; border: none; padding: 10px 20px; cursor: pointer; width: 100%;">
                        📄 Descargar Reporte en PDF
                    </button>
                </a>
            </div>
        """, unsafe_allow_html=True)

def generate_pdf(results):
    """Genera un reporte en PDF con los resultados del diagnóstico."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    
    # Título
    pdf.cell(0, 10, "Reporte de Soporte al Diagnóstico", 0, 1, 'C')
    pdf.ln(10)
    
    # Resultado Principal
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnóstico Principal", 0, 1)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"  - {results['diagnostico_principal']} con una confianza del {results['confianza_principal']:.2%}", 0, 1)
    pdf.cell(0, 10, f"  - Nivel de Confianza General: {results['nivel_confianza']}", 0, 1)
    pdf.ln(5)

    # Diferenciales
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Diagnósticos Diferenciales", 0, 1)
    pdf.set_font("Arial", '', 12)
    for i in range(1, len(results['top_3_diagnosticos'])):
        diag = results['top_3_diagnosticos'][i]
        conf = results['top_3_confianzas'][i]
        pdf.cell(0, 8, f"  {i+1}. {diag} ({conf:.2%})", 0, 1)
    pdf.ln(5)

    # Alertas
    if results['alertas']:
        pdf.set_font("Arial", 'B', 12)
        # Convertir color hexadecimal a RGB para FPDF
        critical_color_hex = COLORS['critical_color'].lstrip('#')
        critical_color_rgb = tuple(int(critical_color_hex[i:i+2], 16) for i in (0, 2, 4))
        pdf.set_text_color(*critical_color_rgb)
        pdf.cell(0, 10, "Alertas Clínicas Identificadas", 0, 1)
        pdf.set_font("Arial", '', 12)
        for alerta in results['alertas']:
            pdf.cell(0, 8, f"  - {alerta.replace('**', '')}", 0, 1)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Este es un reporte generado por un sistema de soporte a la decisión clínica. No reemplaza el juicio de un profesional médico.", 0, 1, 'C')

    return bytes(pdf.output(dest='S'))


def display_analisis(resources):
    st.header("Módulo de Análisis de Resultados")
    st.subheader("Interpretación de la Predicción")

    if 'results' not in st.session_state or 'shap_values' not in st.session_state:
        st.info("Por favor, realice una predicción en el 'Módulo de Predicción de Diagnóstico' primero para ver el análisis de resultados.")
        return

    results = st.session_state['results']
    shap_values = st.session_state['shap_values']
    feature_names = st.session_state['feature_names']
    df_input_processed = st.session_state['df_input_processed']
    diagnostico_principal = results['diagnostico_principal']
    confianza_principal = results['confianza_principal']
    
    st.write(f"**Diagnóstico Principal:** {diagnostico_principal} (Confianza: {confianza_principal:.2%})")
    
    # 1) Gráfico de probabilidades por clase
    st.markdown("---")
    st.subheader("1. Probabilidades por Clase")

    try:
        top_diagnosticos = results.get('top_3_diagnosticos', [])
        top_confianzas = results.get('top_3_confianzas', [])

        # Si no hay top_3, usamos solo el diagnóstico principal
        if not top_diagnosticos or not top_confianzas:
            top_diagnosticos = [diagnostico_principal]
            top_confianzas = [confianza_principal]

        fig_proba, ax_proba = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(top_diagnosticos))

        # Colores: usa un color genérico si no hay definición en COLORS
        bar_colors = []
        for d in top_diagnosticos:
            color_key = f'probability_{d}'
            bar_colors.append(COLORS[color_key] if color_key in COLORS else '#1f77b4')

        ax_proba.barh(y_pos, top_confianzas, align='center', color=bar_colors)
        ax_proba.set_yticks(y_pos)
        ax_proba.set_yticklabels(top_diagnosticos)
        ax_proba.invert_yaxis()
        ax_proba.set_xlabel('Probabilidad')
        ax_proba.set_title('Probabilidad de cada Diagnóstico')
        st.pyplot(fig_proba)

    except Exception as e:
        st.warning(f"No se pudo generar el gráfico de probabilidades: {e}")

    # Get the index of the principal diagnosis
    principal_diag_index = list(DIAGNOSTICO_MAP.keys())[list(DIAGNOSTICO_MAP.values()).index(diagnostico_principal)]

    # 2) Top 10 factores más influyentes usando valores SHAP
    st.markdown("---")
    st.subheader(f"2. Top 10 Factores más Influyentes para {diagnostico_principal}")

    # --- Función robusta para obtener los valores SHAP correctos según el tipo de salida ---
    def select_shap_values(shap_values, class_index=0):
        # Si shap_values es lista o tupla (caso multiclase)
        if isinstance(shap_values, (list, tuple)):
            if class_index < len(shap_values):
                return np.array(shap_values[class_index])
            else:
                return np.array(shap_values[0])
        # Si es numpy array
        sv = np.array(shap_values)
        if sv.ndim == 3:  # (n_classes, n_samples, n_features)
            if class_index < sv.shape[0]:
                return sv[class_index]
            else:
                return sv[0]
        elif sv.ndim == 2:  # (n_samples, n_features)
            return sv
        else:
            return sv.reshape(1, -1)

    # Seleccionar los valores SHAP de la clase principal
    shap_class_values = select_shap_values(shap_values, class_index=principal_diag_index)

    # Asegurar forma (1, n_features)
    if shap_class_values.ndim == 1:
        shap_values_for_summary_plot = shap_class_values.reshape(1, -1)
    elif shap_class_values.shape[0] == 1:
        shap_values_for_summary_plot = shap_class_values
    else:
        shap_values_for_summary_plot = shap_class_values[0].reshape(1, -1)

    # --- Asegurar que las formas de shap y los datos coincidan ---
    if isinstance(df_input_processed, pd.DataFrame):
        data_for_plot = df_input_processed.values
    else:
        data_for_plot = np.array(df_input_processed)

    shap_features = shap_values_for_summary_plot.shape[1]
    data_features = data_for_plot.shape[1]

    if shap_features != data_features:
        min_features = min(shap_features, data_features)
        shap_values_for_summary_plot = shap_values_for_summary_plot[:, :min_features]
        data_for_plot = data_for_plot[:, :min_features]
        feature_names = feature_names[:min_features]

    # --- Graficar ---
    shap.summary_plot(
        shap_values_for_summary_plot,
        data_for_plot,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    st.pyplot(plt.gcf())


    # 3) Visualización tipo waterfall
    st.markdown("---")
    st.subheader(f"3. Cómo cada factor influye en la predicción de {diagnostico_principal}")

    if isinstance(df_input_processed, pd.DataFrame) and len(df_input_processed) == 1:
        # Manejar expected_value de forma segura (puede ser escalar o lista)
        explainer = resources["explainer"]
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            expected_value = explainer.expected_value[min(principal_diag_index, len(explainer.expected_value)-1)]
        else:
            expected_value = explainer.expected_value

        explanation = shap.Explanation(
            values=shap_class_values[0] if shap_class_values.ndim > 1 else shap_class_values,
            base_values=expected_value,
            data=df_input_processed.iloc[0].values,
            feature_names=feature_names
        )

        shap.plots.waterfall(explanation, show=False)
        st.pyplot(plt.gcf())
    else:
        st.warning("No se pudo generar el gráfico de cascada SHAP. Asegúrese de que el input procesado sea un DataFrame de una sola fila.")

    # --------------------------------------------------------------------------
    # 4️⃣ Explicación médica
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Explicación en Lenguaje Médico")

    try:
        shap_for_explanation = shap_class_values[0] if shap_class_values.ndim > 1 else shap_class_values

        # Depuración: verifica tamaños
        st.write(f"🔹 SHAP features: {len(shap_for_explanation)} | Feature names: {len(feature_names)}")

        if len(shap_for_explanation) != len(feature_names):
            st.warning("⚠️ El número de valores SHAP no coincide con el número de características. Se ajustará automáticamente.")
            min_len = min(len(shap_for_explanation), len(feature_names))
            shap_for_explanation = shap_for_explanation[:min_len]
            feature_names = feature_names[:min_len]

        # Generar explicación médica
        medical_explanation = generate_medical_explanation(
            shap_for_explanation,
            feature_names,
            diagnostico_principal,
            df_input_processed.iloc[0]
        )

        # Mostrar texto generado
        if not medical_explanation:
            st.warning("⚠️ No se generó texto de explicación médica. Revise la función `generate_medical_explanation()`.")
            st.code("Posible causa: retorna None o una cadena vacía.")
        else:
            st.markdown(medical_explanation)
    except Exception as e:
        st.error(f"❌ Error al generar la explicación médica: {e}")


    # 5) Recomendaciones clínicas
    st.markdown("---")
    st.subheader("5. Recomendaciones Clínicas")
    clinical_recommendations = get_clinical_recommendations(diagnostico_principal)
    st.markdown(clinical_recommendations)

def display_dashboard():
    st.header("Módulo de Dashboard de Métricas")
    st.info("Esta sección presentará un dashboard con las métricas de rendimiento del modelo.")

# --- Aplicación Principal ---
def main():
    set_custom_style()
    resources = load_resources()
    if resources["error"]:
        st.sidebar.error(resources["error"])
        st.error(resources["error"])
        st.warning("La aplicación no puede funcionar hasta que los archivos del modelo y scaler estén en la ubicación correcta (`models/`).")
        return

    st.sidebar.title("Navegación")
    selection = st.sidebar.radio(
        "Ir a:",
        ["Inicio", "Predicción de Diagnóstico", "Análisis de Resultados", "Dashboard de Métricas"],
        label_visibility="collapsed"
    )

    if selection == "Inicio":
        display_inicio()
    elif selection == "Predicción de Diagnóstico":
        display_prediccion(resources)
    elif selection == "Análisis de Resultados":
        display_analisis(resources)
    elif selection == "Dashboard de Métricas":
        display_dashboard()

if __name__ == "__main__":
    main()