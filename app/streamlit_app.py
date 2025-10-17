import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from fpdf import FPDF
import base64

# --- Configuración de la Página ---
st.set_page_config(
    page_title="CDSS Huancayo - Sistema de Soporte al Diagnóstico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilo y Diseño Profesional (Tema Médico) ---
def set_custom_style():
    """ Inyecta CSS personalizado para un tema médico profesional. """
    st.markdown('''
    <style>
        .stApp {
            background-color: #F0F2F6;
        }
        .st-sidebar {
            background-color: #FFFFFF;
        }
        .st-sidebar .st-emotion-cache-16txtl3 {
            color: #0D47A1;
        }
        h1, h2, h3 {
            color: #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
    </style>
    ''', unsafe_allow_html=True)

# --- Carga de Recursos ---
@st.cache_resource
def load_resources():
    """ Carga el modelo, scaler y otros recursos necesarios. """
    base_path = Path(__file__).resolve().parent.parent
    model_path = base_path / "models" / "final_model.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"
    
    resources = {"model": None, "scaler": None, "error": None}

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
        with open(scaler_path, 'rb') as f:
            resources["scaler"] = pickle.load(f)
    except FileNotFoundError:
        resources["error"] = f"Error: No se encontró el archivo del scaler en: {scaler_path}"
    except Exception as e:
        resources["error"] = f"Error al cargar el scaler: {e}"
        
    return resources

# --- Mapeos y Definiciones ---
DIAGNOSTICO_MAP = {0: 'DM2', 1: 'EDA', 2: 'HTA', 3: 'IRA'}
SEXO_MAP = {'Femenino': 0, 'Masculino': 1}
AREA_MAP = {'Rural': 0, 'Urbano': 1}
SINO_MAP = {'No': 0, 'Sí': 1}

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
                st.warning(alerta)
                alertas.append(alerta)
        if not alertas:
            st.success("Todos los valores clínicos están dentro de los rangos de referencia.")

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
            st.success(f"**Diagnóstico Principal:** {diagnostico_principal}")
            st.metric(label="Nivel de Confianza", value=f"{confianza_principal:.2%}", delta=nivel_confianza)
            
            with st.expander("Ver Diagnósticos Diferenciales"):
                for i in range(1, len(top_3_diagnosticos)):
                    st.text(f"{i+1}. {top_3_diagnosticos[i]} ({top_3_confianzas[i]:.2%})")

    # --- Descarga de PDF ---
    if 'results' in st.session_state:
        st.write("---")
        st.subheader("Descargar Reporte")
        
        pdf_bytes = generate_pdf(st.session_state['results'])
        st.download_button(
            label="Descargar Reporte en PDF",
            data=pdf_bytes,
            file_name=f"reporte_diagnostico_{st.session_state['results']['inputs']['edad']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

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
        pdf.set_text_color(220, 50, 50) # Rojo
        pdf.cell(0, 10, "Alertas Clínicas Identificadas", 0, 1)
        pdf.set_font("Arial", '', 12)
        for alerta in results['alertas']:
            pdf.cell(0, 8, f"  - {alerta.replace('**', '')}", 0, 1)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, "Este es un reporte generado por un sistema de soporte a la decisión clínica. No reemplaza el juicio de un profesional médico.", 0, 1, 'C')

    return pdf.output(dest='S').encode('latin-1')


def display_analisis():
    st.header("Módulo de Análisis de Resultados")
    st.info("Esta sección mostrará gráficos de interpretabilidad (ej. SHAP values) para entender las predicciones del modelo.")

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
        display_analisis()
    elif selection == "Dashboard de Métricas":
        display_dashboard()

if __name__ == "__main__":
    main()