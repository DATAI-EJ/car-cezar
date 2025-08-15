import streamlit as st
import geopandas as gpd
import pandas as pd
from typing import List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import unicodedata
import os
import numpy as np
import duckdb
import logging
import psutil
from io import BytesIO

def safe_format_number(value, decimals=1):
    try:
        if pd.isna(value) or value is None:
            return "0"
        
        num_value = float(value)
        
        if abs(num_value) < 0.001:
            return "0"
        
        if decimals == 0:
            return f"{num_value:,.0f}".replace(',', '.')
        else:
            formatted = f"{num_value:,.{decimals}f}"
            if '.' in formatted:
                parts = formatted.split('.')
                integer_part = parts[0].replace(',', '.')
                decimal_part = parts[1]
                return f"{integer_part},{decimal_part}"
            else:
                return formatted.replace(',', '.')
                
    except (ValueError, TypeError, AttributeError):
        return "Erro"

def format_number_with_dots(number, decimal_places=1):
    if pd.isna(number) or number is None:
        return "0"
    
    try:
        num = float(number)
        if num == 0:
            return "0"
        if decimal_places == 0:
            formatted = f"{num:,.0f}"
        else:
            formatted = f"{num:,.{decimal_places}f}"
        if '.' in formatted:
            parts = formatted.split('.')
            integer_part = parts[0]
            decimal_part = parts[1] if len(parts) > 1 else ""
            integer_part = integer_part.replace(',', '.')
            if decimal_part and decimal_places > 0:
                return f"{integer_part},{decimal_part}"
            else:
                return integer_part
        else:
            return formatted.replace(',', '.')
            
    except (ValueError, TypeError, AttributeError) as e:
        try:
            return f"{float(number):,.{decimal_places}f}".replace(',', '.')
        except:
            return str(number) if number is not None else "0"

def create_custom_tickformat(values):
    if not values:
        return {}
    
    formatted_values = []
    for val in values:
        if val >= 1000000:
            formatted_values.append(f"{format_number_with_dots(val/1000000, 1)}M")
        elif val >= 1000:
            formatted_values.append(f"{format_number_with_dots(val/1000, 0)}k")
        else:
            formatted_values.append(format_number_with_dots(val, 0))
    
    return dict(zip(values, formatted_values))

def preparar_dados_especificos(gdf_input):
    if gdf_input is None or gdf_input.empty:
        return None
    
    df_download = gdf_input.copy()
    if 'geometry' in df_download.columns:
        df_download = df_download.drop('geometry', axis=1)
    
    return df_download

st.set_page_config(
    page_title="Dashboard de Conflitos Ambientais",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ---------- Fundo geral do app ---------- */
[data-testid="stAppViewContainer"] {
    background-color: #fefcf9;
    padding: 2rem;
    font-family: 'Segoe UI', sans-serif;
    color: #333333;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #f3f0eb;
    border-right: 2px solid #d8d2ca;
}
[data-testid="stSidebar"] > div {
    padding: 1rem;
}

/* ---------- Botões ---------- */
.stButton > button {
    background-color: #cbe4d2;
    color: #2d3a2f;
    border: 2px solid #a6c4b2;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    background-color: #b4d6c1;
    color: #1e2a21;
}

/* ---------- Títulos e textos ---------- */
h1, h2, h3 {
    color: #4a4a4a;
}
h1 {
    font-size: 2.2rem;
    border-bottom: 2px solid #d8d2ca;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab"] {
    background-color: #ebe7e1;
    color: #333;
    border-radius: 0.5rem 0.5rem 0 0;
    padding: 0.5rem 1rem;
    margin-right: 0.25rem;
    font-weight: bold;
    border: none;
}
.stTabs [aria-selected="true"] {
    background-color: #d6ccc2;
    color: #111;
}

/* ---------- Text input ---------- */
.stTextInput > div > input {
    background-color: #f9f6f2;
    border: 1px solid #ccc;
    border-radius: 0.5rem;
    padding: 0.5rem;
}

/* ---------- Selectbox ---------- */
.stSelectbox > div {
    background-color: #f9f6f2;
    border-radius: 0.5rem;
}

/* ---------- Expander ---------- */
.stExpander > details {
    background-color: #f2eee9;
    border: 1px solid #ddd3c7;
    border-radius: 0.5rem;
    padding: 0.5rem;
}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #f3f0eb;
}
::-webkit-scrollbar-thumb {
    background-color: #b4d6c1;
    border-radius: 10px;
    border: 2px solid #f3f0eb;
}
</style>
""", unsafe_allow_html=True)

def _apply_layout(fig: go.Figure, title: str, title_size: int = 16) -> go.Figure:
    fig.update_layout(
        template="pastel",
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font_size": title_size
        },
        paper_bgcolor="white",   
        plot_bgcolor="white",     
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCC",
            borderwidth=1,
            font=dict(size=10)
        )
    )
    return fig

base_layout = go.Layout(
    font=dict(family="Times New Roman", size=12),
    plot_bgcolor='white',
    paper_bgcolor='white',
    colorway=px.colors.qualitative.Pastel,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Times New Roman"
    )
)

pastel_template = go.layout.Template(layout=base_layout)
pio.templates["pastel"] = pastel_template
pio.templates.default = "pastel"
PASTEL_SEQ = px.colors.qualitative.Pastel + px.colors.qualitative.Pastel1 + px.colors.qualitative.Pastel2

_original_px_bar = px.bar

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        st.image("logo_cezar.jpg", width=300)
    except:
        st.warning("Logo não encontrada")

st.title("Análise de Conflitos em Áreas Protegidas e Territórios Tradicionais")
st.markdown("Monitoramento integrado de sobreposições em Unidades de Conservação, Terras Indígenas e Territórios Quilombolas")

st.markdown("---")

def _patched_px_bar(*args, **kwargs) -> go.Figure:
    fig: go.Figure = _original_px_bar(*args, **kwargs)
    seq = PASTEL_SEQ
    barmode = getattr(fig.layout, 'barmode', '') or ''
    barras = [t for t in fig.data if isinstance(t, go.Bar)]
    if barmode == 'stack':
        for i, trace in enumerate(barras):
            trace.marker.color = seq[i % len(seq)]
    else:
        if len(barras) == 1:
            trace = barras[0]
            vals = trace.x if getattr(trace, 'orientation', None) != 'h' else trace.y
            if hasattr(vals, 'tolist'):
                vals = vals.tolist()
            trace.marker.color = [seq[i % len(seq)] for i in range(len(vals))]
        else:
            for i, trace in enumerate(barras):
                trace.marker.color = seq[i % len(seq)]
    return fig

px.bar = _patched_px_bar

@st.cache_data
def carregar_cnuc_adaptativo(caminho: str) -> gpd.GeoDataFrame:
    try:
        if not os.path.exists(caminho):
            st.error(f"❌ Arquivo não encontrado: {caminho}")
            return gpd.GeoDataFrame()
        gdf = gpd.read_file(caminho)
        
        if gdf.empty:
            st.warning(f" Shapefile {caminho} está vazio")
            return gpd.GeoDataFrame()

        colunas_essenciais = ['geometry', 'nome_uc', 'municipio']
        colunas_opcionais = ['alerta_km2', 'sigef_km2', 'area_km2', 'c_alertas', 'c_sigef']

        for col in colunas_essenciais:
            if col not in gdf.columns:
                st.error(f"❌ Coluna essencial '{col}' não encontrada em {caminho}")
                return gpd.GeoDataFrame()
        
        for col in colunas_opcionais:
            if col not in gdf.columns:
                gdf[col] = 0
                st.info(f"➕ Coluna '{col}' adicionada com valor 0")
        if 'area_km2' in gdf.columns:
            gdf['ha_total'] = gdf['area_km2'] * 100
        else:
            gdf['ha_total'] = 0
            st.warning(" Coluna 'area_km2' não encontrada, ha_total definido como 0")
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar {caminho}: {str(e)}")
        return gpd.GeoDataFrame()
    try:
        if not os.path.exists(caminho):
            st.error(f"❌ Arquivo não encontrado: {caminho}")
            return gpd.GeoDataFrame()
        
        gdf = gpd.read_file(caminho)
        
        if gdf.empty:
            st.warning(f"⚠️ Shapefile {caminho} está vazio")
            return gpd.GeoDataFrame()
        
        if columns:
            available_cols = [col for col in columns if col in gdf.columns]
            missing_cols = [col for col in columns if col not in gdf.columns]
            
            if missing_cols:
                st.warning(f" Colunas não encontradas em {caminho}: {missing_cols}")
                
            if available_cols:
                gdf = gdf[available_cols]
        
        return gdf
        
    except Exception as e:
        st.error(f" Erro ao carregar {caminho}: {str(e)}")
        return gpd.GeoDataFrame()

@st.cache_data
def carregar_shapefile_cloud_safe(caminho: str, calcular_percentuais: bool = True, columns: list[str] = None) -> gpd.GeoDataFrame:
    try:
        if not os.path.exists(caminho):
            st.error(f"❌ Arquivo não encontrado: {caminho}")
            return gpd.GeoDataFrame()
        
        gdf = gpd.read_file(caminho)
        
        if gdf.empty:
            st.warning(f"⚠️ Shapefile {caminho} está vazio")
            return gpd.GeoDataFrame()
        
        if columns:
            available_cols = [col for col in columns if col in gdf.columns]
            missing_cols = [col for col in columns if col not in gdf.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Colunas não encontradas em {caminho}: {missing_cols}")
                
            if available_cols:
                gdf = gdf[available_cols]
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Erro ao carregar {caminho}: {str(e)}")
        return gpd.GeoDataFrame()

@st.cache_data
def carregar_shapefile(caminho: str, calcular_percentuais: bool = True, columns: list[str] = None) -> gpd.GeoDataFrame:
    """Carrega um shapefile, calcula áreas e percentuais, e otimiza tipos de dados."""
    gdf = gpd.read_file(caminho, columns=columns or [])
    
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.buffer(0) if geom and not geom.is_valid else geom)
    gdf = gdf[gdf["geometry"].notnull() & gdf["geometry"].is_valid]
    
    if "area_km2" in gdf.columns or calcular_percentuais:
        try:
            gdf_proj = gdf.to_crs("EPSG:31983") 
            gdf_proj["area_calc_km2"] = gdf_proj.geometry.area / 1e6
            if "area_km2" in gdf.columns:
                gdf["area_km2"] = gdf["area_km2"].replace(0, np.nan).fillna(gdf_proj["area_calc_km2"])
            else:
                gdf["area_km2"] = gdf_proj["area_calc_km2"]
        except Exception as e:
            st.warning(f"Could not reproject for area calculation: {e}. Using existing 'area_km2' or skipping area calcs.")
            if "area_km2" not in gdf.columns:
                 gdf["area_km2"] = np.nan 

    if calcular_percentuais and "area_km2" in gdf.columns:
        gdf["perc_alerta"] = (gdf.get("alerta_km2", 0) / gdf["area_km2"]) * 100
        gdf["perc_sigef"] = (gdf.get("sigef_km2", 0) / gdf["area_km2"]) * 100
        gdf["perc_alerta"] = gdf["perc_alerta"].replace([np.inf, -np.inf], np.nan).fillna(0)
        gdf["perc_sigef"] = gdf["perc_sigef"].replace([np.inf, -np.inf], np.nan).fillna(0)
    else:
        if "perc_alerta" not in gdf.columns: gdf["perc_alerta"] = 0
        if "perc_sigef" not in gdf.columns: gdf["perc_sigef"] = 0

    gdf["id"] = gdf.index.astype(str)

    for col in gdf.columns:
        if gdf[col].dtype == 'float64':
            gdf[col] = pd.to_numeric(gdf[col], downcast='float', errors='coerce')
        elif gdf[col].dtype == 'int64':
            gdf[col] = pd.to_numeric(gdf[col], downcast='integer', errors='coerce')
        elif gdf[col].dtype == 'object':
            if len(gdf[col].unique()) / len(gdf) < 0.5: 
                 try:
                    gdf[col] = gdf[col].astype('category')
                 except Exception:
                    pass 

    return gdf.to_crs("EPSG:4326")

def preparar_hectares(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Adiciona colunas em hectares ao GeoDataFrame."""
    gdf2 = gdf.copy()
    gdf2['alerta_ha'] = gdf2.get('alerta_km2', 0) * 100
    gdf2['sigef_ha']  = gdf2.get('sigef_km2', 0)  * 100
    gdf2['area_ha']   = gdf2.get('area_km2', 0)   * 100
    
    for col in ['alerta_ha', 'sigef_ha', 'area_ha']:
         if gdf2[col].dtype == 'float64':
            gdf2[col] = pd.to_numeric(gdf2[col], downcast='float', errors='coerce')
         elif gdf2[col].dtype == 'int64':
            gdf2[col] = pd.to_numeric(gdf2[col], downcast='integer', errors='coerce')

    return gdf2

@st.cache_data
def load_csv(uploaded_file, columns: list[str] = None) -> pd.DataFrame:
    usecols_arg = None
    if columns is not None:
        usecols_arg = lambda col: col in columns

    try:
        df = pd.read_csv(
            uploaded_file,
            low_memory=False,
            usecols=usecols_arg
        )
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            low_memory=False,
            usecols=usecols_arg,
            encoding='latin-1'
        )
    except Exception as e:
        st.error(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()


    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Município"})
    
    cols_ocorrencias = [
        "Áreas de conflitos", "Assassinatos", "Conflitos por Terra",
        "Ocupações Retomadas", "Tentativas de Assassinatos", "Trabalho Escravo"
    ]
    existing = [c for c in cols_ocorrencias if c in df.columns]
    
    if existing:
        df["total_ocorrencias"] = df[existing].sum(axis=1)
        df["total_ocorrencias"] = pd.to_numeric(
            df["total_ocorrencias"],
            downcast='integer',
            errors='coerce'
        )
    else:
        df["total_ocorrencias"] = 0

    for col in df.columns:
        dtype = df[col].dtype
        if dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
        elif dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
        elif dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:
                try:
                    df[col] = df[col].astype('category')
                except Exception:
                    pass
    return df
    
@st.cache_data
def carregar_dados_conflitos_municipio(arquivo_excel: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(arquivo_excel, sheet_name='Áreas em Conflito', usecols=['mun', 'Famílias', 'Nome do Conflito']).dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao ler o arquivo Excel de conflitos: {e}")
        return pd.DataFrame()

    lista_original = ['SÃO FÉLIX DO XINGU', 'ALTAMIRA', 'ITAITUBA',
                      'JACAREACANGA', 'NOVO PROGRESSO']

    def clean_mun_name(name):
        if pd.isna(name):
            return None
        name = str(name).strip().lower()
        name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')
        return name

    lista_limpa = [clean_mun_name(m) for m in lista_original]
    lista_limpa = [m for m in lista_limpa if m is not None] 

    df['mun_limpo_list'] = df['mun'].apply(lambda x: [
        clean_mun_name(m) for m in str(x).replace(';', ',').split(',')
    ])
    df_exploded = df.explode('mun_limpo_list')
    df_exploded = df_exploded[df_exploded['mun_limpo_list'].notna() & (df_exploded['mun_limpo_list'] != '')].copy()
    df_filtered = df_exploded[df_exploded['mun_limpo_list'].isin(lista_limpa)].copy()

    if df_filtered.empty:
        st.warning("Nenhum município da lista de interesse encontrado nos dados de conflitos após a limpeza.")
        return pd.DataFrame(columns=['Município', 'Total_Famílias', 'Número_Conflitos'])

    df_filtered['Famílias'] = pd.to_numeric(df_filtered['Famílias'], errors='coerce').fillna(0)
    df_filtered['Famílias'] = pd.to_numeric(df_filtered['Famílias'], downcast='integer', errors='coerce')
    conflitos_presentes = df_filtered['Nome do Conflito'].unique()
    df_conflitos_relevantes = df_exploded[df_exploded['Nome do Conflito'].isin(conflitos_presentes)].copy()

    df_conflitos_relevantes['num_mun'] = df_conflitos_relevantes.groupby('Nome do Conflito', observed=False)['mun_limpo_list'].transform('nunique')
    df_conflitos_relevantes['Fam_por_mun'] = df_conflitos_relevantes['Famílias'] / df_conflitos_relevantes['num_mun']

    df_conflitos_relevantes['num_mun'] = pd.to_numeric(df_conflitos_relevantes['num_mun'], downcast='integer', errors='coerce')
    df_conflitos_relevantes['Fam_por_mun'] = pd.to_numeric(df_conflitos_relevantes['Fam_por_mun'], downcast='float', errors='coerce')

    res = df_conflitos_relevantes.groupby('mun_limpo_list', observed=False).agg({
        'Fam_por_mun':'sum', 
        'Nome do Conflito':'count'
    }).reset_index()

    res.columns = ['Município_Limpo','Total_Famílias','Número_Conflitos']
    res = res.rename(columns={'Município_Limpo': 'Município'})
    cleaned_to_original_map = {clean_mun_name(orig): orig.title() for orig in lista_original}
    res['Município'] = res['Município'].map(cleaned_to_original_map).fillna(res['Município']) 

    res['Total_Famílias'] = pd.to_numeric(res['Total_Famílias'], downcast='integer', errors='coerce')
    res['Número_Conflitos'] = pd.to_numeric(res['Número_Conflitos'], downcast='integer', errors='coerce')

    if not res.empty and len(res['Município'].unique()) / len(res) < 0.5:
        try:
            res['Município'] = res['Município'].astype('category')
        except Exception:
            pass

    return res

def criar_figura(gdf_cnuc_filtered, gdf_sigef_filtered, df_csv_filtered, centro, ids_selecionados, invadindo_opcao):
    try:
        fig = px.choropleth_map(
            gdf_cnuc_filtered,
            geojson=gdf_cnuc_filtered.__geo_interface__,
            locations=gdf_cnuc_filtered.index,
            color=np.ones(len(gdf_cnuc_filtered)),
            color_continuous_scale=[[0, "rgba(34,139,34,0.6)"], [1, "rgba(34,139,34,0.6)"]],
            map_style="open-street-map",
            zoom=5,
            center=centro,
            opacity=0.7,
            hover_data={
                'nome_uc': True,
                'municipio': True,
                'area_km2': ':.2f',
                'alerta_km2': ':.2f',
                'sigef_km2': ':.2f'
            }
        )
        fig.update_coloraxes(showscale=False)
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>" +
                "Município: %{customdata[1]}<br>" +
                "Área: %{customdata[2]:.2f} km²<br>" +
                "Alertas: %{customdata[3]:.2f} km²<br>" +
                "CAR: %{customdata[4]:.2f} km²<extra></extra>"
            )
        )

        if invadindo_opcao:
            if invadindo_opcao.lower() == "todos":
                sigef_plot = gdf_sigef_filtered
            else:
                sigef_plot = gdf_sigef_filtered[
                    gdf_sigef_filtered["invadindo"].str.strip().str.lower() == invadindo_opcao.lower()
                ]
            
            if not sigef_plot.empty:
                fig_sigef = px.choropleth_map(
                    sigef_plot,
                    geojson=sigef_plot.__geo_interface__,
                    locations=sigef_plot.index,
                    color=np.ones(len(sigef_plot)),
                    color_continuous_scale=[[0, "rgba(255,140,0,0.8)"], [1, "rgba(255,140,0,0.8)"]],
                    opacity=0.8
                )
                fig_sigef.update_coloraxes(showscale=False)
                for trace in fig_sigef.data:
                    fig.add_trace(trace)

        if df_csv_filtered is not None and not df_csv_filtered.empty:
            df_plot = df_csv_filtered.dropna(subset=['Latitude', 'Longitude']).drop_duplicates(subset=['Município'])
            
            if not df_plot.empty:
                conflitos_cols = [
                    'Áreas de conflitos', 'Assassinatos', 'Conflitos por Terra',
                    'Ocupações Retomadas', 'Tentativas de Assassinatos', 'Trabalho Escravo'
                ]
                
                existing_cols = [col for col in conflitos_cols if col in df_plot.columns]
                
                if existing_cols:
                    customdata = df_plot[existing_cols]
                    hovertemplate = "<b>%{text}</b><br>"
                    for i, col in enumerate(existing_cols):
                        hovertemplate += f"{col}: %{{customdata[{i}]}}<br>"
                    hovertemplate += "<extra></extra>"
                else:
                    customdata = [[]] * len(df_plot)
                    hovertemplate = "<b>%{text}</b><extra></extra>"

                fig.add_trace(
                    go.Scattermapbox(
                        lat=df_plot['Latitude'],
                        lon=df_plot['Longitude'],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color='red',
                            opacity=0.7,
                            symbol='circle'
                        ),
                        text=df_plot['Município'],
                        textposition="top center",
                        textfont=dict(size=10, color="black"),
                        hovertemplate=hovertemplate,
                        customdata=customdata,
                        name='Municípios'
                    )
                )

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                zoom=5,
                center=centro
            ),
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        
        return fig

    except Exception as e:
        st.error(f"Erro ao criar mapa: {e}")
        return go.Figure()
    
def criar_cards(gdf_cnuc_filtered, gdf_sigef_filtered, invadindo_opcao):
    try:
        ucs_selecionadas = gdf_cnuc_filtered.copy()
        sigef_base = gdf_sigef_filtered.copy()
        
        if ucs_selecionadas.empty:
            return (0.0, 0.0, 0, 0, 0)

        crs_proj = "EPSG:31983"
        ucs_proj = ucs_selecionadas.to_crs(crs_proj)
        sigef_proj = sigef_base.to_crs(crs_proj)

        if invadindo_opcao and invadindo_opcao.lower() != "todos":
            mascara = sigef_proj["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()
            sigef_filtrado = sigef_proj[mascara].copy()
        else:
            sigef_filtrado = sigef_proj.copy()
        if not ucs_proj.empty and not sigef_filtrado.empty:
            sobreposicao = gpd.overlay(
                ucs_proj,
                sigef_filtrado,
                how='intersection',
                keep_geom_type=False,
                make_valid=True
            )
            sobreposicao['area_sobreposta'] = sobreposicao.geometry.area / 1e6
            total_sigef = sobreposicao['area_sobreposta'].sum()
            contagem_sigef_overlay = sobreposicao.shape[0]
        else:
            total_sigef = 0.0
            contagem_sigef_overlay = 0

        total_area_ucs = ucs_proj.geometry.area.sum() / 1e6
        total_alerta = ucs_selecionadas.get("alerta_km2", pd.Series([0])).sum()
        contagem_alerta_uc = ucs_selecionadas.get("c_alertas", pd.Series([0])).sum() 

        perc_alerta = (total_alerta / total_area_ucs * 100) if total_area_ucs > 0 else 0
        perc_sigef = (total_sigef / total_area_ucs * 100) if total_area_ucs > 0 else 0

        municipios = set()
        if "municipio" in ucs_selecionadas.columns:
            for munic in ucs_selecionadas["municipio"]:
                if pd.notna(munic):
                    partes = str(munic).replace(';', ',').split(',')
                    for parte in partes:
                        if parte.strip():
                            municipios.add(parte.strip().title())

        return (
            round(perc_alerta, 1),
            round(perc_sigef, 1),
            len(municipios),
            int(contagem_alerta_uc),
            int(contagem_sigef_overlay) 
        ) 

    except Exception as e:
        st.error(f"Erro crítico ao criar cards: {str(e)}")
        return (0.0, 0.0, 0, 0, 0)
    
def render_cards(perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef):
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    
    card_html_template = """
    <div style="
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        height: 100px;  <!-- Fixed height -->
        display: flex;
        flex-direction: column;
        justify-content: center;">
        <div style="font-size: 0.9rem; color: #6FA8DC;">{titulo}</div>
        <div style="font-size: 1.2rem; font-weight: bold; color: #2F5496;">{valor}</div>
        <div style="font-size: 0.7rem; color: #666;">{descricao}</div>
    </div>
    """
    
    perc_alerta_fmt = f"{perc_alerta:.1f}%".replace('.', ',')
    perc_sigef_fmt = f"{perc_sigef:.1f}%".replace('.', ',')
    
    with col1:
        st.markdown(
            card_html_template.format(
                titulo="Alertas / Ext. Ter.",
                valor=perc_alerta_fmt,
                descricao="Área de alertas sobre extensão territorial"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card_html_template.format(
                titulo="CARs / Ext. Ter.", 
                valor=perc_sigef_fmt,
                descricao="CARs sobre extensão territorial"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            card_html_template.format(
                titulo="Municípios Abrangidos",
                valor=format_number_with_dots(total_unidades, 0),
                descricao="Total de municípios na análise"
            ),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            card_html_template.format(
                titulo="Alertas",
                valor=format_number_with_dots(contagem_alerta, 0),
                descricao="Total de registros de alertas"
            ),
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            card_html_template.format(
                titulo="CARs",
                valor=format_number_with_dots(contagem_sigef, 0),
                descricao="Cadastros Ambientais Rurais"
            ),
            unsafe_allow_html=True
        )

import textwrap

def truncate(text, max_chars=15):
    return text if len(text) <= max_chars else text[:max_chars-3] + "..."

def wrap_label(name, width=30):
    if pd.isna(name): return ""
    return "<br>".join(textwrap.wrap(str(name), width))

def fig_sobreposicoes(gdf_cnuc_ha_filtered):
    gdf = gdf_cnuc_ha_filtered.copy().sort_values("area_ha", ascending=False)
    if gdf.empty:
        return go.Figure()

    gdf["uc_short"] = gdf["nome_uc"].apply(lambda x: wrap_label(x, 15))
    
    fig = go.Figure()
    
    alerta_text = [format_number_with_dots(val, 0) for val in gdf["alerta_ha"]]
    sigef_text = [format_number_with_dots(val, 0) for val in gdf["sigef_ha"]]
    area_text = [format_number_with_dots(val, 0) for val in gdf["area_ha"]]
    
    fig.add_trace(go.Bar(
        name='Alertas',
        x=gdf["uc_short"],
        y=gdf["alerta_ha"],
        marker_color='#99CD85',
        text=alerta_text,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Alertas: %{text} ha<extra></extra>',
        customdata=alerta_text
    ))
    
    fig.add_trace(go.Bar(
        name='CARs',
        x=gdf["uc_short"],
        y=gdf["sigef_ha"],
        marker_color='#CFE0BC',
        text=sigef_text,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>CARs: %{text} ha<extra></extra>',
        customdata=sigef_text
    ))
    
    fig.add_trace(go.Bar(
        name='UCs',
        x=gdf["uc_short"],
        y=gdf["area_ha"],
        marker_color='#7FA653',
        text=area_text,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>UCs: %{text} ha<extra></extra>',
        customdata=area_text
    ))
    
    fig.update_layout(
        barmode='stack',
        height=400,
        xaxis=dict(tickangle=0, tickfont=dict(size=9), title_text=""),
        yaxis=dict(title_text="Área (ha)", tickfont=dict(size=9), tickformat='~s')
    )
    
    return _apply_layout(fig, title="Áreas por UC", title_size=16)

def fig_sobreposicoes_mapbiomas(gdf_cnuc_raw, gdf_alertas_raw):
    """
    Versão atualizada que usa dados do MapBiomas para alertas, garantindo consistência
    """
    if gdf_cnuc_raw.empty:
        return go.Figure()
    
    # Calcular dados de alertas usando MapBiomas (mesma metodologia da aba Desmatamento)
    try:
        crs_proj = "EPSG:31983"
        gdf_cnuc_proj = gdf_cnuc_raw.to_crs(crs_proj)
        gdf_alertas_proj = gdf_alertas_raw.to_crs(crs_proj)
        
        # Análise espacial para alertas
        alerts_in_ucs = gpd.sjoin(gdf_alertas_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
        
        if not alerts_in_ucs.empty:
            alert_area_per_uc = alerts_in_ucs.groupby('nome_uc', observed=False)['AREAHA'].sum().reset_index()
            alert_area_per_uc.columns = ['nome_uc', 'alerta_mapbiomas_ha']
        else:
            alert_area_per_uc = pd.DataFrame(columns=['nome_uc', 'alerta_mapbiomas_ha'])
        
        # Preparar dados das UCs
        gdf_display = gdf_cnuc_raw.copy()
        gdf_display['area_ha'] = gdf_display.get('ha_total', 0)
        gdf_display['sigef_ha'] = gdf_display.get('sigef_km2', 0) * 100
        
        # Merge com dados do MapBiomas
        gdf_display = gdf_display.merge(alert_area_per_uc, on='nome_uc', how='left')
        gdf_display['alerta_mapbiomas_ha'] = gdf_display['alerta_mapbiomas_ha'].fillna(0)
        
        # Ordenar por área da UC
        gdf_display = gdf_display.sort_values("area_ha", ascending=False)
        gdf_display["uc_short"] = gdf_display["nome_uc"].apply(lambda x: wrap_label(x, 15))
        
        # Criar gráfico
        fig = go.Figure()
        
        alerta_text = [format_number_with_dots(val, 0) for val in gdf_display["alerta_mapbiomas_ha"]]
        sigef_text = [format_number_with_dots(val, 0) for val in gdf_display["sigef_ha"]]
        area_text = [format_number_with_dots(val, 0) for val in gdf_display["area_ha"]]
        
        fig.add_trace(go.Bar(
            name='Alertas',
            x=gdf_display["uc_short"],
            y=gdf_display["alerta_mapbiomas_ha"],
            marker_color='#99CD85',
            text=alerta_text,
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>Alertas: %{text} ha<extra></extra>',
            customdata=alerta_text
        ))
        
        fig.add_trace(go.Bar(
            name='CARs',
            x=gdf_display["uc_short"],
            y=gdf_display["sigef_ha"],
            marker_color='#CFE0BC',
            text=sigef_text,
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>CARs: %{text} ha<extra></extra>',
            customdata=sigef_text
        ))
        
        fig.add_trace(go.Bar(
            name='UCs',
            x=gdf_display["uc_short"],
            y=gdf_display["area_ha"],
            marker_color='#7FA653',
            text=area_text,
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>UCs: %{text} ha<extra></extra>',
            customdata=area_text
        ))
        
        fig.update_layout(
            barmode='stack',
            height=400,
            xaxis=dict(tickangle=0, tickfont=dict(size=9), title_text=""),
            yaxis=dict(title_text="Área (ha)", tickfont=dict(size=9), tickformat='~s')
        )
        
        return _apply_layout(fig, title="Áreas por UC (Dados MapBiomas)", title_size=16)
        
    except Exception as e:
        # Fallback para função original se houver erro
        return fig_sobreposicoes(gdf_cnuc_raw)

def fig_contagens_uc(gdf_cnuc_filtered: gpd.GeoDataFrame) -> go.Figure:
    gdf = gdf_cnuc_filtered.copy()
    if gdf.empty:
        return go.Figure()
    gdf["total_counts"] = gdf.get("c_alertas", 0) + gdf.get("c_sigef", 0)
    gdf = gdf.sort_values("total_counts", ascending=False)
    
    gdf["uc_wrap"] = gdf["nome_uc"].apply(lambda x: wrap_label(x, 15))
    
    # Formatar valores para exibição com pontos nos milhares
    alertas_text = [format_number_with_dots(val, 0) for val in gdf.get("c_alertas", 0)]
    sigef_text = [format_number_with_dots(val, 0) for val in gdf.get("c_sigef", 0)]
    
    # Criar figura com cores personalizadas
    fig = go.Figure()
    
    # Adicionar barras manualmente para controle total das cores
    fig.add_trace(go.Bar(
        name='Alertas',
        x=gdf["uc_wrap"],
        y=gdf.get("c_alertas", 0),
        marker_color='#99CD85',
        text=alertas_text,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>Alertas: %{text}<extra></extra>',
        customdata=alertas_text
    ))
    
    fig.add_trace(go.Bar(
        name='CARs',
        x=gdf["uc_wrap"],
        y=gdf.get("c_sigef", 0),
        marker_color='#63783D',
        text=sigef_text,
        textposition='inside',
        hovertemplate='<b>%{x}</b><br>CARs: %{text}<extra></extra>',
        customdata=sigef_text
    ))
    
    fig.update_layout(
        barmode='stack',
        height=400,
        xaxis=dict(tickangle=0, tickfont=dict(size=9), title_text=""),
        yaxis=dict(title_text="Contagens", tickfont=dict(size=9), tickformat='~s')
    )
    
    return _apply_layout(fig, title="Contagens por UC", title_size=16)

def fig_car_por_uc_donut(gdf_cnuc_ha_filtered: gpd.GeoDataFrame, nome_uc: str, modo_valor: str = "percent") -> go.Figure:
    gdf_cnuc_ha = gdf_cnuc_ha_filtered.copy()
    if gdf_cnuc_ha.empty:
         return go.Figure()

    if nome_uc == "Todas":
        area_total = gdf_cnuc_ha["area_ha"].sum()
        area_car = gdf_cnuc_ha["sigef_ha"].sum()
    else:
        row = gdf_cnuc_ha[gdf_cnuc_ha["nome_uc"] == nome_uc]
        if row.empty:
            return go.Figure() 
            
        area_total = row["area_ha"].values[0]
        area_car = row["sigef_ha"].values[0]

    # Garantir que os valores sejam numéricos e não nulos
    area_total = float(area_total) if area_total and not pd.isna(area_total) else 0.0
    area_car = float(area_car) if area_car and not pd.isna(area_car) else 0.0
    
    # Cálculo do percentual - fórmula original mantida para consistência
    percentual = (area_car / area_total) * 100 if area_total > 0 else 0
    
    # Lógica original: usar área total como base, mesmo quando CAR > UC
    if area_car <= area_total:
        area_livre = area_total - area_car
        labels = ["Área CAR", "Área Livre"]
        values = [area_car, area_livre]
        colors = ["#2ca02c", "#d9d9d9"]
    else:
        # Quando CAR > UC, mostrar proporcionalmente
        area_livre = 0
        labels = ["Área CAR"]
        values = [100]  # 100% do gráfico
        colors = ["#2ca02c"]
    
    # Formatar valores para exibição
    area_total_fmt = format_number_with_dots(area_total, 0)
    area_car_fmt = format_number_with_dots(area_car, 0)
    percentual_fmt = f"{percentual:.1f}%".replace('.', ',')
    
    if modo_valor == "percent":
        textinfo = "label+percent"
        center_text = f"UC: {area_total_fmt} ha<br>CAR: {area_car_fmt} ha<br>({percentual_fmt})"
    else:
        textinfo = "label+value"
        center_text = f"UC: {area_total_fmt} ha<br>CAR: {area_car_fmt} ha"
        
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo=textinfo,
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title_text=f"Ocupação do CAR em: {nome_uc}",
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=14, showarrow=False)],
        height=400
    )
    return _apply_layout(fig, title=f"Ocupação do CAR em: {nome_uc}", title_size=16)

def fig_familias(df_conflitos_filtered: pd.DataFrame) -> go.Figure:
    df = df_conflitos_filtered.sort_values('Total_Famílias', ascending=False)
    if df.empty:
        return go.Figure()

    max_val = df['Total_Famílias'].max()
    
    # Formatar valores para exibição
    familias_text = [format_number_with_dots(val, 0) for val in df['Total_Famílias']]

    fig = px.bar(
        df,
        x='Total_Famílias',
        y='Município',
        orientation='h',
        text='Total_Famílias',
        labels={'Total_Famílias': 'Total de Famílias', 'Município': ''}
    )
    fig = _apply_layout(fig, title="Famílias Afetadas")

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            range=[0, max_val * 1.1],
            tickformat='~s'
        ),
        margin=dict(l=80, r=100, t=50, b=20) 
    )

    fig.update_traces(
        text=familias_text,
        textposition='outside',
        cliponaxis=False,                 
        marker_line_color='rgb(80,80,80)',
        marker_line_width=0.5,
        hovertemplate='<b>%{y}</b><br>Famílias: %{text}<extra></extra>',
        customdata=familias_text
    )

    return fig

def fig_conflitos(df_conflitos_filtered: pd.DataFrame) -> go.Figure:
    df = df_conflitos_filtered.sort_values('Número_Conflitos', ascending=False)
    if df.empty:
        return go.Figure() 

    # Formatar valores para exibição
    conflitos_text = [format_number_with_dots(val, 0) for val in df['Número_Conflitos']]

    fig = px.bar(
        df, x='Número_Conflitos', y='Município', orientation='h',
        text='Número_Conflitos'
    )
    fig = _apply_layout(fig, title="Conflitos Registrados")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickformat='~s')
    )
    fig.update_traces(
        text=conflitos_text,
        textposition='outside',
        marker_line_color='rgb(80,80,80)',
        marker_line_width=0.5,
        hovertemplate='<b>%{y}</b><br>Conflitos: %{text}<extra></extra>',
        customdata=conflitos_text
    )
    return fig
    
def clean_text(text: str) -> str:
    if pd.isna(text): return text
    text = str(text).strip().lower()
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def fig_justica(df_proc_filtered: pd.DataFrame) -> dict[str, go.Figure]:
    df_proc = df_proc_filtered.copy()
    figs = {}
    palette = px.defaults.color_discrete_sequence
    bottom_margin = 100

    mapa_classes = {
        "procedimento comum civel": "Proc. Comum Cível",
        "acao civil publica": "Ação Civil Pública",
        "peticao civel": "Petição Cível",
        "cumprimento de sentenca": "Cumpr. Sentença",
        "termo circunstanciado": "Termo Circunstan.",
        "carta precatoria civel": "Carta Prec. Cível",
        "acao penal - procedimento ordinario": "Ação Penal Ordinária",
        "alvara judicial - lei 6858/80": "Alvará Judicial",
        "crimes ambientais": "Crimes Ambientais",
        "homologacao da transacao extrajudicial": "Homolog. Transação"
    }

    mapa_assuntos = {
        "indenizacao por dano ambiental": "Dano Ambiental",
        "obrigacao de fazer / nao fazer": "Obrig. Fazer/Não Fazer",
        "flora": "Flora",
        "fauna": "Fauna",
        "mineracao": "Mineração",
        "poluicao": "Poluição",
        "unidade de conservacao da natureza": "Unid. Conservação",
        "revogacao/anulacao de multa ambiental": "Anulação Multa Ambiental",
        "area de preservacao permanente": "APP",
        "agrotoxicos": "Agrotóxicos"
    }

    mapa_orgaos = {
        "1a vara civel e empresarial de altamira": "1ª V. Cível Altamira",
        "vara civil e empresarial da comarca de sao felix do xingu": "V. Cível São Félix",
        "vara civel de novo progresso": "V. Cível Novo Progresso",
        "2a vara civel e empresarial de altamira": "2ª V. Cível Altamira",
        "3a vara civel e empresarial de altamira": "3ª V. Cível Altamira",
        "1a vara civel e empresarial de itaituba": "1ª V. Cível Itaituba",
        "juizado especial civel e criminal de itaituba": "JEC Itaituba",
        "2a vara civel e empresarial de itaituba": "2ª V. Cível Itaituba",
        "vara criminal de itaituba": "V. Criminal Itaituba",
        "vara unica de jacareacanga": "V. Única Jacareacanga"
    }

    # Top 10 Municípios
    if 'municipio' in df_proc.columns and not df_proc.empty:
        df_proc['municipio'] = df_proc['municipio'].apply(clean_text)
        top = df_proc['municipio'].value_counts().head(10).reset_index()
        top.columns = ['Municipio', 'Quantidade']
        if not top.empty:
            top['label'] = top['Municipio'].apply(lambda x: wrap_label(x, 20))
            fig_mun = px.bar(
                top, y='label', x='Quantidade', orientation='h',
                color_discrete_sequence=palette
            )
            # Formatar texto das barras
            quantidade_text = [format_number_with_dots(v, 0) for v in top['Quantidade']]
            fig_mun.update_traces(
                text=quantidade_text,
                textposition='auto', 
                cliponaxis=False,
                hovertemplate='<b>%{y}</b><br>Quantidade: %{text}<extra></extra>',
                customdata=quantidade_text
            )
            fig_mun.update_layout(
                margin=dict(l=150, r=60, t=50, b=bottom_margin),
                height=500,
                yaxis=dict(autorange="reversed")
            )
            figs['mun'] = _apply_layout(fig_mun, "Top 10 Municípios com Mais Processos", 16)
        else:
             figs['mun'] = go.Figure().update_layout(title="Top 10 Municípios com Mais Processos", annotations=[dict(text="Sem dados", showarrow=False)])
    else:
         figs['mun'] = go.Figure().update_layout(title="Top 10 Municípios com Mais Processos", annotations=[dict(text="Sem dados", showarrow=False)])


    # Evolução Mensal de Processos
    if 'data_ajuizamento' in df_proc.columns and not df_proc.empty:
        df_proc['ano_mes'] = (
            pd.to_datetime(df_proc['data_ajuizamento'], errors='coerce')
              .dt.to_period('M')
              .dt.to_timestamp()
        )
        mensal = df_proc.groupby('ano_mes', observed=False).size().reset_index(name='Quantidade')
        if not mensal.empty:
            fig_temp = px.line(
                mensal,
                x='ano_mes', y='Quantidade',
                markers=True, text='Quantidade'
            )
            fig_temp.update_traces(
                mode='lines+markers+text',
                textposition='top center',
                text=[format_number_with_dots(v, 0) for v in mensal['Quantidade']],
                hovertemplate='Mês: %{x}<br>Quantidade: %{text}<extra></extra>',
                customdata=[format_number_with_dots(v, 0) for v in mensal['Quantidade']]
            )
            fig_temp.update_layout(
                margin=dict(l=80, r=60, t=50, b=bottom_margin),
                height=400,
                yaxis=dict(range=[0, mensal['Quantidade'].max() * 1.1])
            )
            figs['temp'] = _apply_layout(fig_temp, "Evolução Mensal de Processos", 16)
        else:
             figs['temp'] = go.Figure().update_layout(title="Evolução Mensal de Processos", annotations=[dict(text="Sem dados", showarrow=False)])
    else:
         figs['temp'] = go.Figure().update_layout(title="Evolução Mensal de Processos", annotations=[dict(text="Sem dados", showarrow=False)])


    # Top 10 Classes, Assuntos e Órgãos
    mappings = [
        ('class', 'classe', 'Top 10 Classes Processuais', mapa_classes),
        ('ass', 'assuntos', 'Top 10 Assuntos', mapa_assuntos),
        ('org', 'orgao_julgador', 'Top 10 Órgãos Julgadores', mapa_orgaos)
    ]

    for key, col, title, mapa in mappings:
        if col in df_proc.columns and not df_proc.empty:
            series_de_strings_limpas = df_proc[col].apply(clean_text)
            series_categorica = pd.Series(series_de_strings_limpas, dtype="category")
            try:
                series_com_categorias_renomeadas = series_categorica.cat.rename_categories(mapa)
            except ValueError as e:
                print(f"Aviso para coluna '{col}': Não foi possível renomear todas as categorias usando o mapa fornecido. Verifique se as chaves do mapa correspondem às categorias existentes após clean_text. Erro: {e}")
                series_com_categorias_renomeadas = series_de_strings_limpas.replace(mapa)
                series_com_categorias_renomeadas = pd.Series(series_com_categorias_renomeadas, dtype="category")

            df = (
                series_com_categorias_renomeadas
                .value_counts()
                .head(10)
                .reset_index()
            )
            
            df.columns = [col, 'Quantidade']
            
            if not df.empty:
                df['label'] = df[col].apply(lambda x: wrap_label(x, 30))
                fig = px.bar(
                    df, y='label', x='Quantidade', orientation='h',
                    color_discrete_sequence=palette
                )
                # Formatar texto das barras
                quantidade_text = [format_number_with_dots(v, 0) for v in df['Quantidade']]
                fig.update_traces(
                    text=quantidade_text,
                    textposition='auto', 
                    cliponaxis=False,
                    hovertemplate='<b>%{y}</b><br>Quantidade: %{text}<extra></extra>',
                    customdata=quantidade_text
                )
                fig.update_layout(
                    margin=dict(l=180, r=60, t=50, b=bottom_margin),
                    height=500,
                    yaxis=dict(autorange="reversed")
                )
                figs[key] = _apply_layout(fig, title, 16)
            else:
                figs[key] = go.Figure().update_layout(title=title, annotations=[dict(text="Sem dados", showarrow=False)])
        else:
            figs[key] = go.Figure().update_layout(title=title, annotations=[dict(text="Sem dados", showarrow=False)])

    return figs 

def graficos_inpe(data_frame_entrada: pd.DataFrame, ano_selecionado_str: str, gdf_cnuc_raw: gpd.GeoDataFrame = None) -> dict[str, go.Figure]:
    df = data_frame_entrada.copy()
    def create_placeholder_fig(title_message: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=title_message,
            xaxis_visible=False,
            yaxis_visible=False,
            annotations=[dict(text="Não há dados suficientes para exibir este gráfico.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )
        return fig

    base_error_title = f"Período: {ano_selecionado_str}"

    if df.empty:
        return {
            'temporal': create_placeholder_fig(f"Evolução Temporal ({base_error_title})"),
            'top_risco': create_placeholder_fig(f"Top Risco ({base_error_title})"),
            'top_precip': create_placeholder_fig(f"Top Precipitação ({base_error_title})"),
            'mapa': create_placeholder_fig(f"Mapa de Focos ({base_error_title})")
        }

    # --- Gráfico de Evolução Temporal do Risco de Fogo ---
    fig_temp = create_placeholder_fig(f"Evolução Temporal do Risco de Fogo ({ano_selecionado_str})")
    if 'DataHora' in df.columns and 'RiscoFogo' in df.columns:
        df_temp_indexed = df.set_index('DataHora')
        df_risco_valido_temp = df_temp_indexed[df_temp_indexed['RiscoFogo'].between(0, 1)]
        if not df_risco_valido_temp.empty:
            monthly_risco = df_risco_valido_temp['RiscoFogo'].resample('ME').mean().reset_index()
            monthly_risco['RiscoFogo'] = monthly_risco['RiscoFogo'].fillna(0)

            if not monthly_risco.empty:
                fig_temp = go.Figure()
                fig_temp.add_trace(go.Scatter(
                    x=monthly_risco['DataHora'].dt.to_period('M').astype(str),
                    y=monthly_risco['RiscoFogo'],
                    name='Risco de Fogo Mensal',
                    mode='lines+markers+text',
                    marker=dict(size=8, color='#FF4136', line=dict(width=1, color='#444')),
                    line=dict(width=2, color='#FF4136'),
                    text=[f'{v:.2f}'.replace('.', ',') for v in monthly_risco['RiscoFogo']],
                    textposition='top center'
                ))
                fig_temp.update_layout(
                    title_text=f'Evolução Mensal do Risco de Fogo ({ano_selecionado_str})',
                    xaxis_title='Mês',
                    yaxis_title='Risco Médio de Fogo',
                    height=400,
                    margin=dict(l=60, r=80, t=80, b=40),
                    showlegend=True,
                    hovermode='x unified'
                )

    # --- Gráfico Top Municípios por Risco de Fogo ---
    fig_risco = create_placeholder_fig(f"Top Municípios - Risco de Fogo ({ano_selecionado_str})")
    if 'mun_corrigido' in df.columns and 'RiscoFogo' in df.columns:
        df_risco_valido = df[df['RiscoFogo'].between(0, 1)]
        if not df_risco_valido.empty:
            top_risco_data = df_risco_valido.groupby('mun_corrigido', observed=False)['RiscoFogo'].mean().nlargest(10).sort_values()
            if not top_risco_data.empty:
                # Formatar valores de risco com vírgula para decimais
                risco_text = [f"{v:.2f}".replace('.', ',') for v in top_risco_data.values]
                fig_risco = go.Figure(go.Bar(
                    y=top_risco_data.index,
                    x=top_risco_data.values,
                    orientation='h',
                    marker_color='#FF8C7A',
                    text=risco_text,
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Risco Médio: %{text}<extra></extra>',
                    customdata=risco_text
                ))
                fig_risco.update_layout(
                    title_text=f'Top Municípios por Risco Médio de Fogo ({ano_selecionado_str})',
                    xaxis_title='Risco Médio de Fogo',
                    yaxis_title='Município',
                    height=400,
                    margin=dict(l=100, r=80, t=50, b=40)
                )

    # --- Gráfico Top Municípios por Precipitação ---
    fig_precip = create_placeholder_fig(f"Top Municípios - Precipitação Média ({ano_selecionado_str})")
    if 'mun_corrigido' in df.columns and 'Precipitacao' in df.columns:
        df_precip_valida = df[df['Precipitacao'] >= 0]
        if not df_precip_valida.empty:
            top_precip_data = df_precip_valida.groupby('mun_corrigido', observed=False)['Precipitacao'].mean().nlargest(10).sort_values()
            if not top_precip_data.empty:
                # Formatar valores de precipitação com mais casas decimais
                precip_text = [f"{format_number_with_dots(v, 2)} mm" for v in top_precip_data.values]
                fig_precip = go.Figure(go.Bar(
                    y=top_precip_data.index,
                    x=top_precip_data.values,
                    orientation='h',
                    marker_color='#B3D9FF',
                    text=precip_text,
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Precipitação Média: %{text}<extra></extra>',
                    customdata=precip_text
                ))
                fig_precip.update_layout(
                    title_text=f'Top Municípios por Precipitação Média ({ano_selecionado_str})',
                    xaxis_title='Precipitação Média (mm)',
                    yaxis_title='Município',
                    height=400,
                    margin=dict(l=100, r=120, t=50, b=40)  # Aumentar margem direita para evitar corte dos "mm"
                )

    # --- Mapa de Distribuição dos Focos de Calor ---
    fig_map = create_placeholder_fig(f"Mapa de Distribuição dos Focos de Calor ({ano_selecionado_str})")
    map_required_cols = ['Latitude', 'Longitude', 'RiscoFogo', 'mun_corrigido', 'DataHora']
    if all(col in df.columns for col in map_required_cols):
        df_map_plot = df[map_required_cols + (['Precipitacao'] if 'Precipitacao' in df.columns else [])].copy()
        df_map_plot.dropna(subset=['Latitude', 'Longitude', 'RiscoFogo', 'mun_corrigido'], inplace=True)
        df_map_plot = df_map_plot[df_map_plot['RiscoFogo'].between(0, 1)]
        if 'Precipitacao' in df_map_plot.columns:
             df_map_plot = df_map_plot[df_map_plot['Precipitacao'] >= 0]
        else:
            df_map_plot['Precipitacao'] = 0

        if not df_map_plot.empty:
            sample_size = 50000
            if len(df_map_plot) > sample_size:
                df_map_plot_sampled = df_map_plot.sample(sample_size, random_state=1)
            else:
                df_map_plot_sampled = df_map_plot

            if not df_map_plot_sampled.empty:
                centro_map = {
                    'lat': df_map_plot_sampled['Latitude'].mean(),
                    'lon': df_map_plot_sampled['Longitude'].mean()
                }
                lat_range = df_map_plot_sampled['Latitude'].max() - df_map_plot_sampled['Latitude'].min()
                lon_range = df_map_plot_sampled['Longitude'].max() - df_map_plot_sampled['Longitude'].min()
                max_range = max(lat_range, lon_range, 0.01)

                zoom_level = 3.5
                if max_range < 1: zoom_level = 7
                elif max_range < 5: zoom_level = 5
                elif max_range < 10: zoom_level = 4
                fig_map = go.Figure()
                if gdf_cnuc_raw is not None and not gdf_cnuc_raw.empty:
                    gdf_cnuc_geo = gdf_cnuc_raw.to_crs("EPSG:4326")
                    
                    for idx, row in gdf_cnuc_geo.iterrows():
                        if row.geometry and hasattr(row.geometry, 'exterior'):
                            if row.geometry.geom_type == 'Polygon':
                                coords = list(row.geometry.exterior.coords)
                                lons, lats = zip(*coords)
                                
                                fig_map.add_trace(go.Scattermapbox(
                                    lon=lons,
                                    lat=lats,
                                    mode='lines',
                                    fill='toself',
                                    fillcolor='rgba(34,139,34,0.2)',
                                    line=dict(color='rgba(34,139,34,0.8)', width=1),
                                    name='Unidades de Conservação',
                                    showlegend=False,  
                                    hovertemplate=f"<b>{row.get('nome_uc', 'UC')}</b><extra></extra>",
                                    text=row.get('nome_uc', 'UC')
                                ))
                        elif row.geometry and row.geometry.geom_type == 'MultiPolygon':
                            for poly in row.geometry.geoms:
                                coords = list(poly.exterior.coords)
                                lons, lats = zip(*coords)
                                
                                fig_map.add_trace(go.Scattermapbox(
                                    lon=lons,
                                    lat=lats,
                                    mode='lines',
                                    fill='toself',
                                    fillcolor='rgba(34,139,34,0.2)',
                                    line=dict(color='rgba(34,139,34,0.8)', width=1),
                                    name='Unidades de Conservação',
                                    showlegend=False,
                                    hovertemplate=f"<b>{row.get('nome_uc', 'UC')}</b><extra></extra>",
                                    text=row.get('nome_uc', 'UC')
                                ))

                fig_map.add_trace(go.Scattermapbox(
                    lat=df_map_plot_sampled['Latitude'],
                    lon=df_map_plot_sampled['Longitude'],
                    mode='markers',
                    marker=dict(
                        size=df_map_plot_sampled['Precipitacao'] / 10 + 3,  
                        color=df_map_plot_sampled['RiscoFogo'],
                        colorscale='YlOrRd',
                        showscale=False,  
                        sizemin=3,
                        opacity=0.7
                    ),
                    text=df_map_plot_sampled['mun_corrigido'],
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "Risco de Fogo: %{marker.color:.2f}<br>" +
                        "Precipitação: %{marker.size:.1f} mm<br>" +
                        "<extra></extra>"
                    ),
                    name='Focos de Calor',
                    showlegend=False 
                ))

                fig_map.update_layout(
                    title_text=f'Mapa de Distribuição dos Focos de Calor ({ano_selecionado_str})',
                    mapbox=dict(
                        style='open-street-map',
                        zoom=zoom_level,
                        center=centro_map
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False 
                )

    return {
        'temporal': fig_temp,
        'top_risco': fig_risco,
        'top_precip': fig_precip,
        'mapa': fig_map
    }

def fig_focos_calor_por_uc(df_inpe: pd.DataFrame, gdf_cnuc: gpd.GeoDataFrame) -> go.Figure:
    """
    Cria um gráfico de barras mostrando a quantidade de focos de calor por UC.
    """
    if df_inpe.empty or gdf_cnuc.empty:
        return go.Figure()
    
    try:
        from shapely.geometry import Point
    
        df_valid = df_inpe.dropna(subset=['Latitude', 'Longitude']).copy()
        if df_valid.empty:
            return go.Figure()
        geometry = [Point(lon, lat) for lon, lat in zip(df_valid['Longitude'], df_valid['Latitude'])]
        gdf_focos = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
        crs_proj = "EPSG:31983"
        gdf_focos_proj = gdf_focos.to_crs(crs_proj)
        gdf_cnuc_proj = gdf_cnuc.to_crs(crs_proj)
        focos_in_ucs = gpd.sjoin(gdf_focos_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
        
        if focos_in_ucs.empty:
            return go.Figure()
        focos_por_uc = focos_in_ucs.groupby('nome_uc', observed=False).size().reset_index(name='quantidade_focos')
        focos_por_uc = focos_por_uc.sort_values('quantidade_focos', ascending=False).head(10)
        
        focos_por_uc['uc_wrap'] = focos_por_uc['nome_uc'].apply(lambda x: wrap_label(x, 15))
        
        focos_text = [format_number_with_dots(val, 0) for val in focos_por_uc['quantidade_focos']]
        
        fig = px.bar(
            focos_por_uc,
            x='uc_wrap',
            y='quantidade_focos',
            labels={"quantidade_focos": "Quantidade de Focos", "uc_wrap": "UC"},
            color='quantidade_focos',
            color_continuous_scale='Reds'
        )
        
        fig.update_traces(
            text=focos_text,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Focos de Calor: %{text}<extra></extra>',
            customdata=focos_text
        )
        
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), title_text="")  
        fig.update_yaxes(title_text="Quantidade de Focos", tickfont=dict(size=9))
        fig.update_layout(
            height=500,
            margin=dict(l=80, r=80, t=80, b=120),  
            showlegend=False
        )
        
        return _apply_layout(fig, title="Focos de Calor por UC", title_size=16)
        
    except Exception as e:
        st.warning(f"Erro ao processar focos de calor por UC: {e}")
        return go.Figure()

def mostrar_tabela_unificada(gdf_alertas_filtered, gdf_sigef_filtered, gdf_cnuc_filtered):
    df_a = gdf_alertas_filtered[['MUNICIPIO', 'AREAHA']].rename(columns={'MUNICIPIO':'municipio', 'AREAHA':'alerta_ha'})
    df_c = gdf_cnuc_filtered[['municipio', 'ha_total']].rename(columns={'ha_total':'uc_ha'}) 

    df_a['alerta_ha'] = pd.to_numeric(df_a['alerta_ha'], errors='coerce').fillna(0)
    df_c['uc_ha'] = pd.to_numeric(df_c['uc_ha'], errors='coerce').fillna(0)

    df_alertas_mun = df_a.groupby('municipio', observed=True, as_index=False)['alerta_ha'].sum()
    df_cnuc_mun = df_c.groupby('municipio', observed=True, as_index=False)['uc_ha'].sum()

    df_merged = df_alertas_mun.merge(df_cnuc_mun, on='municipio', how='outer').fillna(0)

    cols = ['alerta_ha', 'uc_ha']
    for c in cols:
        df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(0)
    
    total_alertas = df_merged['alerta_ha'].sum()
    total_uc = df_merged['uc_ha'].sum()

    df_merged = df_merged[~((df_merged[cols] == 0).all(axis=1))]
    df_merged = df_merged.sort_values('municipio').reset_index(drop=True)
    df_merged = df_merged.rename(columns={
        'municipio': 'MUNICÍPIO',
        'alerta_ha': 'ALERTAS(HA)',
        'uc_ha': 'CNUC(HA)'
    })

    total_row = pd.DataFrame([{
        'MUNICÍPIO': 'TOTAL(HA)',
        'ALERTAS(HA)': total_alertas,
        'CNUC(HA)': total_uc
    }])
    
    df_merged = pd.concat([df_merged, total_row], ignore_index=True)

    styles = []
    colors = {
        'ALERTAS(HA)':'#fde0dd', 
        'CNUC(HA)':'#edf8e9'
    }
    for i, c in enumerate(df_merged.columns):
        if c in colors:
            styles.append({'selector': f'td.col{i}', 'props': [('background-color', colors[c])]})
    
    styles.append({
        'selector': 'tr:last-child',
        'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]
    })

    def format_area_with_dots(val):
        if pd.isna(val):
            return ""
        return format_number_with_dots(val, 2)

    styled = (
        df_merged.style
                 .format({c: format_area_with_dots for c in ['ALERTAS(HA)', 'CNUC(HA)']})
                 .set_table_styles(styles)
                 .set_table_attributes('style="border-collapse:collapse"')
    )

    st.subheader("Tabela Área")
    st.markdown(styled.to_html(), unsafe_allow_html=True)

def fig_desmatamento_uc(gdf_cnuc_filtered: gpd.GeoDataFrame, gdf_alertas_filtered: gpd.GeoDataFrame) -> go.Figure:
    if gdf_cnuc_filtered.empty or gdf_alertas_filtered.empty:
        return go.Figure() 

    crs_proj = "EPSG:31983" 
    gdf_cnuc_proj = gdf_cnuc_filtered.to_crs(crs_proj)
    gdf_alertas_proj = gdf_alertas_filtered.to_crs(crs_proj)

    if not gdf_alertas_proj.empty and not gdf_cnuc_proj.empty:
        alerts_in_ucs = gpd.sjoin(gdf_alertas_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
    else:
        alerts_in_ucs = gpd.GeoDataFrame()


    if alerts_in_ucs.empty:
         return go.Figure() 

    alert_area_per_uc = alerts_in_ucs.groupby('nome_uc', observed=False)['AREAHA'].sum().reset_index()
    alert_area_per_uc.columns = ['nome_uc', 'alerta_ha_total'] 

    alert_area_per_uc = alert_area_per_uc.sort_values('alerta_ha_total', ascending=False)

    alert_area_per_uc['uc_wrap'] = alert_area_per_uc['nome_uc'].apply(lambda x: wrap_label(x, 15)) 

    fig = px.bar(
        alert_area_per_uc,
        x='uc_wrap',
        y='alerta_ha_total',
        labels={"alerta_ha_total":"Área de Alertas (ha)","uc_wrap":"UC"},
        text_auto=True,
    )

    # Formatar valores para exibição
    alerta_text = [format_number_with_dots(val, 0) for val in alert_area_per_uc['alerta_ha_total']]

    fig.update_traces(
        customdata=np.stack([alerta_text, alert_area_per_uc.nome_uc], axis=-1),
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Área de Alertas: %{customdata[0]} ha<extra></extra>" 
        ),
        text=alerta_text, 
        textposition="outside", 
        marker_line_color="rgb(80,80,80)",
        marker_line_width=0.5,
        cliponaxis=False  # Permite que o texto apareça fora dos limites do gráfico
    )

    max_val = alert_area_per_uc["alerta_ha_total"].max()

    fig.update_xaxes(tickangle=0, tickfont=dict(size=9), title_text="")
    fig.update_yaxes(
        title_text="Área (ha)", 
        tickfont=dict(size=9),
        range=[0, max_val * 1.2]  
    )
    fig.update_layout(
        height=450,  
        margin=dict(l=80, r=80, t=100, b=80),  
        showlegend=False
    ) 

    fig = _apply_layout(fig, title="Área de Alertas (Desmatamento) por UC", title_size=16)

    return fig

def fig_desmatamento_temporal(gdf_alertas_filtered: gpd.GeoDataFrame) -> go.Figure:
    """Cria um gráfico de linha mostrando a evolução temporal da área de alertas de desmatamento."""
    if gdf_alertas_filtered.empty or 'DATADETEC' not in gdf_alertas_filtered.columns:
        fig = go.Figure()
        fig.update_layout(title="Evolução Temporal de Alertas (Desmatamento)",
                          xaxis_title="Data", yaxis_title="Área (ha)")
        return _apply_layout(fig, title="Evolução Temporal de Alertas (Desmatamento)", title_size=16)

    gdf_alertas_filtered['DATADETEC'] = pd.to_datetime(gdf_alertas_filtered['DATADETEC'], errors='coerce')
    gdf_alertas_filtered['AREAHA'] = pd.to_numeric(gdf_alertas_filtered['AREAHA'], errors='coerce')

    df_valid_dates = gdf_alertas_filtered.dropna(subset=['DATADETEC', 'AREAHA'])

    if df_valid_dates.empty:
         fig = go.Figure()
         fig.update_layout(title="Evolução Temporal de Alertas (Desmatamento)",
                          xaxis_title="Data", yaxis_title="Área (ha)")
         return _apply_layout(fig, title="Evolução Temporal de Alertas (Desmatamento)", title_size=16)

    df_monthly = df_valid_dates.set_index('DATADETEC').resample('ME')['AREAHA'].sum().reset_index()
    df_monthly['DATADETEC'] = df_monthly['DATADETEC'].dt.to_period('M').astype(str)

    fig = px.line(
        df_monthly,
        x='DATADETEC',
        y='AREAHA',
        labels={"AREAHA":"Área (ha)","DATADETEC":"Mês/Ano"},
        markers=True,
        text='AREAHA'
    )

    area_text = [format_number_with_dots(val, 0) for val in df_monthly['AREAHA']]

    fig.update_traces(
        mode='lines+markers+text',
        textposition='top center',
        text=area_text,
        hovertemplate=(
            "Mês/Ano: %{x}<br>"
            "Área de Alertas: %{text} ha<extra></extra>"
        ),
        customdata=area_text
    )

    fig.update_xaxes(title_text="Mês/Ano", tickangle=45)
    fig.update_yaxes(title_text="Área (ha)")
    fig.update_layout(height=400)

    fig = _apply_layout(fig, title="Evolução Mensal de Alertas (Desmatamento)", title_size=16)

    return fig

def fig_desmatamento_municipio(gdf_alertas_filtered: gpd.GeoDataFrame) -> go.Figure:
    """Cria um gráfico de barras mostrando a área total de alertas de desmatamento por município."""
    df = gdf_alertas_filtered.sort_values('AREAHA', ascending=False)
    if df.empty:
        return go.Figure()

    area_text = [format_number_with_dots(val, 0) for val in df['AREAHA']]

    fig = px.bar(
        df,
        x='AREAHA',
        y='MUNICIPIO',
        orientation='h',
        text='AREAHA',
        labels={'AREAHA': 'Área (ha)', 'MUNICIPIO': ''}
    )
    fig = _apply_layout(fig, title="Desmatamento por Município")

    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis=dict(
            tickformat='~s'                 
        ),
        margin=dict(l=80, r=100, t=50, b=20) 
    )

    fig.update_traces(
        text=area_text,
        textposition='outside',
        cliponaxis=False,                 
        marker_line_color='rgb(80,80,80)',
        marker_line_width=0.5,
        hovertemplate='<b>%{y}</b><br>Área: %{text} ha<extra></extra>',
        customdata=area_text
    )

    return fig

def fig_desmatamento_mapa_pontos(gdf_alertas_filtered: gpd.GeoDataFrame) -> go.Figure:
    """Cria um mapa de dispersão dos alertas de desmatamento."""
    if gdf_alertas_filtered.empty or 'AREAHA' not in gdf_alertas_filtered.columns or 'geometry' not in gdf_alertas_filtered.columns:
        fig = go.Figure()
        fig.update_layout(title="Mapa de Alertas (Desmatamento)")
        return _apply_layout(fig, title="Mapa de Alertas (Desmatamento)", title_size=16)

    gdf_alertas_filtered['AREAHA'] = pd.to_numeric(gdf_alertas_filtered['AREAHA'], errors='coerce')

    try:
        gdf_proj = gdf_alertas_filtered.to_crs("EPSG:31983").copy()
        centroids_proj = gdf_proj.geometry.centroid
        centroids_geo = centroids_proj.to_crs("EPSG:4326")

        gdf_map = gdf_alertas_filtered.to_crs("EPSG:4326").copy()
        gdf_map['Latitude'] = centroids_geo.y
        gdf_map['Longitude'] = centroids_geo.x

    except Exception as e:
        st.warning(f"Could not calculate or reproject centroids for map: {e}. Skipping map.")
        fig = go.Figure()
        fig.update_layout(title="Mapa de Alertas (Desmatamento)")
        return _apply_layout(fig, title="Mapa de Alertas (Desmatamento)", title_size=16)

    gdf_map = gdf_map.dropna(subset=['Latitude', 'Longitude'])

    if gdf_map.empty:
        fig = go.Figure()
        fig.update_layout(title="Mapa de Alertas (Desmatamento)")
        return _apply_layout(fig, title="Mapa de Alertas (Desmatamento)", title_size=16)

    minx, miny, maxx, maxy = gdf_map.total_bounds
    center = {'lat': (miny + maxy) / 2, 'lon': (minx + maxx) / 2}
    span_lat = maxy - miny
    lon_range = maxx - minx
    max_range = max(span_lat, lon_range, 0.01)

    zoom_level = 3.5
    if max_range < 0.1: zoom_level = 10
    elif max_range < 0.5: zoom_level = 8
    elif max_range < 1: zoom_level = 7
    elif max_range < 5: zoom_level = 5
    elif max_range < 10: zoom_level = 4
    elif max_range < 20: zoom_level = 3.5
    zoom_level = int(round(zoom_level))

    sample_size = 50000
    if len(gdf_map) > sample_size:
        gdf_map_plot = gdf_map.sample(sample_size, random_state=1)
    else:
        gdf_map_plot = gdf_map

    if gdf_map_plot.empty:
        fig = go.Figure()
        fig.update_layout(title="Mapa de Alertas (Desmatamento)")
        return _apply_layout(fig, title="Mapa de Alertas (Desmatamento)", title_size=16)

    fig = px.scatter_map(
        gdf_map_plot,
        lat='Latitude',
        lon='Longitude',
        size='AREAHA',
        color='AREAHA',
        color_continuous_scale="Reds",
        range_color=(0, gdf_map_plot['AREAHA'].quantile(0.95)),
        hover_name='CODEALERTA',
        hover_data={
            'AREAHA': ':.2f ha',
            'MUNICIPIO': True if 'MUNICIPIO' in gdf_map_plot.columns else False,
            'DATADETEC': True if 'DATADETEC' in gdf_map_plot.columns else False,
            'Latitude': False,
            'Longitude': False
        },
        size_max=15,
        zoom=zoom_level,
        center=center,
        opacity=0.7,
        map_style='open-street-map' 
    )

    fig.update_traces(showlegend=False)
    fig.update_coloraxes(showscale=False)  

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            zoom=zoom_level,
            center=center
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        hovermode='closest',
        showlegend=False  
    )
    
    fig.update_mapboxes(style='open-street-map')

    fig = _apply_layout(fig, title="Distribuição Espacial de Alertas (Desmatamento)", title_size=16)

    return fig

import gc
import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import warnings

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

DB_CONFIG = {
    'host': 'dataiesb.iesbtech.com.br',
    'database': '2312120036_Joel',
    'user': '2312120036_Joel',
    'password': '2312120036_Joel',
    'port': '5432',
    'schema': 'CPT',
    'table': 'queimadas'
}

CHUNK_SIZE = 15000 
MEMORY_THRESHOLD = 85  

class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        return (f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
                f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    def get_engine(self):
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self._connection_string,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
            except Exception:
                return None
        return self._engine
    
    def dispose(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None

class DataProcessor:
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self._base_filters = [
            "riscofogo BETWEEN 0 AND 1",
            "precipitacao >= 0",
            "diasemchuva >= 0",
            "latitude BETWEEN -15 AND 5",
            "longitude BETWEEN -60 AND -45"
        ]
    
    def _check_memory_usage(self) -> bool:
        """Verifica uso de memória"""
        return psutil.virtual_memory().percent < MEMORY_THRESHOLD
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
        
        int_cols = df.select_dtypes(include=['int64']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
        
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if col != 'DataHora' and df[col].nunique() / len(df) < 0.4:
                df[col] = df[col].astype('category')
        
        return df
    
    def _get_row_count(self, engine, where_clause: str) -> int:
        try:
            count_query = text(f"""
                SELECT COUNT(*) 
                FROM "{DB_CONFIG['schema']}"."{DB_CONFIG['table']}"
                WHERE {where_clause}
            """)
            
            with engine.connect() as conn:
                result = conn.execute(count_query)
                return result.scalar() or 0
        except Exception:
            return 0
    
    def _build_base_query(self) -> str:
        return f"""
            SELECT
                datahora,
                riscofogo,
                precipitacao,
                mun_corrigido,
                diasemchuva,
                latitude,
                longitude
            FROM "{DB_CONFIG['schema']}"."{DB_CONFIG['table']}"
        """
    
    def _load_data_chunks(self, engine, base_query: str, where_clause: str, 
                         total_rows: int) -> Optional[pd.DataFrame]:
        chunks = []
        
        try:
            for offset in range(0, total_rows, CHUNK_SIZE):
                if not self._check_memory_usage():
                    gc.collect()
                    if not self._check_memory_usage():
                        break
                
                chunk_query = text(f"""
                    {base_query}
                    WHERE {where_clause}
                    LIMIT {CHUNK_SIZE} OFFSET {offset}
                """)
                
                chunk_df = pd.read_sql(chunk_query, engine, parse_dates=['datahora'])
                chunk_df = self._optimize_dataframe(chunk_df)
                chunks.append(chunk_df)
                
                del chunk_df
                gc.collect()
            
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
                return df
            
        except Exception:
            pass
        
        return None
    
    def load_inpe_data(self, year: Optional[int] = None) -> Optional[pd.DataFrame]:
        engine = self.db_manager.get_engine()
        if not engine:
            return None
        
        try:
            filters = self._base_filters.copy()
            if year is not None:
                filters.append(f"EXTRACT(YEAR FROM datahora) = {year}")
            where_clause = " AND ".join(filters)
            
            total_rows = self._get_row_count(engine, where_clause)
            if total_rows == 0:
                return pd.DataFrame()
            
            base_query = self._build_base_query()
            
            if total_rows <= CHUNK_SIZE:
                query = text(f"{base_query} WHERE {where_clause}")
                df = pd.read_sql(query, engine, parse_dates=['datahora'])
            else:
                df = self._load_data_chunks(engine, base_query, where_clause, total_rows)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                'datahora': 'DataHora',
                'riscofogo': 'RiscoFogo',
                'precipitacao': 'Precipitacao',
                'mun_corrigido': 'mun_corrigido',
                'diasemchuva': 'DiaSemChuva',
                'latitude': 'Latitude',
                'longitude': 'Longitude'
            })
            
            df = self._optimize_dataframe(df)
            df = df.dropna(subset=['DataHora', 'mun_corrigido'])
            
            gc.collect()
            return df
            
        except Exception:
            return None
        finally:
            self.db_manager.dispose()
    
    def get_available_years(self) -> List[int]:
        engine = self.db_manager.get_engine()
        if not engine:
            return []
        
        try:
            query = text(f"""
                SELECT DISTINCT EXTRACT(YEAR FROM datahora) AS year
                FROM "{DB_CONFIG['schema']}"."{DB_CONFIG['table']}"
                WHERE datahora IS NOT NULL
                ORDER BY year
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query)
                years = [int(row[0]) for row in result.fetchall() if row[0] is not None]
            
            return years
            
        except Exception:
            return []
        finally:
            self.db_manager.dispose()

class RankingProcessor:
    
    @staticmethod
    def _process_chunk_aggregation(chunk: pd.DataFrame, theme: str) -> pd.DataFrame:
        chunk_clean = chunk.dropna(subset=['mun_corrigido']).copy()
        
        agg_configs = {
            "Maior Risco de Fogo": {
                'RiscoFogo': ['mean', 'max', 'count'],
                'DataHora': ['min', 'max']
            },
            "Maior Precipitação (evento)": {
                'Precipitacao': ['mean', 'max', 'sum', 'count'],
                'DataHora': ['min', 'max']
            },
            "Máx. Dias Sem Chuva": {
                'DiaSemChuva': ['mean', 'max', 'count'],
                'DataHora': ['min', 'max']
            }
        }
        
        if theme in agg_configs:
            return chunk_clean.groupby('mun_corrigido', observed=True).agg(agg_configs[theme])
        
        return pd.DataFrame()
    
    @staticmethod
    def _combine_chunk_results(results: List[pd.DataFrame], theme: str) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()
        
        combine_configs = {
            "Maior Risco de Fogo": {
                ('RiscoFogo', 'mean'): 'mean',
                ('RiscoFogo', 'max'): 'max',
                ('RiscoFogo', 'count'): 'sum',
                ('DataHora', 'min'): 'min',
                ('DataHora', 'max'): 'max'
            },
            "Maior Precipitação (evento)": {
                ('Precipitacao', 'mean'): 'mean',
                ('Precipitacao', 'max'): 'max',
                ('Precipitacao', 'sum'): 'sum',
                ('Precipitacao', 'count'): 'sum',
                ('DataHora', 'min'): 'min',
                ('DataHora', 'max'): 'max'
            },
            "Máx. Dias Sem Chuva": {
                ('DiaSemChuva', 'mean'): 'mean',
                ('DiaSemChuva', 'max'): 'max',
                ('DiaSemChuva', 'count'): 'sum',
                ('DataHora', 'min'): 'min',
                ('DataHora', 'max'): 'max'
            }
        }
        
        if theme in combine_configs:
            return pd.concat(results).groupby(level=0, observed=True).agg(combine_configs[theme])
        
        return pd.DataFrame()
    
    @staticmethod
    def _format_ranking_result(df_agg: pd.DataFrame, theme: str) -> Tuple[pd.DataFrame, str]:
        if df_agg.empty:
            return pd.DataFrame(), ''
        
        formatters = {
            "Maior Risco de Fogo": (
                RankingProcessor._format_fire_risk_ranking,
                'Risco Médio'
            ),
            "Maior Precipitação (evento)": (
                RankingProcessor._format_precipitation_ranking,
                'Precipitação Máxima (mm)'
            ),
            "Máx. Dias Sem Chuva": (
                RankingProcessor._format_dry_days_ranking,
                'Máx. Dias Sem Chuva'
            )
        }
        
        if theme in formatters:
            formatter_func, col_name = formatters[theme]
            df_rank = formatter_func(df_agg)
            
            if not df_rank.empty:
                df_rank.insert(0, 'Posição', range(1, len(df_rank) + 1))
            
            return df_rank, col_name
        
        return pd.DataFrame(), ''
    
    @staticmethod
    def _format_fire_risk_ranking(df_agg: pd.DataFrame) -> pd.DataFrame:
        df_agg = df_agg.round(4)
        df_rank = df_agg.nlargest(20, ('RiscoFogo', 'mean')).reset_index()
        
        df_rank.columns = ['Município', 'Risco Médio', 'Risco Máximo', 'Nº Registros', 
                           'Primeira Ocorrência', 'Última Ocorrência']
        
        df_rank['Primeira Ocorrência'] = pd.to_datetime(df_rank['Primeira Ocorrência']).dt.strftime('%d/%m/%Y')
        df_rank['Última Ocorrência'] = pd.to_datetime(df_rank['Última Ocorrência']).dt.strftime('%d/%m/%Y')
        
        return df_rank
    
    @staticmethod
    def _format_precipitation_ranking(df_agg: pd.DataFrame) -> pd.DataFrame:
        df_agg = df_agg.round(2)
        df_rank = df_agg.nlargest(20, ('Precipitacao', 'max')).reset_index()
        
        df_rank.columns = ['Município', 'Precipitação Máxima (mm)', 'Precipitação Média (mm)',
                           'Precipitação Total (mm)', 'Nº Registros', 'Primeira Ocorrência', 
                           'Última Ocorrência']
        
        df_rank['Primeira Ocorrência'] = pd.to_datetime(df_rank['Primeira Ocorrência']).dt.strftime('%d/%m/%Y')
        df_rank['Última Ocorrência'] = pd.to_datetime(df_rank['Última Ocorrência']).dt.strftime('%d/%m/%Y')
        
        return df_rank
    
    @staticmethod
    def _format_dry_days_ranking(df_agg: pd.DataFrame) -> pd.DataFrame:
        df_agg = df_agg.round(1)
        df_rank = df_agg.nlargest(20, ('DiaSemChuva', 'max')).reset_index()
        
        df_rank.columns = ['Município', 'Máx. Dias Sem Chuva', 'Média Dias Sem Chuva',
                           'Nº Registros', 'Primeira Ocorrência', 'Última Ocorrência']
        
        df_rank['Primeira Ocorrência'] = pd.to_datetime(df_rank['Primeira Ocorrência']).dt.strftime('%d/%m/%Y')
        df_rank['Última Ocorrência'] = pd.to_datetime(df_rank['Última Ocorrência']).dt.strftime('%d/%m/%Y')
        
        return df_rank
    
    def process_ranking(self, df: pd.DataFrame, theme: str, period: str) -> Tuple[pd.DataFrame, str]:
        if df is None or df.empty:
            return pd.DataFrame(), ''
        
        try:
            if len(df) > CHUNK_SIZE:
                chunks = [df[i:i + CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
                results = []
                
                for chunk in chunks:
                    chunk_result = self._process_chunk_aggregation(chunk, theme)
                    if not chunk_result.empty:
                        results.append(chunk_result)
                    
                    del chunk
                    gc.collect()
                
                df_agg = self._combine_chunk_results(results, theme)
                del results
                gc.collect()
            else:
                df_agg = self._process_chunk_aggregation(df, theme)
            
            df_rank, col_ord = self._format_ranking_result(df_agg, theme)
            
            del df_agg
            gc.collect()
            
            return df_rank, col_ord
            
        except Exception:
            return pd.DataFrame(), ''

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def get_cached_data_optimized(year: Optional[int] = None) -> Optional[pd.DataFrame]:
    processor = DataProcessor()
    original_query = processor._build_base_query
    
    def optimized_query():
        return f"""
            SELECT
                datahora,
                riscofogo,
                precipitacao,
                mun_corrigido,
                diasemchuva,
                latitude,
                longitude
            FROM "{DB_CONFIG['schema']}"."{DB_CONFIG['table']}"
        """
    
    processor._build_base_query = optimized_query
    
    try:
        df_full = processor.load_inpe_data(year)
        
        if df_full is None or df_full.empty:
            return pd.DataFrame()
        if len(df_full) > 50000:
            df_sample = df_full.groupby('mun_corrigido', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(10, len(x) // 10)), random_state=42)
                if len(x) > 10 else x
            ).reset_index(drop=True)
            
            return df_sample
        else:
            return df_full
            
    except Exception as e:
        print(f"Erro no carregamento otimizado: {e}")
        return pd.DataFrame()
    finally:
        processor._build_base_query = original_query

@st.cache_data(ttl=7200, show_spinner=False, max_entries=1)
def get_summary_stats() -> dict:
    """
    Carrega estatísticas resumidas para exibição rápida inicial.
    """
    try:
        processor = DataProcessor()
        engine = processor.db_manager.get_engine()
        if not engine:
            return {}
        stats_query = text(f"""
            SELECT 
                COUNT(*) as total_registros,
                COUNT(DISTINCT mun_corrigido) as total_municipios,
                AVG(riscofogo) as risco_medio,
                AVG(precipitacao) as precip_media,
                MIN(datahora) as data_inicio,
                MAX(datahora) as data_fim
            FROM "{DB_CONFIG['schema']}"."{DB_CONFIG['table']}"
            WHERE riscofogo BETWEEN 0 AND 1
            AND precipitacao >= 0
            AND diasemchuva >= 0
            AND latitude BETWEEN -15 AND 5
            AND longitude BETWEEN -60 AND -45
        """)
        
        with engine.connect() as conn:
            result = conn.execute(stats_query).fetchone()
            
            if result:
                return {
                    'total_registros': result[0] or 0,
                    'total_municipios': result[1] or 0,
                    'risco_medio': result[2] or 0,
                    'precip_media': result[3] or 0,
                    'data_inicio': result[4],
                    'data_fim': result[5]
                }
        return {}
    except Exception:
        return {}

@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def get_cached_data_filtered_by_uc(year: Optional[int] = None) -> Optional[pd.DataFrame]:
    processor = DataProcessor()
    df_full = processor.load_inpe_data(year)
    
    if df_full is None or df_full.empty:
        return pd.DataFrame()
    
    try:
        gdf_focos = gpd.GeoDataFrame(
            df_full,
            geometry=gpd.points_from_xy(df_full['Longitude'], df_full['Latitude']),
            crs='EPSG:4326'
        )
        
        gdf_ucs_reproj = gdf_cnuc_raw.to_crs('EPSG:4326')
        gdf_ucs_buffer = gdf_ucs_reproj.copy()
        gdf_ucs_buffer['geometry'] = gdf_ucs_buffer.geometry.buffer(0.01)
        focos_nas_ucs = gpd.sjoin(gdf_focos, gdf_ucs_buffer, how='inner', predicate='intersects')
        cols_originais = df_full.columns.tolist()
        df_filtrado = focos_nas_ucs[cols_originais].copy()
        df_filtrado = df_filtrado.drop_duplicates()
        
        return df_filtrado
        
    except Exception as e:
        print(f"Erro na filtragem espacial: {e}")
        return df_full

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def get_cached_data(year: Optional[int] = None) -> Optional[pd.DataFrame]:
    processor = DataProcessor()
    return processor.load_inpe_data(year)

@st.cache_data(ttl=3600, show_spinner=False, max_entries=1)
def get_available_years() -> List[int]:
    processor = DataProcessor()
    return processor.get_available_years()

@st.cache_data(ttl=1800, show_spinner=False, max_entries=5)
def get_cached_ranking(df_hash: str, theme: str, period: str) -> Tuple[pd.DataFrame, str]:
    parts = df_hash.split('_')
    if len(parts) >= 2:
        year_option = parts[0]
        
        if year_option == "Todos":
            df = get_cached_data_filtered_by_uc(None)
        else:
            try:
                year = int(year_option)
                df = get_cached_data_filtered_by_uc(year)
            except ValueError:
                df = get_cached_data_filtered_by_uc(None)
    else:
        df = get_cached_data_filtered_by_uc(None)
    
    if df is None:
        return pd.DataFrame(), ''
    
    processor = RankingProcessor()
    return processor.process_ranking(df, theme, period)

def initialize_data() -> Tuple[List[str], pd.DataFrame]:
    try:
        stats = get_summary_stats()
        
        if stats and stats.get('total_registros', 0) > 0:
            if stats.get('data_inicio') and stats.get('data_fim'):
                ano_inicio = stats['data_inicio'].year if hasattr(stats['data_inicio'], 'year') else 2020
                ano_fim = stats['data_fim'].year if hasattr(stats['data_fim'], 'year') else 2024
                years = list(range(ano_inicio, ano_fim + 1))
            else:
                years = get_available_years()
        else:
            years = get_available_years()
        
        year_options = ["Todos os Anos"] + [str(year) for year in years]
        base_df = get_cached_data_optimized(None)
        
        return year_options, base_df if base_df is not None else pd.DataFrame()
    except Exception as e:
        print(f"Erro na inicialização: {e}")
        return ["Todos os Anos"], pd.DataFrame()

def get_year_data(year_option: str, base_df: pd.DataFrame) -> pd.DataFrame:
    if year_option == "Todos os Anos":
        return base_df if not base_df.empty else pd.DataFrame()
    else:
        try:
            year = int(year_option)
            if base_df.empty:
                return get_cached_data_optimized(year)
            else:
                year_data = base_df[base_df['DataHora'].dt.year == year].copy()
                if year_data.empty:
                    return get_cached_data_optimized(year)
                return year_data
        except (ValueError, KeyError):
            return pd.DataFrame()

def render_interface():
    YEAR_OPTIONS, DF_BASE = initialize_data()
    
    if DF_BASE.empty:
        st.error("Dados não disponíveis.")
        return
    
    st.header("Focos de Calor")
    
    ano_sel = st.selectbox(
        'Período:',
        YEAR_OPTIONS,
        index=0,
        key="ano_focos_calor"
    )

    df_year = get_year_data(ano_sel, DF_BASE)
    
    if df_year.empty:
        st.warning(f"Sem dados para {ano_sel}")
        return

    st.subheader("Ranking de Municípios")
    
    col1, col2 = st.columns(2)
    with col1:
        tema_rank = st.selectbox(
            'Indicador:',
            ["Maior Risco de Fogo", "Maior Precipitação (evento)", "Máx. Dias Sem Chuva"],
            key="tema_ranking"
        )
    df_hash = f"{ano_sel}_{len(df_year)}_{tema_rank}"
    periodo = "Todo o Período" if ano_sel == "Todos os Anos" else f"Ano {ano_sel}"
    df_rank, col_ord = get_cached_ranking(df_hash, tema_rank, periodo)
    
    if not df_rank.empty:
        st.dataframe(df_rank, use_container_width=True, hide_index=True)
    else:
        st.info("Dados não disponíveis para este ranking.")
import os
files_to_check = ["cnuc.shp", "alertas.shp", "sigef.shp"]
gdf_alertas_cols = ['geometry', 'MUNICIPIO', 'AREAHA', 'ANODETEC', 'DATADETEC', 'CODEALERTA', 'ESTADO', 'BIOMA', 'VPRESSAO']
# Colunas essenciais - outras serão verificadas dinamicamente
gdf_cnuc_cols_base = ['geometry', 'nome_uc', 'municipio'] 
gdf_sigef_cols = ['geometry', 'municipio', 'area_km2', 'invadindo']
df_csv_cols = ["Unnamed: 0", "Áreas de conflitos", "Assassinatos", "Conflitos por Terra", "Ocupações Retomadas", "Tentativas de Assassinatos", "Trabalho Escravo", "Latitude", "Longitude"]
df_proc_cols = ['numero_processo', 'data_ajuizamento', 'municipio', 'classe', 'assuntos', 'orgao_julgador', 'ultima_atualizaçao']


gdf_alertas_raw = carregar_shapefile(
    r"alertas.shp",
    calcular_percentuais=False,
    columns=gdf_alertas_cols
)
gdf_alertas_raw = gdf_alertas_raw.rename(columns={"id":"id_alerta"})

gdf_cnuc_raw = carregar_cnuc_adaptativo(r"cnuc.shp")

gdf_cnuc_ha_raw = preparar_hectares(gdf_cnuc_raw)

gdf_sigef_raw = carregar_shapefile(
    r"sigef.shp",
    calcular_percentuais=False,
    columns=gdf_sigef_cols
)
gdf_sigef_raw   = gdf_sigef_raw.rename(columns={"id":"id_sigef"})

if 'MUNICIPIO' in gdf_sigef_raw.columns and 'municipio' not in gdf_sigef_raw.columns:
    gdf_sigef_raw = gdf_sigef_raw.rename(columns={'MUNICIPIO': 'municipio'})
elif 'municipio' not in gdf_sigef_raw.columns:
    st.warning("Coluna 'municipio' ou 'MUNICIPIO' não encontrada em sigef.shp. Adicionando coluna placeholder.")
    gdf_sigef_raw['municipio'] = None 

limites = gdf_cnuc_raw.total_bounds
centro = {
    "lat": (limites[1] + limites[3]) / 2,
    "lon": (limites[0] + limites[2]) / 2
}

df_csv_raw = load_csv(
    r"CPT-PA-count.csv", 
    columns=df_csv_cols
)
df_confmun_raw = carregar_dados_conflitos_municipio(
    r"CPTF-PA.xlsx"
)

@st.cache_data
def load_df_proc(caminho: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(caminho, sep=";", encoding="windows-1252", usecols=columns)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer', errors='coerce')
        elif df[col].dtype == 'object':
            if len(df[col].unique()) / len(df) < 0.5:
                 try:
                    df[col] = df[col].astype('category')
                 except Exception:
                    pass
    return df

df_proc_raw    = load_df_proc(
    r"processos_tjpa_completo_atualizada_pronto.csv",
    columns=df_proc_cols
)

tabs = st.tabs(["Sobreposições", "CPT", "Justiça", "Queimadas", "Desmatamento"])

with tabs[0]:
    st.header("Sobreposições")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre sobreposições territoriais, incluindo:
        - Percentuais de alertas e CARs sobre extensão territorial
        - Distribuição por municípios
        - Áreas e contagens por Unidade de Conservação
        
        Os dados são provenientes do CNUC (Cadastro Nacional de Unidades de Conservação) e SIGEF (Sistema de Gestão Fundiária).
        """)
        st.markdown(
            "**Fonte Geral da Seção:** MMA - Ministério do Meio Ambiente. Cadastro Nacional de Unidades de Conservação. Brasília: MMA.",
            unsafe_allow_html=True
               )

    perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef = criar_cards(gdf_cnuc_raw, gdf_sigef_raw, None)
    
    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        ucs_disponiveis = ['Todas'] + list(gdf_cnuc_raw['nome_uc'].unique()) if not gdf_cnuc_raw.empty and 'nome_uc' in gdf_cnuc_raw.columns else ['Todas']
        uc_selecionada = st.selectbox('Filtrar por UC:', ucs_disponiveis, key="filtro_uc")
    with col_f2:
        estados_disponiveis = ['Todos'] + list(gdf_alertas_raw['ESTADO'].unique()) if not gdf_alertas_raw.empty and 'ESTADO' in gdf_alertas_raw.columns else ['Todos']
        estado_selecionado = st.selectbox('Filtrar por Estado:', estados_disponiveis, key="filtro_estado")
    
    gdf_cnuc_filtrado = gdf_cnuc_raw.copy()
    gdf_alertas_filtrado_cards = gdf_alertas_raw.copy()
    
    if uc_selecionada != 'Todas':
        gdf_cnuc_filtrado = gdf_cnuc_filtrado[gdf_cnuc_filtrado['nome_uc'] == uc_selecionada]
    
    if estado_selecionado != 'Todos':
        gdf_alertas_filtrado_cards = gdf_alertas_filtrado_cards[gdf_alertas_filtrado_cards['ESTADO'] == estado_selecionado]
    
    total_ucs = len(gdf_cnuc_filtrado) if not gdf_cnuc_filtrado.empty else 0

    area_total_ucs = 0
    area_alertas_ucs = 0
    area_cars_ucs = 0
    
    if not gdf_cnuc_filtrado.empty:
        if 'ha_total' in gdf_cnuc_filtrado.columns:
            area_total_ucs = gdf_cnuc_filtrado['ha_total'].sum()
        if 'alerta_km2' in gdf_cnuc_filtrado.columns:
            area_alertas_ucs = gdf_cnuc_filtrado['alerta_km2'].sum() * 100 
        if 'sigef_km2' in gdf_cnuc_filtrado.columns:
            area_cars_ucs = gdf_cnuc_filtrado['sigef_km2'].sum() * 100  
    
    try:
        area_total_ucs = float(area_total_ucs) if pd.notna(area_total_ucs) else 0
        area_alertas_ucs = float(area_alertas_ucs) if pd.notna(area_alertas_ucs) else 0
        area_cars_ucs = float(area_cars_ucs) if pd.notna(area_cars_ucs) else 0
    except (ValueError, TypeError):
        area_total_ucs = 0
        area_alertas_ucs = 0
        area_cars_ucs = 0
    
    # Calcular dados para municípios (filtrados por estado)
    municipios_para = ['Altamira', 'São Félix do Xingu', 'Itaituba', 'Jacareacanga', 'Novo Progresso', 'Trairão']
    if estado_selecionado == 'PA' or estado_selecionado == 'Pará':
        total_municipios = 6  # Municípios conhecidos do Pará
    elif estado_selecionado == 'Todos':
        total_municipios = 6  # Total conhecido (6 do Pará)
    else:
        total_municipios = len(gdf_alertas_filtrado_cards['MUNICIPIO'].unique()) if not gdf_alertas_filtrado_cards.empty and 'MUNICIPIO' in gdf_alertas_filtrado_cards.columns else 0
    alertas_municipios = len(gdf_alertas_filtrado_cards) if not gdf_alertas_filtrado_cards.empty else 0
    area_alertas_municipios = gdf_alertas_filtrado_cards['AREAHA'].sum() if not gdf_alertas_filtrado_cards.empty and 'AREAHA' in gdf_alertas_filtrado_cards.columns else 0
    cars_municipios = len(gdf_sigef_raw) if not gdf_sigef_raw.empty else 0
    
    # Garantir que os valores sejam números válidos
    try:
        area_alertas_municipios = float(area_alertas_municipios) if pd.notna(area_alertas_municipios) else 0
    except (ValueError, TypeError):
        area_alertas_municipios = 0
    
    card_template = """
    <div style="
        background-color:#F9F9FF;
        border:1px solid #E0E0E0;
        padding:1rem;
        border-radius:8px;
        box-shadow:0 2px 4px rgba(0,0,0,0.1);
        text-align:center;
        height:100px;
        display:flex;
        flex-direction:column;
        justify-content:center;">
        <h5 style="margin:0; font-size:0.9rem; color:#2F5496;">{0}</h5>
        <p style="margin:0; font-size:1.2rem; font-weight:bold; color:#2F5496;">{1}</p>
        <small style="color:#666;">{2}</small>
    </div>
    """
    
    # Primeira linha: Unidades de Conservação
    st.markdown("### Unidades de Conservação:")
    cols_uc = st.columns(4, gap="small")
    titulos_uc = [
        ("UCs", safe_format_number(total_ucs, 0), "Total de Unidades de Conservação"),
        ("Área Total UCs (ha)", safe_format_number(area_total_ucs, 1), "Área total das UCs em hectares"),
        ("Área Alertas UCs (ha)", safe_format_number(area_alertas_ucs, 1), "Área de alertas em UCs (ha)"),
        ("Área CARs UCs (ha)", safe_format_number(area_cars_ucs, 1), "Área de CARs em UCs (ha)")
    ]
    for col, (t, v, d) in zip(cols_uc, titulos_uc):
        col.markdown(card_template.format(t, v, d), unsafe_allow_html=True)
    
    titulo_regiao = f"### {estado_selecionado if estado_selecionado != 'Todos' else 'Municípios'}:"
    st.markdown(titulo_regiao)
    cols_mun = st.columns(4, gap="small")
    titulos_mun = [
        ("Municípios", safe_format_number(total_municipios, 0), f"Municípios em {estado_selecionado if estado_selecionado != 'Todos' else 'todos os estados'}"),
        ("Alertas Totais", safe_format_number(alertas_municipios, 0), f"Alertas em {estado_selecionado if estado_selecionado != 'Todos' else 'todos os estados'}"),
        ("Área Alertas (ha)", safe_format_number(area_alertas_municipios, 1), "Área total de alertas (ha)"),
        ("CARs Totais", safe_format_number(cars_municipios, 0), f"CARs em {estado_selecionado if estado_selecionado != 'Todos' else 'todos os estados'}")
    ]
    for col, (t, v, d) in zip(cols_mun, titulos_mun):
        col.markdown(card_template.format(t, v, d), unsafe_allow_html=True)

    st.divider()

    row1_map, row1_chart1 = st.columns([3, 2], gap="large")
    with row1_map:
        opcoes_invadindo = ["Selecione", "Todos"] + sorted(gdf_sigef_raw["invadindo"].str.strip().unique().tolist())
        invadindo_opcao_temp = st.selectbox("Tipo de sobreposição:", opcoes_invadindo, index=0, help="Selecione o tipo de área sobreposta para análise")
        invadindo_opcao = None if invadindo_opcao_temp == "Selecione" else invadindo_opcao_temp
        gdf_cnuc_map = gdf_cnuc_raw.copy()
        gdf_sigef_map = gdf_sigef_raw.copy()
        ids_selecionados_map = []

        if invadindo_opcao and invadindo_opcao.lower() != "todos":
            sigef_filtered_for_sjoin = gdf_sigef_map[gdf_sigef_map["invadindo"].str.strip().str.lower() == invadindo_opcao.lower()]
            if not sigef_filtered_for_sjoin.empty:
                 gdf_cnuc_proj_sjoin = gdf_cnuc_map.to_crs(sigef_filtered_for_sjoin.crs)
                 gdf_filtrado_map = gpd.sjoin(gdf_cnuc_proj_sjoin, sigef_filtered_for_sjoin, how="inner", predicate="intersects")
                 ids_selecionados_map = gdf_filtrado_map["id"].unique().tolist()
            else:
                 ids_selecionados_map = [] 

        st.subheader("Mapa de Unidades")
        fig_map = criar_figura(gdf_cnuc_map, gdf_sigef_map, df_csv_raw, centro, ids_selecionados_map, invadindo_opcao)
        st.plotly_chart(
            fig_map,
            use_container_width=True,
            height=300,
            config={"scrollZoom": True}
        )
        st.caption("Figura 1.1: Distribuição espacial das unidades de conservação.")
        with st.expander("Detalhes e Fonte da Figura 1.1"):
            st.write("""
            **Interpretação:**
            O mapa mostra a distribuição espacial das unidades de conservação na região, destacando as áreas com sobreposições selecionadas.

            **Observações:**
            - Áreas em destaque indicam unidades de conservação
            - Cores diferentes representam diferentes tipos de unidades
            - Sobreposições são destacadas quando selecionadas no filtro

            **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
            """)

        st.subheader("Proporção da Área do CAR sobre a UC")
        uc_names = ["Todas"] + sorted(gdf_cnuc_ha_raw["nome_uc"].unique())
        nome_uc = st.selectbox("Selecione a Unidade de Conservação:", uc_names)
        modo_input = st.radio("Mostrar valores como:", ["Hectares (ha)", "% da UC"], horizontal=True)
        modo = "absoluto" if modo_input == "Hectares (ha)" else "percent"
        fig = fig_car_por_uc_donut(gdf_cnuc_ha_raw, nome_uc, modo)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Figura 1.2: Comparação entre área do CAR e área restante da UC.")
        with st.expander("Detalhes e Fonte da Figura 1.2"):
            st.write("""
            **Interpretação:**
            Este gráfico mostra a proporção entre a área cadastrada no CAR e a área restante da Unidade de Conservação (UC).

            **Observações:**
            - A área restante é o que sobra da UC após considerar a área cadastrada no CAR
            - Pode ocorrer de o CAR ultrapassar 100% devido a sobreposições ou múltiplos cadastros em uma mesma área
            - Valores podem ser visualizados em hectares ou percentual, conforme seleção acima

            **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
            """)

    with row1_chart1:
        st.subheader("Áreas por UC")
        st.plotly_chart(fig_sobreposicoes_mapbiomas(gdf_cnuc_raw, gdf_alertas_raw), use_container_width=True, height=350)
        st.caption("Figura 1.3: Distribuição de áreas por unidade de conservação (dados MapBiomas).") 
        with st.expander("Detalhes e Fonte da Figura 1.3"):
            st.write("""
            **Interpretação:**
            O gráfico apresenta a área em hectares de cada unidade de conservação, permitindo comparar suas extensões territoriais.

            **Observações:**
            - Barras representam área em hectares
            - Linha tracejada indica a média
            - Ordenado por tamanho da área

            **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
            """)

        st.subheader("Contagens por UC")
        st.plotly_chart(fig_contagens_uc(gdf_cnuc_raw), use_container_width=True, height=350)
        st.caption("Figura 1.4: Contagem de sobreposições por unidade de conservação.")
        with st.expander("Detalhes e Fonte da Figura 1.4"):
            st.write("""
            **Interpretação:**
            O gráfico mostra o número de alertas e CARs sobrepostos a cada unidade de conservação.

            **Observações:**
            - Barras empilhadas mostram alertas e CARs
            - Linha tracejada indica média total
            - Ordenado por total de sobreposições

            **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
            """)

    st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
        <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Tabela Unificada</h3>
        <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Visualização unificada dos dados de alertas e CNUC.</p>
    </div>""", unsafe_allow_html=True)
    mostrar_tabela_unificada(gdf_alertas_raw, gdf_sigef_raw, gdf_cnuc_raw)
    st.caption("Tabela 1.1: Dados consolidados por município.")
    with st.expander("Detalhes e Fonte da Tabela 1.1"):
        st.write("""
        **Interpretação:**
        A tabela apresenta os dados consolidados por município, incluindo:
        - Área de alertas em hectares
        - Área do CNUC em hectares

        **Observações:**
        - Valores em hectares
        - Totais na última linha
        - Células coloridas por tipo de dado

        **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
        """)
    
    # Dados Completos
    st.divider()
    st.markdown("### 📊 Dados Completos")
    
    dados_tabs = st.tabs(["Alertas", "Unidades de Conservação", "SIGEF"])
    
    with dados_tabs[0]:
        st.markdown("**Dados brutos de alertas de desmatamento:**")
        if not gdf_alertas_raw.empty:
            df_alertas_display = gdf_alertas_raw.drop(columns=['geometry']) if 'geometry' in gdf_alertas_raw.columns else gdf_alertas_raw
            st.dataframe(df_alertas_display, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de alertas disponível.")
    
    with dados_tabs[1]:
        st.markdown("**Dados brutos das Unidades de Conservação:**")
        if not gdf_cnuc_raw.empty:
            df_cnuc_display = gdf_cnuc_raw.drop(columns=['geometry']) if 'geometry' in gdf_cnuc_raw.columns else gdf_cnuc_raw
            st.dataframe(df_cnuc_display, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de UCs disponível.")
    
    with dados_tabs[2]:
        st.markdown("**Dados brutos do SIGEF:**")
        if not gdf_sigef_raw.empty:
            df_sigef_display = gdf_sigef_raw.drop(columns=['geometry']) if 'geometry' in gdf_sigef_raw.columns else gdf_sigef_raw
            st.dataframe(df_sigef_display, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado do SIGEF disponível.")


with tabs[1]:
    st.header("Impacto Social")
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre impactos sociais relacionados a conflitos agrários, incluindo:
        - Famílias afetadas
        - Conflitos registrados
        - Ocupações retomadas

        Os dados são provenientes da Comissão Pastoral da Terra (CPT).
        """)
        st.markdown(
            "**Fonte Geral da Seção:** CPT - Comissão Pastoral da Terra. Conflitos no Campo Brasil. Goiânia: CPT Nacional.",
            unsafe_allow_html=True
        )

    df_tabela_social = df_confmun_raw.copy()

    df_csv_cleaned = df_csv_raw.copy()
    if 'Município' in df_csv_cleaned.columns:
        df_csv_cleaned['Município'] = df_csv_cleaned['Município'].apply(lambda x: str(x).strip().title() if pd.notna(x) else None)

    if 'Município' in df_tabela_social.columns:
         df_tabela_social['Município'] = df_tabela_social['Município'].apply(lambda x: str(x).strip().title() if pd.notna(x) else None)

    csv_cols_to_merge = ['Município']
    if 'Ocupações Retomadas' in df_csv_cleaned.columns:
        csv_cols_to_merge.append('Ocupações Retomadas')

    if len(csv_cols_to_merge) > 1:
        df_csv_agg = df_csv_cleaned[csv_cols_to_merge].groupby('Município', observed=False).sum().reset_index()
        df_tabela_social = df_tabela_social.merge(df_csv_agg, on='Município', how='left').fillna(0)
    else:
        if 'Ocupações Retomadas' not in df_tabela_social.columns:
            df_tabela_social['Ocupações Retomadas'] = 0


    df_tabela_social = df_tabela_social.sort_values('Total_Famílias', ascending=False)

    df_display = df_tabela_social.rename(columns={
        'Município': 'Município',
        'Total_Famílias': 'Famílias Afetadas',
        'Número_Conflitos': 'Conflitos Registrados',
        'Ocupações Retomadas': 'Ocupações Retomadas'
    })

    display_cols = ['Município', 'Famílias Afetadas', 'Conflitos Registrados']
    for col in display_cols:
        if col not in df_display.columns:
            df_display[col] = 0

    linha_total = pd.DataFrame({
        'Município': ['TOTAL'],
        'Famílias Afetadas': [df_display['Famílias Afetadas'].sum()],
        'Conflitos Registrados': [df_display['Conflitos Registrados'].sum()]
    })
    df_display_com_total = pd.concat([df_display, linha_total], ignore_index=True)
    
    if 'Ocupações Retomadas' in df_display_com_total.columns:
        df_display_com_total = df_display_com_total.drop(columns=['Ocupações Retomadas'])

    def aplicar_cor_social(val, col):
        if col == 'Município':
            return 'background-color: #f0f0f0' if val == 'TOTAL' else ''
        elif col == 'Famílias Afetadas':
            return 'background-color: #ffebee; font-weight: bold' if val == df_display_com_total[col].iloc[-1] else 'background-color: #ffebee'
        elif col == 'Conflitos Registrados':
            return 'background-color: #fff3e0; font-weight: bold' if val == df_display_com_total[col].iloc[-1] else 'background-color: #fff3e0'
        return ''

    def format_with_dots(val):
        if pd.isna(val):
            return ""
        return format_number_with_dots(val, 0)

    styled_df = df_display_com_total.style.apply(
        lambda x: [aplicar_cor_social(val, col) for val, col in zip(x, df_display_com_total.columns)],
        axis=1
    ).format({
        'Famílias Afetadas': format_with_dots,
        'Conflitos Registrados': format_with_dots
    })

    col_fam, col_conf = st.columns(2, gap="large")
    with col_fam:
        st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
            <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Famílias Afetadas</h3>
            <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Distribuição do número de famílias afetadas por conflitos por município.</p>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_familias(df_confmun_raw), use_container_width=True, height=400, key="familias")
        st.caption("Figura 3.1: Distribuição de famílias afetadas por município.")
        with st.expander("Detalhes e Fonte da Figura 3.1"):
            st.write("""
            **Interpretação:**
            O gráfico apresenta o número total de famílias afetadas por conflitos em cada município.

            **Observações:**
            - Dados agregados por município
            - Valores apresentados em ordem decrescente
            - Inclui todos os tipos de conflitos registrados

            **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
            """)
    with col_conf:
        st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
            <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Conflitos Registrados</h3>
            <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Número total de conflitos registrados por município.</p>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_conflitos(df_confmun_raw), use_container_width=True, height=400, key="conflitos")
        st.caption("Figura 3.2: Distribuição de conflitos registrados por município.")
        with st.expander("Detalhes e Fonte da Figura 3.2"):
            st.write("""
            **Interpretação:**
            O gráfico mostra o número total de conflitos registrados em cada município.

            **Observações:**
            - Contagem total de ocorrências por município
            - Ordenação por quantidade de conflitos
            - Inclui todos os tipos de conflitos documentados

            **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
            """)

    st.markdown("---")
    st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 1rem 0 0.5rem 0;">
        <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Tabela Consolidada de Impactos Sociais</h3>
        <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Dados consolidados de impactos sociais por município.</p>
    </div>""", unsafe_allow_html=True)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    st.caption("Tabela 3.1: Dados consolidados de impactos sociais por município.")
    with st.expander("Detalhes e Fonte da Tabela 3.1"):
        st.write("""
        **Interpretação:**
        A tabela apresenta os dados consolidados por município, incluindo:
        - Número de famílias afetadas por conflitos
        - Quantidade de conflitos registrados

        **Observações:**
        - Valores absolutos por município
        - Totais na última linha
        - Células coloridas por tipo de dado
        - Ordenação por número de famílias afetadas

        **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
        """)
    
    # Dados Completos
    st.divider()
    st.markdown("### 📊 Dados Completos")
    
    dados_tabs_cpt = st.tabs(["Conflitos por Município", "Dados CPT Brutos"])
    
    with dados_tabs_cpt[0]:
        st.markdown("**Dados de conflitos consolidados por município:**")
        if not df_confmun_raw.empty:
            st.dataframe(df_confmun_raw, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de conflitos por município disponível.")
    
    with dados_tabs_cpt[1]:
        st.markdown("**Dados brutos da CPT:**")
        if not df_csv_raw.empty:
            st.dataframe(df_csv_raw, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado bruto da CPT disponível.")
    
with tabs[2]:
    st.header("Processos Judiciais")
    
    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre processos judiciais relacionados a questões ambientais, incluindo:
        - Distribuição por municípios
        - Classes processuais
        - Assuntos
        - Órgãos julgadores
        
        Os dados são provenientes do Tribunal de Justiça do Estado do Pará.
        """)
    
    st.markdown(
        "**Fonte Geral da Seção:** CNJ - Conselho Nacional de Justiça.",
        unsafe_allow_html=True
    )
    
    if 'data_ajuizamento' in df_proc_raw.columns:
        df_proc_raw['data_ajuizamento'] = pd.to_datetime(df_proc_raw['data_ajuizamento'], errors='coerce')
    if 'ultima_atualizaçao' in df_proc_raw.columns:
        df_proc_raw['ultima_atualizaçao'] = pd.to_datetime(df_proc_raw['ultima_atualizaçao'], errors='coerce')

    figs_j = fig_justica(df_proc_raw)
    
    cols = st.columns(2, gap="large")
    
    with cols[0]:
        st.markdown("""
        <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:0.5rem;">
        <h3 style="margin:0 0 .5rem 0;">Top 10 Municípios</h3>
        <p style="margin:0;font-size:.95em;color:#666;">Municípios com maior número de processos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'mun' in figs_j and figs_j['mun'] is not None:
            st.plotly_chart(figs_j['mun'].update_layout(height=400), use_container_width=True, key="jud_mun")
        else:
            st.warning("Gráfico de municípios não pôde ser gerado.")
        
        st.caption("Figura 4.1: Top 10 municípios com mais processos.")
        with st.expander("ℹ️ Detalhes e Fonte da Figura 4.1", expanded=False):
            st.write("""
            **Interpretação:**
            Distribuição dos processos por municípios.
            
            **Fonte:** CNJ - Conselho Nacional de Justiça.
            """)
    
    with cols[1]:
        st.markdown("""
        <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:0.5rem;">
        <h3 style="margin:0 0 .5rem 0;">Classes Processuais</h3>
        <p style="margin:0;font-size:.95em;color:#666;">Top 10 classes mais frequentes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'class' in figs_j and figs_j['class'] is not None:
            st.plotly_chart(figs_j['class'].update_layout(height=400), use_container_width=True, key="jud_class")
        else:
            st.warning("Gráfico de classes não pôde ser gerado.")
        
        st.caption("Figura 4.2: Top 10 classes processuais.")
        with st.expander("ℹ️ Detalhes e Fonte da Figura 4.2", expanded=False):
            st.write("""
            **Interpretação:**
            Distribuição dos processos por classes processuais.
            
            **Fonte:** CNJ - Conselho Nacional de Justiça.
            """)
    
    cols2 = st.columns(2, gap="large")
    
    with cols2[0]:
        st.markdown("""
        <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:0.5rem;">
        <h3 style="margin:0 0 .5rem 0;">Assuntos</h3>
        <p style="margin:0;font-size:.95em;color:#666;">Top 10 assuntos mais recorrentes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'ass' in figs_j and figs_j['ass'] is not None:
            st.plotly_chart(figs_j['ass'].update_layout(height=400), use_container_width=True, key="jud_ass")
        else:
            st.warning("Gráfico de assuntos não pôde ser gerado.")
        
        st.caption("Figura 4.3: Top 10 assuntos.")
        with st.expander("ℹ️ Detalhes e Fonte da Figura 4.3", expanded=False):
            st.write("""
            **Interpretação:**
            Distribuição dos processos por assuntos.
            
            **Fonte:** CNJ - Conselho Nacional de Justiça.
            """)
    
    with cols2[1]:
        st.markdown("""
        <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:0.5rem;">
        <h3 style="margin:0 0 .5rem 0;">Órgãos Julgadores</h3>
        <p style="margin:0;font-size:.95em;color:#666;">Top 10 órgãos com mais processos.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'org' in figs_j and figs_j['org'] is not None:
            st.plotly_chart(figs_j['org'].update_layout(height=400), use_container_width=True, key="jud_org")
        else:
            st.warning("Gráfico de órgãos julgadores não pôde ser gerado.")
        
        st.caption("Figura 4.4: Top 10 órgãos julgadores.")
        with st.expander("ℹ️ Detalhes e Fonte da Figura 4.4", expanded=False):
            st.write("""
            **Interpretação:**
            Distribuição dos processos por órgãos julgadores.
            
            **Fonte:** CNJ - Conselho Nacional de Justiça.
            """)
    
    st.markdown("""
    <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin:1rem 0 .5rem 0;">
    <h3 style="margin:0 0 .5rem 0;">Evolução Mensal de Processos</h3>
    <p style="margin:0;font-size:.95em;color:#666;">Variação mensal ao longo do período.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'temp' in figs_j and figs_j['temp'] is not None:
        st.plotly_chart(figs_j['temp'], use_container_width=True, key="jud_temp")
    else:
        st.warning("Gráfico de evolução temporal não pôde ser gerado.")
    
    st.caption("Figura 4.5: Evolução temporal dos processos judiciais.")
    with st.expander("ℹ️ Detalhes e Fonte da Figura 4.5", expanded=False):
        st.write("""
        **Interpretação:**
        Evolução mensal dos processos.
        
        **Fonte:** CNJ - Conselho Nacional de Justiça.
        """)
    st.markdown("""
    <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin:1rem 0 .5rem 0;">
    <h3 style="margin:0 0 .5rem 0;">Análise Interativa de Processos</h3>
    <p style="margin:0;font-size:.95em;color:#666;">Tabela com filtros para análise detalhada dos dados.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        tipo_analise = st.selectbox(
            "Escolha o tipo de análise:",
            ["Municípios com mais processos", "Órgãos mais atuantes", "Classes processuais mais frequentes", "Assuntos mais recorrentes", "Dados gerais relevantes"],
            key="tipo_analise_proc"
        )
    
    with col2:
        if 'data_ajuizamento' in df_proc_raw.columns:
            df_proc_raw['ano'] = pd.to_datetime(df_proc_raw['data_ajuizamento'], errors='coerce').dt.year
            anos_disponiveis = sorted([ano for ano in df_proc_raw['ano'].dropna().unique() if not pd.isna(ano)])
            if anos_disponiveis:
                ano_selecionado = st.selectbox(
                    "Filtrar por ano:",
                    ["Todos os anos"] + anos_disponiveis,
                    key="ano_filter_proc"
                )
            else:
                ano_selecionado = "Todos os anos"
        else:
            ano_selecionado = "Todos os anos"
    
    df_proc_filtered_year = df_proc_raw.copy()
    if ano_selecionado != "Todos os anos":
        df_proc_filtered_year = df_proc_filtered_year[df_proc_filtered_year['ano'] == ano_selecionado]
    
    df_filtrado = df_proc_filtered_year.copy()

    if tipo_analise == "Municípios com mais processos":
        if 'municipio' in df_filtrado.columns and 'numero_processo' in df_filtrado.columns and 'data_ajuizamento' in df_filtrado.columns:
            df_filtrado['municipio'] = df_filtrado['municipio'].apply(clean_text)
            tabela_resumo = df_filtrado.groupby('municipio', observed=False).agg({
                'numero_processo': 'count',
                'data_ajuizamento': ['min', 'max']
            }).round(2)
            tabela_resumo.columns = ['Total de Processos', 'Primeiro Processo', 'Último Processo']
            tabela_resumo = tabela_resumo.sort_values('Total de Processos', ascending=False).head(20)
            tabela_resumo = tabela_resumo.reset_index()
            
            st.dataframe(tabela_resumo, use_container_width=True)
            st.caption("Tabela 4.1: Top 20 municípios com mais processos judiciais.")
        else:
             st.info("Dados insuficientes para gerar esta tabela.")
        
    elif tipo_analise == "Órgãos mais atuantes":
        if 'orgao_julgador' in df_filtrado.columns and 'numero_processo' in df_filtrado.columns and 'data_ajuizamento' in df_filtrado.columns:
            df_filtrado['orgao_julgador'] = df_filtrado['orgao_julgador'].apply(clean_text)
            tabela_resumo = df_filtrado.groupby('orgao_julgador', observed=False).agg({
                'numero_processo': 'count',
                'data_ajuizamento': ['min', 'max']
            }).round(2)
            tabela_resumo.columns = ['Total de Processos', 'Primeiro Processo', 'Último Processo']
            tabela_resumo = tabela_resumo.sort_values('Total de Processos', ascending=False).head(15)
            tabela_resumo = tabela_resumo.reset_index()
            
            st.dataframe(tabela_resumo, use_container_width=True)
            st.caption("Tabela 4.1: Top 15 órgãos julgadores mais atuantes.")
        else:
             st.info("Dados insuficientes para gerar esta tabela.")

    elif tipo_analise == "Classes processuais mais frequentes":
        if 'classe' in df_filtrado.columns and 'numero_processo' in df_filtrado.columns and 'data_ajuizamento' in df_filtrado.columns:
            df_filtrado['classe'] = df_filtrado['classe'].apply(clean_text)
            tabela_resumo = df_filtrado.groupby('classe', observed=False).agg({
                'numero_processo': 'count',
                'data_ajuizamento': ['min', 'max']
            }).round(2)
            tabela_resumo.columns = ['Total de Processos', 'Primeiro Processo', 'Último Processo']
            tabela_resumo = tabela_resumo.sort_values('Total de Processos', ascending=False).head(15)
            tabela_resumo = tabela_resumo.reset_index()
            
            st.dataframe(tabela_resumo, use_container_width=True)
            st.caption("Tabela 4.1: Top 15 classes processuais mais frequentes.")
        else:
             st.info("Dados insuficientes para gerar esta tabela.")

    elif tipo_analise == "Assuntos mais recorrentes":
        if 'assuntos' in df_filtrado.columns and 'numero_processo' in df_filtrado.columns and 'data_ajuizamento' in df_filtrado.columns:
            df_filtrado['assuntos'] = df_filtrado['assuntos'].apply(clean_text)
            tabela_resumo = df_filtrado.groupby('assuntos', observed=False).agg({
                'numero_processo': 'count',
                'data_ajuizamento': ['min', 'max']
            }).round(2)
            tabela_resumo.columns = ['Total de Processos', 'Primeiro Processo', 'Último Processo']
            tabela_resumo = tabela_resumo.sort_values('Total de Processos', ascending=False).head(15)
            tabela_resumo = tabela_resumo.reset_index()
            
            st.dataframe(tabela_resumo, use_container_width=True)
            st.caption("Tabela 4.1: Top 15 assuntos mais recorrentes.")
        else:
             st.info("Dados insuficientes para gerar esta tabela.")

    else: 
        colunas_relevantes = ['numero_processo', 'data_ajuizamento', 'municipio', 'classe', 'assuntos', 'orgao_julgador']
        colunas_existentes = [col for col in colunas_relevantes if col in df_filtrado.columns]
        
        if colunas_existentes:
            df_relevante = df_filtrado[colunas_existentes].copy()
            
            for col in ['municipio', 'classe', 'assuntos', 'orgao_julgador']:
                if col in df_relevante.columns:
                    df_relevante[col] = df_relevante[col].apply(clean_text)
            
            if 'data_ajuizamento' in df_relevante.columns:
                df_relevante = df_relevante.sort_values('data_ajuizamento', ascending=False)
            
            st.dataframe(df_relevante.head(500), use_container_width=True)
            st.caption("Tabela 4.1: Dados gerais relevantes dos processos judiciais (limitado a 500 registros).")
        else:
            st.info("Não foi possível carregar os dados relevantes.")
    
    with st.expander("ℹ️ Sobre esta tabela", expanded=False):
        if tipo_analise == "Municípios com mais processos":
            st.write("""
            Esta tabela mostra os municípios com maior número de processos judiciais,
            incluindo o total de processos e o período de atuação (primeiro e último processo).
            """)
        elif tipo_analise == "Órgãos mais atuantes":
            st.write("""
            Esta tabela apresenta os órgãos julgadores com maior volume de processos,
            mostrando sua atividade ao longo do tempo.
            """)
        elif tipo_analise == "Classes processuais mais frequentes":
            st.write("""
            Esta tabela mostra as classes processuais mais utilizadas nos processos judiciais,
            indicando os tipos de ações mais comuns no sistema judiciário.
            """)
        elif tipo_analise == "Assuntos mais recorrentes":
            st.write("""
            Esta tabela apresenta os assuntos mais frequentes nos processos judiciais,
            revelando as principais questões levadas ao judiciário.
            """)
        else:
            st.write("""
            Esta tabela apresenta os dados gerais mais relevantes dos processos judiciais,
            ordenados por data de ajuizamento (mais recentes primeiro).
            Limitada a 500 registros para melhor performance.
            """)
    
    st.markdown(
        "**Fonte:** CNJ - Conselho Nacional de Justiça.",
        unsafe_allow_html=True
    )
    
    st.divider()
    st.markdown("### 📊 Dados Completos")
    st.markdown("**Dados brutos dos processos judiciais:**")
    if not df_proc_raw.empty:
        st.dataframe(df_proc_raw, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum dado de processos judiciais disponível.")
    

with tabs[3]:
    st.header("Focos de Calor")

    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write(
            "Esta análise apresenta dados sobre focos de calor detectados por satélite, incluindo:"
        )
        st.write("- Risco de fogo") 
        st.write("- Precipitação acumulada")
        st.write("- Distribuição espacial")
        st.write("- Análise por Unidades de Conservação")
        
        st.markdown("---")
        st.markdown("**Sobre o Risco de Fogo:** O valor do Risco de Fogo varia de 0.0 a 1.0 e é classificado como:")
        st.write("- **Mínimo:** abaixo de 0,15")
        st.write("- **Baixo:** de 0,15 a 0,4")
        st.write("- **Médio:** de 0,4 a 0,7")
        st.write("- **Alto:** de 0,7 a 0,95")
        st.write("- **Crítico:** acima de 0,95")
        
        st.markdown(
            "**Fonte:** BD Queimadas - INPE, 2025.",
            unsafe_allow_html=True
        )

    st.subheader("Focos de Calor em Unidades de Conservação")
    
    YEAR_OPTIONS, DF_BASE = initialize_data()
    
    if DF_BASE is not None and not DF_BASE.empty and not gdf_cnuc_raw.empty:

        try:
            from shapely.geometry import Point
            df_valid = DF_BASE.dropna(subset=['Latitude', 'Longitude']).copy()
            if not df_valid.empty:
                geometry = [Point(lon, lat) for lon, lat in zip(df_valid['Longitude'], df_valid['Latitude'])]
                gdf_focos = gpd.GeoDataFrame(df_valid, geometry=geometry, crs="EPSG:4326")
                crs_proj = "EPSG:31983"
                gdf_focos_proj = gdf_focos.to_crs(crs_proj)
                gdf_cnuc_proj = gdf_cnuc_raw.to_crs(crs_proj)
                focos_in_ucs = gpd.sjoin(gdf_focos_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
                
                total_focos_geral = len(DF_BASE)
                focos_em_ucs = len(focos_in_ucs) if not focos_in_ucs.empty else 0
                percentual_ucs = (focos_em_ucs / total_focos_geral * 100) if total_focos_geral > 0 else 0
                
                col1, col2, col3 = st.columns(3, gap="medium")
                
                card_template = """
                <div style="
                    background-color:#F9F9FF;
                    border:1px solid #E0E0E0;
                    padding:1.5rem;
                    border-radius:8px;
                    box-shadow:0 2px 4px rgba(0,0,0,0.1);
                    text-align:center;
                    height:120px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;">
                    <h4 style="margin:0; font-size:1rem; color:#2F5496;">{titulo}</h4>
                    <p style="margin:0.5rem 0 0 0; font-size:1.8rem; font-weight:bold; color:#2F5496;">{valor}</p>
                    <small style="color:#666; margin:0;">{descricao}</small>
                </div>
                """
                
                with col1:
                    st.markdown(
                        card_template.format(
                            titulo="Focos em UCs",
                            valor=format_number_with_dots(focos_em_ucs, 0),
                            descricao="Total de focos detectados em UCs"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.markdown(
                        card_template.format(
                            titulo="Total de Focos",
                            valor=format_number_with_dots(total_focos_geral, 0),
                            descricao="Total geral de focos detectados"
                        ),
                        unsafe_allow_html=True
                    )
                
                with col3:
                    st.markdown(
                        card_template.format(
                            titulo="% em UCs",
                            valor=f"{percentual_ucs:.1f}%".replace('.', ','),
                            descricao="Percentual de focos em UCs"
                        ),
                        unsafe_allow_html=True
                    )
                
                if not focos_in_ucs.empty:
                    st.markdown("**Ranking de UCs com mais focos de calor:**")
                    focos_por_uc = focos_in_ucs.groupby('nome_uc', observed=False).size().reset_index(name='quantidade_focos')
                    focos_por_uc = focos_por_uc.sort_values('quantidade_focos', ascending=False).head(10)
                    ranking_display = focos_por_uc.copy()
                    ranking_display.index = range(1, len(ranking_display) + 1)
                    ranking_display.columns = ['Unidade de Conservação', 'Quantidade de Focos']
                    st.dataframe(ranking_display, use_container_width=True)
                else:
                    st.info("Nenhum foco de calor detectado dentro das Unidades de Conservação.")
            else:
                st.warning("Dados de coordenadas não disponíveis para análise espacial.")
        except Exception as e:
            st.warning(f"Erro ao processar focos de calor em UCs: {e}")
    else:
        st.info("Dados não disponíveis para análise de focos em UCs.")
    
    st.divider()

    if DF_BASE is not None and not DF_BASE.empty:
        ano_sel_graf = st.selectbox(
            'Período para gráficos:',
            YEAR_OPTIONS,
            index=0, 
            key="ano_focos_calor_global_tab3"
        )
        
        df_graf = get_year_data(ano_sel_graf, DF_BASE)
        
        ano_param = None if ano_sel_graf == "Todos os Anos" else int(ano_sel_graf)
        display_graf = ("todo o período histórico" if ano_param is None else f"o ano de {ano_param}")

        if not df_graf.empty:
            df_hash = f"{ano_sel_graf}_{len(df_graf)}"
            
            figs = graficos_inpe(df_graf, ano_sel_graf, gdf_cnuc_raw)
            
            st.subheader("Evolução Temporal do Risco de Fogo")
            st.plotly_chart(figs['temporal'], use_container_width=True)
            st.caption(f"Figura: Evolução mensal do risco médio de fogo para {display_graf}.")

            col1, col2 = st.columns(2, gap="large")
            with col1:
                st.subheader("Top Municípios por Risco Médio de Fogo")
                st.plotly_chart(figs['top_risco'], use_container_width=True)
            with col2:
                st.subheader("Mapa de Distribuição dos Focos de Calor")
                st.plotly_chart(figs['mapa'], use_container_width=True, height=500, config={'scrollZoom': True})
            
            st.divider()
            col3, col4 = st.columns(2, gap="large")
            with col3:
                st.subheader("Top Municípios por Precipitação Acumulada")
                st.plotly_chart(figs['top_precip'], use_container_width=True)
            with col4:
                st.subheader("Focos de Calor por Unidade de Conservação")
                fig_focos_uc = fig_focos_calor_por_uc(df_graf, gdf_cnuc_raw)
                if fig_focos_uc and fig_focos_uc.data:
                    st.plotly_chart(fig_focos_uc, use_container_width=True, height=500)
                    st.caption("Figura: Top 10 Unidades de Conservação com maior quantidade de focos de calor.")
                else:
                    st.info("Não foram encontrados focos de calor dentro das Unidades de Conservação para o período selecionado.")
        else:
            st.warning(f"Nenhum dado para {ano_sel_graf}.")
            
        st.divider()
        st.header("Ranking de Municípios por Indicadores de Queimadas")
        st.caption("Classifica municípios pelo maior registro de cada indicador.")
        colA, colB = st.columns(2)
        with colA:
            ano_sel_rank = st.selectbox(
                'Período para ranking:', YEAR_OPTIONS,
                index=0, key="ano_ranking_tab3"
            )
        with colB:
            tema_rank = st.selectbox(
                'Indicador para ranking:',
                ["Maior Risco de Fogo", "Maior Precipitação (evento)", "Máx. Dias Sem Chuva"],
                key="tema_ranking"
            )
        
        ano_rank_param = None if ano_sel_rank == "Todos os Anos" else int(ano_sel_rank)
        periodo_rank = ("Todo o Período Histórico" if ano_rank_param is None else f"Ano de {ano_rank_param}")

        st.subheader(f"Ranking por {tema_rank} ({periodo_rank})")
        
        rank_hash = f"{ano_sel_rank}_{tema_rank}_opt"
        df_rank, col_ord = get_cached_ranking(rank_hash, tema_rank, periodo_rank)
        
        if df_rank is not None and not df_rank.empty:
            st.dataframe(df_rank, use_container_width=True, hide_index=True)
        else:
            st.info("Sem dados válidos para este ranking.")
            
        st.divider()
        st.markdown("### 📊 Dados Completos")
        st.markdown("**Dados brutos de focos de calor:**")
        if DF_BASE is not None and not DF_BASE.empty:
            st.dataframe(DF_BASE, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum dado de focos de calor disponível.")
            
    else:
        st.error("Não foi possível carregar os dados de queimadas. Verifique a conexão com o banco de dados.")

@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def processar_dados_desmatamento(_gdf_alertas, ano_selecionado):
    """Processa e filtra dados de desmatamento com cache para melhor performance."""
    if ano_selecionado != 'Todos':
        gdf_filtrado = _gdf_alertas[_gdf_alertas['ANODETEC'] == ano_selecionado].copy()
    else:
        gdf_filtrado = _gdf_alertas.copy()
    
    if 'AREAHA' in gdf_filtrado.columns:
        gdf_filtrado['AREAHA'] = pd.to_numeric(gdf_filtrado['AREAHA'], errors='coerce')
    
    return gdf_filtrado

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def calcular_ranking_municipios_desmatamento(_gdf_alertas):
    """Calcula ranking de municípios por desmatamento com cache."""
    required_ranking_cols = ['ESTADO', 'MUNICIPIO', 'AREAHA', 'ANODETEC', 'BIOMA', 'VPRESSAO']
    if not all(col in _gdf_alertas.columns for col in required_ranking_cols):
        return pd.DataFrame()
    
    _gdf_alertas['AREAHA'] = pd.to_numeric(_gdf_alertas['AREAHA'], errors='coerce')
    
    ranking_municipios = _gdf_alertas.groupby(['ESTADO', 'MUNICIPIO'], observed=False).agg({
        'AREAHA': ['sum', 'count', 'mean'],
        'ANODETEC': ['min', 'max'],
        'BIOMA': lambda x: x.mode().iloc[0] if not x.empty and x.mode().size > 0 else 'N/A',
        'VPRESSAO': lambda x: x.mode().iloc[0] if not x.empty and x.mode().size > 0 else 'N/A'
    }).round(2)
    
    ranking_municipios.columns = ['Área Total (ha)', 'Qtd Alertas', 'Área Média (ha)',
                                  'Ano Min', 'Ano Max', 'Bioma Principal', 'Vetor Pressão']
    
    ranking_municipios = ranking_municipios.reset_index()
    ranking_municipios = ranking_municipios.sort_values('Área Total (ha)', ascending=False)
    ranking_municipios.insert(0, 'Posição', range(1, len(ranking_municipios) + 1))
    
    return ranking_municipios

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def obter_anos_disponiveis_desmatamento(_gdf_alertas):
    """Obtém anos disponíveis para filtro com cache."""
    if _gdf_alertas.empty or 'ANODETEC' not in _gdf_alertas.columns:
        return ['Todos']
    return ['Todos'] + sorted(_gdf_alertas['ANODETEC'].dropna().unique().tolist())

@st.cache_data(ttl=3600, show_spinner=False, max_entries=2)
def preprocessar_dados_desmatamento_temporal(_gdf_alertas):
    """Preprocessa dados para o gráfico temporal com cache."""
    if _gdf_alertas.empty:
        return pd.DataFrame()
    
    temporal_data = _gdf_alertas.copy()
    if 'AREAHA' in temporal_data.columns:
        temporal_data['AREAHA'] = pd.to_numeric(temporal_data['AREAHA'], errors='coerce')
    
    return temporal_data

@st.cache_data(ttl=3600, show_spinner=False, max_entries=3)
def calcular_bounds_desmatamento(_gdf_alertas):
    """Calcula bounds dos dados de desmatamento com cache para otimizar mapas."""
    if _gdf_alertas.empty:
        return None
    
    try:
        minx, miny, maxx, maxy = _gdf_alertas.total_bounds
        return {'lat': (miny + maxy) / 2, 'lon': (minx + maxx) / 2, 'bounds': (minx, miny, maxx, maxy)}
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False, max_entries=5)
def processar_intersecao_uc_desmatamento(_gdf_cnuc, _gdf_alertas):
    if _gdf_cnuc.empty or _gdf_alertas.empty:
        return pd.DataFrame()
    
    try:
        crs_proj = "EPSG:31983"
        gdf_cnuc_proj = _gdf_cnuc.to_crs(crs_proj)
        gdf_alertas_proj = _gdf_alertas.to_crs(crs_proj)
        
        alerts_in_ucs = gpd.sjoin(gdf_alertas_proj, gdf_cnuc_proj, how="inner", predicate="intersects")
        
        if alerts_in_ucs.empty:
            return pd.DataFrame()
        
        alert_area_per_uc = alerts_in_ucs.groupby('nome_uc', observed=False)['AREAHA'].sum().reset_index()
        alert_area_per_uc.columns = ['nome_uc', 'alerta_ha_total']
        alert_area_per_uc = alert_area_per_uc.sort_values('alerta_ha_total', ascending=False)
        
        return alert_area_per_uc
    except Exception:
        return pd.DataFrame()

with tabs[4]:
    st.header("Desmatamento")

    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre áreas de alerta de desmatamento, incluindo:
        - Distribuição por Unidade de Conservação
        - Evolução temporal
        - Distribuição por município
        - Distribuição espacial (Mapa)

        Os dados são provenientes do MapBiomas Alerta.
        """)
        st.markdown(
            "**Fonte Geral da Seção:** MapBiomas Alerta. Plataforma de Dados de Alertas de Desmatamento. Disponível em: https://alerta.mapbiomas.org/. Acesso em: maio de 2025.",
            unsafe_allow_html=True
        )

    st.write("**Filtro Global:**")
    anos_disponiveis = obter_anos_disponiveis_desmatamento(gdf_alertas_raw)
    ano_global_selecionado = st.selectbox('Ano de Detecção:', anos_disponiveis, key="filtro_ano_global")
    gdf_alertas_filtrado = processar_dados_desmatamento(gdf_alertas_raw, ano_global_selecionado)

    st.divider()

    col_charts, col_map = st.columns([2, 3], gap="large")

    with col_charts:
        if not gdf_cnuc_raw.empty and not gdf_alertas_filtrado.empty:
            dados_uc_desmatamento = processar_intersecao_uc_desmatamento(gdf_cnuc_raw, gdf_alertas_filtrado)
            
            if not dados_uc_desmatamento.empty:
                dados_uc_desmatamento['uc_wrap'] = dados_uc_desmatamento['nome_uc'].apply(lambda x: wrap_label(x, 15))
                
                fig_desmat_uc = px.bar(
                    dados_uc_desmatamento,
                    x='uc_wrap',
                    y='alerta_ha_total',
                    labels={"alerta_ha_total":"Área de Alertas (ha)","uc_wrap":"UC"},
                    text_auto=True,
                )
                
                alerta_text = [format_number_with_dots(val, 0) for val in dados_uc_desmatamento['alerta_ha_total']]
                
                fig_desmat_uc.update_traces(
                    customdata=np.stack([alerta_text, dados_uc_desmatamento.nome_uc], axis=-1),
                    hovertemplate=(
                        "<b>%{customdata[1]}</b><br>"
                        "Área de Alertas: %{customdata[0]} ha<extra></extra>" 
                    ),
                    text=alerta_text, 
                    textposition="outside", 
                    marker_line_color="rgb(80,80,80)",
                    marker_line_width=0.5,
                    cliponaxis=False
                )
                
                fig_desmat_uc = _apply_layout(fig_desmat_uc, title="Área de Alertas (Desmatamento) por UC", title_size=16)
                
                st.subheader("Área de Alertas por UC")
                st.plotly_chart(fig_desmat_uc, use_container_width=True, height=400, key="desmat_uc_chart")
                st.caption("Figura 6.1: Área total de alertas de desmatamento por unidade de conservação.")
                
                with st.expander("Detalhes e Fonte da Figura 6.1"):
                    st.write("""
                    **Interpretação:**
                    O gráfico mostra a área total (em hectares) de alertas de desmatamento detectados dentro de cada unidade de conservação.

                    **Observações:**
                    - Barras representam a área total de alertas em hectares por UC.
                    - Ordenado por área de alertas em ordem decrescente.

                    **Fonte:** MapBiomas Alerta. *Plataforma de Dados de Alertas de Desmatamento*. Disponível em: https://alerta.mapbiomas.org/. Acesso em: maio de 2025.
                    """)
            else:
                st.info("Nenhum alerta de desmatamento encontrado sobrepondo as Unidades de Conservação para o período selecionado.")
        else:
            st.warning("Dados de Unidades de Conservação ou Alertas de Desmatamento não disponíveis para esta análise.")

        st.divider()

    with col_map:
        if not gdf_alertas_filtrado.empty:
            # Usar função com cache para calcular bounds
            bounds_info = calcular_bounds_desmatamento(gdf_alertas_filtrado)
            if bounds_info:
                fig_desmat_map_pts = fig_desmatamento_mapa_pontos(gdf_alertas_filtrado)
                if fig_desmat_map_pts and fig_desmat_map_pts.data:
                    st.subheader("Mapa de Alertas")
                    st.plotly_chart(
                        fig_desmat_map_pts,
                        use_container_width=True,
                        height=850,
                        config={'scrollZoom': True},
                        key="desmat_mapa_pontos_chart"
                    )
                    st.caption("Figura 6.3: Distribuição espacial dos alertas de desmatamento.")
                    with st.expander("Detalhes e Fonte da Figura"):
                        st.write("""
                        **Interpretação:**
                        O mapa mostra a localização e a área (representada pelo tamanho e cor do ponto) dos alertas de desmatamento.

                        **Observações:**
                        - Cada ponto representa um alerta de desmatamento.
                        - O tamanho e a cor do ponto são proporcionais à área desmatada (em hectares).
                        - Áreas com maior concentração de pontos indicam maior atividade de desmatamento.

                        **Fonte:** MapBiomas Alerta. *Plataforma de Dados de Alertas de Desmatamento*. Disponível em: https://alerta.mapbiomas.org/. Acesso em: maio de 2025.
                        """)
                else:
                    st.info("Dados de alertas de desmatamento não contêm informações geográficas válidas para o mapa no período selecionado.")
            else:
                st.info("Dados de alertas de desmatamento não contêm informações geográficas válidas para o mapa no período selecionado.")
        else:
            st.warning("Dados de Alertas de Desmatamento não disponíveis para esta análise.")

    st.divider()
    st.subheader("Ranking de Municípios por Desmatamento")
    if not gdf_alertas_filtrado.empty:
        # Usar função com cache para calcular ranking
        ranking_municipios = calcular_ranking_municipios_desmatamento(gdf_alertas_filtrado)
        
        if not ranking_municipios.empty:
            # Aplicar formatação apenas nos valores para exibição
            ranking_display = ranking_municipios.copy()
            ranking_display['Área Total (ha)'] = ranking_display['Área Total (ha)'].apply(lambda x: format_number_with_dots(x, 2))
            ranking_display['Área Média (ha)'] = ranking_display['Área Média (ha)'].apply(lambda x: f"{x:.2f}".replace('.', ','))

            st.dataframe(
                ranking_display.head(10),
                use_container_width=True,
                hide_index=True,
                height=400
            )
            st.caption("Tabela 6.1: Ranking dos municípios com maior área de alertas de desmatamento (Top 10).")
            with st.expander("Detalhes da Tabela 6.1 e Informações das Colunas"):
                st.write("""
                **Interpretação:**
                Ranking dos municípios ordenados pela área total de alertas de desmatamento detectados, com informações complementares sobre quantidade de alertas, período e características predominantes.

                **Informações das Colunas:**
                - **Posição**: Ranking baseado na área total de desmatamento
                - **Estado**: Estado onde se localiza o município
                - **Município**: Município onde se localiza o alerta
                - **Área Total (ha)**: Soma de todas as áreas de alertas do município em hectares
                - **Qtd Alertas**: Quantidade total de alertas detectados no município
                - **Área Média (ha)**: Área média por alerta no município
                - **Ano Min/Max**: Período de detecção dos alertas (primeiro e último ano)
                - **Bioma Principal**: Bioma mais frequente nos alertas do município
                - **Vetor Pressão**: Principal vetor de pressão detectado nos alertas

                **Fonte:** MapBiomas Alerta. *Plataforma de Dados de Alertas de Desmatamento*. Disponível em: https://alerta.mapbiomas.org/. Acesso em: maio de 2025.
                """)
        else:
            st.info("Dados insuficientes para gerar o ranking de municípios.")
    else:
        st.info("Dados não disponíveis para o ranking no período selecionado")

    st.divider()

    if not gdf_alertas_raw.empty:
        # Usar dados preprocessados com cache
        dados_temporais = preprocessar_dados_desmatamento_temporal(gdf_alertas_raw)
        if not dados_temporais.empty:
            fig_desmat_temp = fig_desmatamento_temporal(dados_temporais)
            if fig_desmat_temp and fig_desmat_temp.data:
                st.subheader("Evolução Temporal de Alertas")
                st.plotly_chart(fig_desmat_temp, use_container_width=True, height=400, key="desmat_temporal_chart")
                st.caption("Figura 6.4: Evolução mensal da área total de alertas de desmatamento.")
                with st.expander("Detalhes e Fonte da Figura 6.4"):
                    st.write("""
                    **Interpretação:**
                    O gráfico de linha mostra a variação mensal da área total (em hectares) de alertas de desmatamento ao longo do tempo.

                    **Observações:**
                    - Cada ponto representa a soma da área de alertas para um determinado mês.
                    - A linha conecta os pontos para mostrar a tendência temporal.
                    - Valores são exibidos acima de cada ponto para facilitar a leitura.

                    **Fonte:** MapBiomas Alerta. *Plataforma de Dados de Alertas de Desmatamento*. Disponível em: https://alerta.mapbiomas.org/. Acesso em: maio de 2025.
                    """)
            else:
                st.info("Dados de alertas de desmatamento não contêm informações temporais válidas.")
        else:
            st.info("Dados de alertas de desmatamento não contêm informações temporais válidas.")
    
    # Dados Completos
    st.divider()
    st.markdown("### 📊 Dados Completos")
    st.markdown("**Dados brutos de alertas de desmatamento:**")
    if not gdf_alertas_raw.empty:
        df_alertas_display = gdf_alertas_raw.drop(columns=['geometry']) if 'geometry' in gdf_alertas_raw.columns else gdf_alertas_raw
        st.dataframe(df_alertas_display, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhum dado de alertas de desmatamento disponível.")

