import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import unicodedata
import os
import numpy as np

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

px.set_mapbox_access_token(os.getenv("MAPBOX_TOKEN"))

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

px.defaults.color_discrete_sequence = (
    px.colors.qualitative.Pastel   
  + px.colors.qualitative.Pastel1  
  + px.colors.qualitative.Pastel2  
)

px.defaults.template = "pastel"

_original_px_bar = px.bar

st.title("Análise de Conflitos em Áreas Protegidas e Territórios Tradicionais")
st.markdown("Monitoramento integrado de sobreposições em Unidades de Conservação, Terras Indígenas e Territórios Quilombolas")
st.markdown("---")

def _patched_px_bar(*args, **kwargs) -> go.Figure:
    fig: go.Figure = _original_px_bar(*args, **kwargs)
    seq = px.defaults.color_discrete_sequence
    barmode = fig.layout.barmode or ''
    barras = [t for t in fig.data if isinstance(t, go.Bar)]
    if barmode == 'stack':
        for i, trace in enumerate(barras):
            trace.marker.color = seq[i % len(seq)]
    else:
        if len(barras) == 1:
            trace = barras[0]
            vals = trace.x if trace.orientation != 'h' else trace.y
            trace.marker.color = [seq[i % len(seq)] for i in range(len(vals))]
        else:
            for i, trace in enumerate(barras):
                trace.marker.color = seq[i % len(seq)]
    return fig

px.bar = _patched_px_bar

@st.cache_data
def carregar_shapefile(caminho: str, calcular_percentuais: bool = True) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(caminho)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    gdf = gdf[gdf["geometry"].notnull() & gdf["geometry"].is_valid]
    gdf_proj = gdf.to_crs("EPSG:31983")
    gdf_proj["area_calc_km2"] = gdf_proj.geometry.area / 1e6
    if "area_km2" in gdf.columns:
        gdf["area_km2"] = gdf["area_km2"].replace(0, None).fillna(gdf_proj["area_calc_km2"])
    else:
        gdf["area_km2"] = gdf_proj["area_calc_km2"]
    if calcular_percentuais:
        gdf["perc_alerta"] = (gdf.get("alerta_km2", 0) / gdf["area_km2"]) * 100
        gdf["perc_sigef"] = (gdf.get("sigef_km2", 0) / gdf["area_km2"]) * 100
    else:
        gdf["perc_alerta"] = 0
        gdf["perc_sigef"] = 0
    gdf["id"] = gdf.index.astype(str)
    return gdf.to_crs("EPSG:4326")

def preparar_hectares(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Adiciona colunas em hectares ao GeoDataFrame."""
    gdf2 = gdf.copy()
    gdf2['alerta_ha'] = gdf2['alerta_km2'] * 100
    gdf2['sigef_ha']  = gdf2['sigef_km2']  * 100
    gdf2['area_ha']   = gdf2['area_km2']   * 100
    return gdf2

@st.cache_data
def load_csv(caminho: str) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    df = df.rename(columns={"Unnamed: 0": "Município"})
    cols = [
        "Áreas de conflitos", "Assassinatos", "Conflitos por Terra",
        "Ocupações Retomadas", "Tentativas de Assassinatos", "Trabalho Escravo"
    ]
    df["total_ocorrencias"] = df[cols].sum(axis=1)
    return df

@st.cache_data
def carregar_dados_conflitos_municipio(arquivo_excel: str) -> pd.DataFrame:
    df = pd.read_excel(arquivo_excel, sheet_name='Áreas em Conflito').dropna(how='all')
    df['mun'] = df['mun'].apply(lambda x: [
        unicodedata.normalize('NFD', str(m).lower()).encode('ascii','ignore').decode().strip().title()
        for m in str(x).split(',')
    ])
    df2 = df.explode('mun')
    df2['Famílias'] = pd.to_numeric(df2['Famílias'], errors='coerce').fillna(0)
    df2['num_mun'] = df2.groupby('Nome do Conflito')['mun'].transform('nunique')
    df2['Fam_por_mun'] = df2['Famílias'] / df2['num_mun']
    res = df2.groupby('mun').agg({'Fam_por_mun':'sum','Nome do Conflito':'count'}).reset_index()
    res.columns = ['Município','Total_Famílias','Número_Conflitos']
    return res

def criar_figura(ids_selecionados, invadindo_opcao):
    fig = px.choropleth_mapbox(
        gdf_cnuc,
        geojson=gdf_cnuc.__geo_interface__,
        locations="id",
        hover_data=["nome_uc", "municipio", "perc_alerta", "perc_sigef", "alerta_km2", "sigef_km2", "area_km2"],
        mapbox_style="open-street-map",
        center=centro,
        zoom=4,
        opacity=0.7
    )
    
    if ids_selecionados:
        ids_selecionados = list(set(ids_selecionados))
        gdf_sel = gdf_cnuc[gdf_cnuc["id"].isin(ids_selecionados)]
        fig_sel = px.choropleth_mapbox(
            gdf_sel,
            geojson=gdf_cnuc.__geo_interface__,
            locations="id",
            hover_data=["nome_uc", "municipio", "perc_alerta", "perc_sigef", "alerta_km2", "sigef_km2", "area_km2"],
            mapbox_style="open-street-map",
            center=centro,
            zoom=4,
            opacity=0.8
        )
        for trace in fig_sel.data:
            fig.add_trace(trace)
    
    if invadindo_opcao is not None:
        gdf_sigef_filtrado = gdf_sigef if invadindo_opcao.lower() == "todos" else gdf_sigef[gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()]
        trace_sigef = go.Choroplethmapbox(
            geojson=gdf_sigef_filtrado.__geo_interface__,
            locations=gdf_sigef_filtrado["id_sigef"],
            z=[1] * len(gdf_sigef_filtrado),
            colorscale=[[0, "#FF4136"], [1, "#FF4136"]],
            marker_opacity=0.5,
            marker_line_width=1,
            showlegend=False,
            showscale=False
        )
        fig.add_trace(trace_sigef)
    
    df_csv_unique = df_csv.drop_duplicates(subset=['Município'])
    
    cidades = df_csv_unique["Município"].unique()
    cores_paleta = px.colors.qualitative.Pastel
    color_map = {cidade: cores_paleta[i % len(cores_paleta)] for i, cidade in enumerate(cidades)}
    
    for cidade in cidades:
        df_cidade = df_csv_unique[df_csv_unique["Município"] == cidade]
        base_size = list(df_cidade["total_ocorrencias"] * 10)
        outline_size = [s + 4 for s in base_size]
        
        trace_cpt_outline = go.Scattermapbox(
            lat=df_cidade["Latitude"],
            lon=df_cidade["Longitude"],
            mode="markers",
            marker=dict(size=outline_size, color="black", sizemode="area", opacity=0.8),
            hoverinfo="none",
            showlegend=False
        )
        
        trace_cpt = go.Scattermapbox(
            lat=df_cidade["Latitude"],
            lon=df_cidade["Longitude"],
            mode="markers",
            marker=dict(size=base_size, color=color_map[cidade], sizemode="area"),
            text=df_cidade.apply(lambda linha: f"Município: {linha['Município']}<br>Áreas de conflitos: {linha['Áreas de conflitos']}<br>Assassinatos: {linha['Assassinatos']}", axis=1),
            hoverinfo="text",
            name=f"Ocorrências - {cidade}",
            showlegend=True
        )

        fig.add_trace(trace_cpt_outline)
        fig.add_trace(trace_cpt)
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#CCC",
            borderwidth=1,
            font=dict(size=10)
        ),
        height=700
    )
    return fig

def criar_cards(ids_selecionados, invadindo_opcao):
    try:
        ucs_selecionadas = gdf_cnuc[gdf_cnuc["id"].isin(ids_selecionados)] if ids_selecionados else gdf_cnuc.copy()
        
        if ucs_selecionadas.empty:
            return (0.0, 0.0, 0, 0, 0)

        crs_proj = "EPSG:31983"
        ucs_proj = ucs_selecionadas.to_crs(crs_proj)
        sigef_proj = gdf_sigef.to_crs(crs_proj)

        if invadindo_opcao and invadindo_opcao.lower() != "todos":
            mascara = sigef_proj["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()
            sigef_filtrado = sigef_proj[mascara].copy()
        else:
            sigef_filtrado = sigef_proj.copy()

        sobreposicao = gpd.overlay(
            ucs_proj,
            sigef_filtrado,
            how='intersection',
            keep_geom_type=False,
            make_valid=True
        )
        sobreposicao['area_sobreposta'] = sobreposicao.geometry.area / 1e6
        total_sigef = sobreposicao['area_sobreposta'].sum()
        total_area_ucs = ucs_proj.geometry.area.sum() / 1e6
        total_alerta = ucs_selecionadas["alerta_km2"].sum()

        perc_alerta = (total_alerta / total_area_ucs * 100) if total_area_ucs > 0 else 0
        perc_sigef = (total_sigef / total_area_ucs * 100) if total_area_ucs > 0 else 0

        municipios = set()
        for munic in ucs_selecionadas["municipio"]:
            partes = str(munic).replace(';', ',').split(',')
            for parte in partes:
                if parte.strip():
                    municipios.add(parte.strip().title())

        return (
            round(perc_alerta, 1),
            round(perc_sigef, 1),
            len(municipios),
            int(ucs_selecionadas["c_alertas"].sum()),
            int(sobreposicao.shape[0])
        ) 

    except Exception as e:
        st.error(f"Erro crítico: {str(e)}")
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
    
    with col1:
        st.markdown(
            card_html_template.format(
                titulo="Alertas / Ext. Ter.",
                valor=f"{perc_alerta:.1f}%",
                descricao="Área de alertas sobre extensão territorial"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card_html_template.format(
                titulo="CARs / Ext. Ter.", 
                valor=f"{perc_sigef:.1f}%",
                descricao="CARs sobre extensão territorial"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            card_html_template.format(
                titulo="Municípios Abrangidos",
                valor=f"{total_unidades}",
                descricao="Total de municípios na análise"
            ),
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            card_html_template.format(
                titulo="Alertas",
                valor=f"{contagem_alerta}",
                descricao="Total de registros de alertas"
            ),
            unsafe_allow_html=True
        )

    with col5:
        st.markdown(
            card_html_template.format(
                titulo="CARs",
                valor=f"{contagem_sigef}",
                descricao="Cadastros Ambientais Rurais"
            ),
            unsafe_allow_html=True
        )

import textwrap

def truncate(text, max_chars=15):
    return text if len(text) <= max_chars else text[:max_chars-3] + "..."

def wrap_label(name, width=30):
    return "<br>".join(textwrap.wrap(name, width))

def fig_sobreposicoes(gdf_cnuc_ha):
    gdf = gdf_cnuc_ha.copy().sort_values("area_ha", ascending=False)
    gdf["uc_short"] = gdf["nome_uc"].apply(lambda x: wrap_label(x, 15))
    
    fig = px.bar(
        gdf,
        x="uc_short",
        y=["alerta_ha","sigef_ha","area_ha"],
        labels={"value":"Área (ha)","uc_short":"UC"},
        barmode="stack",
        text_auto=True,
    )
    fig.update_traces(
        customdata=np.stack([gdf.alerta_ha, gdf.sigef_ha, gdf.area_ha, gdf.nome_uc], axis=-1),
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "Alerta: %{customdata[0]:.0f} ha<br>"
            "CAR:     %{customdata[1]:.0f} ha<br>"
            "Total:   %{customdata[2]:.0f} ha<extra></extra>"
        ),
        texttemplate="%{y:.0f}",
        textposition="inside",
        marker_line_color="rgb(80,80,80)",
        marker_line_width=0.5,
    )
    media = gdf["area_ha"].mean()
    fig.add_shape(
        type="line", x0=-0.5, x1=len(gdf["uc_short"])-0.5,
        y0=media, y1=media,
        line=dict(color="FireBrick", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=len(gdf["uc_short"])-0.5, y=media,
        text=f"Média = {media:.0f} ha",
        showarrow=False, yshift=10,
        font=dict(color="FireBrick", size=10)
    )
    fig.update_xaxes(tickangle=0, tickfont=dict(size=9), title_text="")
    fig.update_yaxes(title_text="Área (ha)", tickfont=dict(size=9))
    fig.update_layout(height=400)
    return _apply_layout(fig, title="Áreas por UC", title_size=16)

def fig_contagens_uc(gdf_cnuc: gpd.GeoDataFrame) -> go.Figure:
    gdf = gdf_cnuc.copy()
    gdf["total_counts"] = gdf["c_alertas"] + gdf["c_sigef"]
    gdf = gdf.sort_values("total_counts", ascending=False)
    
    def wrap_label(name, width=15):
        return "<br>".join(textwrap.wrap(name, width))
    gdf["uc_wrap"] = gdf["nome_uc"].apply(lambda x: wrap_label(x, 15))
    
    fig = px.bar(
        gdf,
        x="uc_wrap",
        y=["c_alertas","c_sigef"],
        labels={"value":"Contagens","uc_wrap":"UC"},
        barmode="stack",
        text_auto=True,
    )
    
    fig.update_traces(
        customdata=np.stack([gdf.c_alertas, gdf.c_sigef, gdf.total_counts, gdf.nome_uc], axis=-1),
        hovertemplate=(
            "<b>%{customdata[3]}</b><br>"
            "Alertas: %{customdata[0]}<br>"
            "CARs:    %{customdata[1]}<br>"
            "Total:   %{customdata[2]}<extra></extra>"
        ),
        texttemplate="%{y:.0f}",
        textposition="inside",
        marker_line_color="rgb(80,80,80)",
        marker_line_width=0.5,
    )
    
    media = gdf["total_counts"].mean()
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(gdf["uc_wrap"])-0.5,
        y0=media, y1=media,
        line=dict(color="FireBrick", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=len(gdf["uc_wrap"])-0.5, y=media,
        text=f"Média = {media:.0f}",
        showarrow=False, yshift=10,
        font=dict(color="FireBrick", size=10)
    )
    
    fig.update_xaxes(tickangle=0, tickfont=dict(size=9), title_text="")
    fig.update_yaxes(title_text="Contagens", tickfont=dict(size=9))
    fig.update_layout(height=400)
    
    return _apply_layout(fig, title="Contagens por UC", title_size=16)

def fig_ocupacoes(df_csv: pd.DataFrame) -> go.Figure:
    df = (
        df_csv
        .sort_values('Áreas de conflitos', ascending=False)
        .reset_index(drop=True)
    )
    df['Mun_wrap'] = df['Município'].apply(lambda x: wrap_label(x, width=20))
    seq = px.defaults.color_discrete_sequence
    bar_colors = [seq[i % len(seq)] for i in range(len(df))]

    fig = px.bar(
        df,
        x='Áreas de conflitos',
        y='Mun_wrap',
        orientation='h',
        text='Áreas de conflitos',
        labels={
            'Áreas de conflitos': 'Ocupações Retomadas',
            'Mun_wrap': 'Município'
        },
    )

    fig.update_traces(
        marker=dict(
            color=bar_colors,
            line_color='rgb(80,80,80)',
            line_width=0.5
        ),
        texttemplate='%{text:.0f}',
        textposition='outside'
    )
    avg = df['Áreas de conflitos'].mean()
    fig.add_shape(
        type='line',
        x0=avg, x1=avg,
        yref='paper', y0=0, y1=1,
        line=dict(color='FireBrick', width=2, dash='dash')
    )
    fig.add_annotation(
        x=avg, y=1.02,
        xref='x', yref='paper',
        text=f"Média = {avg:.1f}",
        showarrow=False,
        font=dict(color='FireBrick', size=10)
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder='array',
            categoryarray=df['Mun_wrap'][::-1]
        )
    )
    fig = _apply_layout(fig, title="Ocupações Retomadas por Município", title_size=18)
    fig.update_layout(
        height=450,
        margin=dict(l=150, r=20, t=60, b=20)
    )

    return fig

def fig_familias(df_conflitos: pd.DataFrame) -> go.Figure:
    df = df_conflitos.sort_values('Total_Famílias', ascending=False)
    max_val = df['Total_Famílias'].max()

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
            tickformat=',d'                 
        ),
        margin=dict(l=80, r=100, t=50, b=20) 
    )

    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        cliponaxis=False,                 
        marker_line_color='rgb(80,80,80)',
        marker_line_width=0.5
    )

    return fig

def fig_conflitos(df_conflitos: pd.DataFrame) -> go.Figure:
    df = df_conflitos.sort_values('Número_Conflitos', ascending=False)
    fig = px.bar(
        df, x='Número_Conflitos', y='Município', orientation='h',
        text='Número_Conflitos'
    )
    fig = _apply_layout(fig, title="Conflitos Registrados")
    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )
    fig.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside',
        marker_line_color='rgb(80,80,80)',
        marker_line_width=0.5
    )
    return fig

def clean_text(text: str) -> str:
    if pd.isna(text): return text
    text = str(text).strip().lower()
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def fig_justica(df_proc: pd.DataFrame) -> dict[str, go.Figure]:
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
    df_proc['municipio'] = df_proc['municipio'].apply(clean_text)
    top = df_proc['municipio'].value_counts().head(10).reset_index()
    top.columns = ['Municipio', 'Quantidade']
    top['label'] = top['Municipio'].apply(lambda x: wrap_label(x, 20))
    fig_mun = px.bar(
        top, y='label', x='Quantidade', orientation='h',
        color_discrete_sequence=palette
    )
    fig_mun.update_traces(texttemplate='%{x}', textposition='auto', cliponaxis=False)
    fig_mun.update_layout(
        margin=dict(l=150, r=60, t=50, b=bottom_margin),
        height=500,
        yaxis=dict(autorange="reversed")
    )
    figs['mun'] = _apply_layout(fig_mun, "Top 10 Municípios com Mais Processos", 16)

    # Evolução Mensal de Processos
    df_proc['ano_mes'] = (
        pd.to_datetime(df_proc['data_ajuizamento'], errors='coerce')
          .dt.to_period('M')
          .dt.to_timestamp()
    )
    mensal = df_proc.groupby('ano_mes').size().reset_index(name='Quantidade')

    fig_temp = px.line(
        mensal,
        x='ano_mes', y='Quantidade',
        markers=True, text='Quantidade'
    )
    fig_temp.update_traces(
        mode='lines+markers+text',
        textposition='top center',
        texttemplate='%{text}'
    )
    fig_temp.update_layout(
        margin=dict(l=80, r=60, t=50, b=bottom_margin),
        height=400,
        yaxis=dict(range=[0, mensal['Quantidade'].max() * 1.1])
    )
    figs['temp'] = _apply_layout(fig_temp, "Evolução Mensal de Processos", 16)

    # Top 10 Classes, Assuntos e Órgãos
    mappings = [
        ('class', 'classe', 'Top 10 Classes Processuais', mapa_classes),
        ('ass', 'assuntos', 'Top 10 Assuntos', mapa_assuntos),
        ('org', 'orgao_julgador', 'Top 10 Órgãos Julgadores', mapa_orgaos)
    ]

    for key, col, title, mapa in mappings:
        df_proc[col] = df_proc[col].apply(clean_text)
        df = (
            df_proc[col]
            .replace(mapa)
            .value_counts()
            .head(10)
            .reset_index()
        )
        df.columns = [col, 'Quantidade']
        df['label'] = df[col].apply(lambda x: wrap_label(x, 30))
        fig = px.bar(
            df, y='label', x='Quantidade', orientation='h',
            color_discrete_sequence=palette
        )
        fig.update_traces(texttemplate='%{x}', textposition='auto', cliponaxis=False)
        fig.update_layout(
            margin=dict(l=180, r=60, t=50, b=bottom_margin),
            height=500,
            yaxis=dict(autorange="reversed")
        )
        figs[key] = _apply_layout(fig, title, 16)

    return figs

def corrige_coord(x):
    if pd.isna(x):
        return np.nan
    return x / 1e5 if abs(x) > 180 else x

@st.cache_data
def carregar_dados_fogo(
    caminho_csv: str = r"Areas_de_interesse_ordenado.csv",
    sep: str = ';',
    encoding: str = 'latin1'
) -> pd.DataFrame:
    try:
        df = pd.read_csv(caminho_csv, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(caminho_csv, sep=sep, encoding='utf-8', errors='replace')
    df['DataHora'] = pd.to_datetime(df['DataHora'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DataHora'])
    df['date'] = df['DataHora'].dt.date
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce').map(lambda x: x/1e5 if abs(x)>180 else x)
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce').map(lambda x: x/1e5 if abs(x)>180 else x)
    for col in ['DiaSemChuva','Precipitacao','RiscoFogo','FRP']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col]==-999, col] = np.nan
    df = df.dropna(subset=['Latitude','Longitude'])
    minx, miny, maxx, maxy = gdf_cnuc.total_bounds
    df = df[(df.Latitude >= miny) & (df.Latitude <= maxy) & (df.Longitude >= minx) & (df.Longitude <= maxx)]
    return df

def criar_figuras_fogo(df: pd.DataFrame, ano: int | None = None) -> dict[str, go.Figure]:
    if ano is not None:
        df = df[df['date'].dt.year == ano]
    df = (
        df
        .dropna(subset=['date','RiscoFogo','Precipitacao','Municipio','Latitude','Longitude'])
        .assign(date=lambda d: pd.to_datetime(d['date'], errors='coerce'))
    )
    cores = px.defaults.color_discrete_sequence

    daily = df.groupby(df['date'].dt.date).size().rename('count')
    rolling = daily.rolling(window=7, min_periods=1).mean().rename('m7')
    ts_df = pd.concat([daily, rolling], axis=1).reset_index().rename(columns={'index':'date'})

    monthly = df.set_index('date').resample('M').size().rename('count')
    rolling_monthly = monthly.rolling(window=3, min_periods=1).mean().rename('m3')
    ts_month = pd.concat([monthly, rolling_monthly], axis=1).reset_index()
    ts_month.columns = ['date','count','m3']

    annual = df.set_index('date').resample('Y').size().rename('count')
    rolling_annual = annual.rolling(window=2, min_periods=1).mean().rename('m2')
    ts_ann = pd.concat([annual, rolling_annual], axis=1).reset_index()
    ts_ann.columns = ['date','count','m2']

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_df['date'], y=ts_df['count'],
        mode='lines+markers+text',
        line=dict(shape='spline', width=2, smoothing=1.2, color=cores[0]),
        marker=dict(size=6, color=cores[1], line=dict(width=1, color='black')),
        text=ts_df['count'], textposition='top center', texttemplate='%{text}',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Focos: %{y}<extra></extra>'
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_df['date'], y=ts_df['m7'],
        mode='lines', line=dict(color=cores[2], width=3),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Média7d: %{y:.1f}<extra></extra>'
    ))
    fig_ts.update_layout(
        xaxis=dict(
            type='date',
            range=[ts_df['date'].min(), ts_df['date'].max()],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all', label='All')
                ]
            ),
            showgrid=False
        ),
        yaxis=dict(title='Número de Focos', gridcolor='lightgrey'),
        margin=dict(l=40, r=20, t=60, b=40),
        height=450,
        legend=dict(orientation='h', y=1.02, x=1)
    )
    figs = {'ts': fig_ts}

    top = df['Municipio'].value_counts().head(10).rename_axis('Município').reset_index(name='Focos')
    top['Mun_wrap'] = top['Município'].apply(lambda x: x)
    top['Categoria'] = top['Focos'].rank(method='first', ascending=False).apply(lambda r: 'Top 3' if r<=3 else 'Outros')
    fig_top = px.bar(
        top.sort_values('Focos', ascending=True),
        x='Focos', y='Mun_wrap', orientation='h', color='Categoria', text='Focos',
        color_discrete_map={'Top 3': cores[2], 'Outros': cores[3]}
    )
    fig_top.update_traces(texttemplate='%{text}', textposition='outside')
    fig_top.update_layout(margin=dict(l=150, r=20, t=60, b=40), height=400)
    figs['top_municipios'] = fig_top

    df_map = df[(df['Latitude']>-90)&(df['Latitude']<90)&(df['Longitude']>-180)&(df['Longitude']<180)]
    p99 = df_map['Precipitacao'].quantile(0.99)
    r99 = df_map['RiscoFogo'].quantile(0.99)
    df_s = df_map[(df_map['Precipitacao']>=0)&(df_map['Precipitacao']<=p99)&(df_map['RiscoFogo']>=0)&(df_map['RiscoFogo']<=r99)]
    fig_map = px.scatter_mapbox(
        df_s, lat='Latitude', lon='Longitude', color='RiscoFogo',
        color_continuous_scale='YlOrRd', size=None,
        hover_name='Municipio', hover_data={'date':True,'RiscoFogo':True},
        zoom=5, height=400, template=None
    )
    fig_map.update_traces(marker=dict(size=8, opacity=0.7), marker_showscale=True)
    fig_map.update_layout(mapbox=dict(style='open-street-map'), margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    figs['scatter_prec_risco'] = fig_map

    return figs

def app_fogo(caminho_csv: str, sep: str = ';', encoding: str = 'latin1'):
    df_fogo = carregar_dados_fogo(caminho_csv, sep=sep, encoding=encoding)
    figs = criar_figuras_fogo(df_fogo)
    
    st.sidebar.header("Focos de Calor")
    opcao = st.sidebar.selectbox(
        "Selecione um gráfico:",
        ["Série Temporal", "Histograma de Risco", "Precip x Risco", "Top Municípios"]
    )
    st.header("Análise de Focos de Calor")

    if opcao == "Série Temporal":
        st.plotly_chart(figs['ts'], use_container_width=True)
    elif opcao == "Histograma de Risco":
        st.plotly_chart(figs['hist_risco'], use_container_width=True)
    elif opcao == "Precip x Risco":
        st.plotly_chart(figs['scatter_prec_risco'], use_container_width=True)
    else:
        st.plotly_chart(figs['top_municipios'], use_container_width=True)

@st.cache_data(show_spinner=False)
def load_inpe(filepaths: list[str], chunksize: int = 100_000) -> pd.DataFrame:
    cols = ['RiscoFogo', 'Precipitacao', 'mun_corrigido', 'DiaSemChuva', 'Latitude', 'Longitude']
    dfs = []
    total_chunks = 0
    progress = st.progress(0)
    for path in filepaths:
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
        delim = ',' if sample.count(',') > sample.count(';') else ';'
        for i, chunk in enumerate(pd.read_csv(path, sep=delim, usecols=cols, iterator=True, chunksize=chunksize, encoding='utf-8')):
            chunk = chunk.dropna()
            chunk = chunk[chunk['RiscoFogo'] > 0]
            dfs.append(chunk)
            total_chunks += 1
            progress.progress(min(total_chunks / (20 * len(filepaths)), 1.0))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=cols)

def graficos_inpe(df: pd.DataFrame) -> dict:
    figs = {}
    # Top 10 municípios por risco de fogo
    top_risco = df.groupby('mun_corrigido')['RiscoFogo'].mean().nlargest(10).sort_values()
    fig_risco = go.Figure(go.Bar(
        y=top_risco.index,
        x=top_risco.values,
        orientation='h',
        marker_color='#AECBFA',
        text=top_risco.values,
        texttemplate='<b>%{text:.2f}</b>', 
        textposition='outside',             
    ))
    fig_risco.update_layout(
        title='Top Municípios - Risco de Fogo',
        xaxis_title='Risco Médio',
        height=400,
        margin=dict(l=60, r=80, t=50, b=40) 
    )
    figs['top_risco'] = fig_risco

    # Top 10 municípios por precipitação
    top_precip = df.groupby('mun_corrigido')['Precipitacao'].mean().nlargest(10).sort_values()
    fig_precip = go.Figure(go.Bar(
        y=top_precip.index,
        x=top_precip.values,
        orientation='h',
        marker_color='#FFE0B2',
        text=top_precip.values,
        texttemplate='<b>%{text:.1f} mm</b>', 
        textposition='outside',
    ))
    fig_precip.update_layout(
        title='Top Municípios - Precipitação',
        xaxis_title='Precipitação Média (mm)',
        height=400,
        margin=dict(l=60, r=80, t=50, b=40)
    )
    figs['top_precip'] = fig_precip

    # Mapa de dispersão (sem alterações)
    max_points = 50_000
    df_plot = df.sample(max_points, random_state=1) if len(df) > max_points else df

    lat_min, lat_max = df_plot['Latitude'].min(), df_plot['Latitude'].max()
    lon_min, lon_max = df_plot['Longitude'].min(), df_plot['Longitude'].max()
    centro = {'lat': (lat_min + lat_max) / 2, 'lon': (lon_min + lon_max) / 2}
    span = max(lat_max - lat_min, lon_max - lon_min)
    zoom = 10 if span < 1 else 8 if span < 5 else 6 if span < 10 else 4

    fig_map = px.scatter_mapbox(
        df_plot,
        lat='Latitude',
        lon='Longitude',
        color='RiscoFogo',
        size='Precipitacao',
        hover_name='mun_corrigido',
        size_max=15,
        color_continuous_scale='orrd',
        zoom=zoom,
        center=centro
    )
    fig_map.update_layout(
        mapbox=dict(style='open-street-map'),
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=False
    )
    figs['mapa'] = fig_map

    return figs

gdf_cnuc = carregar_shapefile(
    r"cnuc.shp"
)
gdf_cnuc_ha = preparar_hectares(gdf_cnuc)
gdf_sigef = carregar_shapefile(
    r"sigef.shp",
    calcular_percentuais=False
)
gdf_sigef   = gdf_sigef.rename(columns={"id":"id_sigef"})
limites = gdf_cnuc.total_bounds
centro = {
    "lat": (limites[1] + limites[3]) / 2,
    "lon": (limites[0] + limites[2]) / 2
}
df_csv     = load_csv(
    r"CPT-PA-count.csv"
)
df_confmun = carregar_dados_conflitos_municipio(
    r"CPTF-PA.xlsx"
)
df_proc    = pd.read_csv(
    r"processos_tjpa_completo_atualizada_pronto.csv",
    sep=";", encoding="windows-1252"
)

with st.sidebar:
    st.header("⚙️ Filtros Principais") 
    st.subheader("Área de Interesse")
    opcoes_invadindo = ["Selecione", "Todos"] + sorted(
        gdf_sigef["invadindo"].str.strip().unique().tolist()
    )
    invadindo_opcao = st.selectbox(
        "Tipo de sobreposição:",
        opcoes_invadindo,
        index=0,
        help="Selecione o tipo de área sobreposta para análise"
    )

if invadindo_opcao == "Selecione":
    invadindo_opcao = None

if invadindo_opcao and invadindo_opcao.lower() != "todos":
    gdf_filtrado = gpd.sjoin(
        gdf_cnuc,
        gdf_sigef[
            gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.lower()
        ],
        how="inner", 
        predicate="intersects"
    )
    ids_selecionados = gdf_filtrado["id"].unique().tolist()
else:
    ids_selecionados = []

caminho_fogo = r"Areas_de_interesse_ordenado.csv"
df_fogo = carregar_dados_fogo(caminho_fogo, sep=';', encoding='latin1')
figs_fogo = criar_figuras_fogo(df_fogo)

fig_map = criar_figura(ids_selecionados, invadindo_opcao)
perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef = criar_cards(
    ids_selecionados,
    invadindo_opcao
)

tabs = st.tabs(["Sobreposições", "Queimadas", "Famílias", "Justiça", "INPE"])

with tabs[0]:
    st.header("Sobreposições")
    cols = st.columns(5, gap="small")
    titulos = [
        ("Alertas / Ext. Ter.", f"{perc_alerta:.1f}%", "Área de alertas sobre extensão territorial"),
        ("CARs / Ext. Ter.", f"{perc_sigef:.1f}%", "CARs sobre extensão territorial"),
        ("Municípios", f"{total_unidades}", "Total de municípios na análise"),
        ("Alertas", f"{contagem_alerta}", "Total de registros de alertas"),
        ("CARs", f"{contagem_sigef}", "Cadastros Ambientais Rurais")
    ]
    card_template = """
    <div style="
        background-color:#F9F9FF;
        border:1px solid #E0E0E0;
        padding:1rem;
        border-radius:8px;
        box-shadow:0 2px 5px rgba(0,0,0,0.1);
        text-align:center;
        height:100px;
        display:flex;
        flex-direction:column;
        justify-content:center;">
        <h5 style="margin:0; font-size:0.9rem;">{0}</h5>
        <p style="margin:0; font-size:1.2rem; font-weight:bold; color:#2F5496;">{1}</p>
        <small style="color:#666;">{2}</small>
    </div>
    """
    for col, (t, v, d) in zip(cols, titulos):
        col.markdown(card_template.format(t, v, d), unsafe_allow_html=True)

    st.divider()

    row1_map, row1_chart1 = st.columns([3, 2], gap="large")
    with row1_map:
        st.subheader("Mapa de Unidades")
        st.plotly_chart(fig_map, use_container_width=True)
    with row1_chart1:
        st.subheader("Áreas por UC")
        st.plotly_chart(fig_sobreposicoes(gdf_cnuc_ha), use_container_width=True, height=350)
        st.subheader("Contagens por UC")
        st.plotly_chart(fig_contagens_uc(gdf_cnuc), use_container_width=True, height=350)

    st.divider()

    st.subheader("Ocupações Retomadas")
    st.plotly_chart(fig_ocupacoes(df_csv), use_container_width=True, height=300)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

figs_fogo = criar_figuras_fogo(df_fogo)

with tabs[1]:
    st.header('Focos de Calor')

    col_filtros, _ = st.columns([1.5, 3])
    with col_filtros:
        st.markdown("### 🔎 Filtro por Ano")
        anos = pd.to_datetime(df_fogo['date'], errors='coerce').dropna().dt.year.unique().tolist()
        anos.sort()
        ano_sel = st.selectbox('Selecione o ano:', [None] + anos, format_func=lambda x: 'Todos' if x is None else str(x))

    df_fogo['date'] = pd.to_datetime(df_fogo['date'], errors='coerce')
    figs_fogo_ano = criar_figuras_fogo(df_fogo, ano=ano_sel)

    st.subheader(f"Evolução Diária — {ano_sel or 'Todos os anos'}")
    st.plotly_chart(figs_fogo_ano['ts'], use_container_width=True, height=500)

    col1, col2 = st.columns(2, gap='large')
    with col1:
        st.subheader('Top Municípios')
        st.caption("Novo progresso tem apenas um registro, falta de dados.")        
        st.plotly_chart(figs_fogo_ano['top_municipios'], use_container_width=True, height=400)
    with col2:
        st.subheader('Mapa de Focos')
        st.plotly_chart(figs_fogo_ano['scatter_prec_risco'], use_container_width=True, height=400)

with tabs[2]:
    st.header("Impacto Social")
    col_fam, col_conf = st.columns(2, gap="large")
    with col_fam:
        st.subheader("Famílias Afetadas")
        st.plotly_chart(fig_familias(df_confmun), use_container_width=True, height=400, key="familias")
    with col_conf:
        st.subheader("Conflitos Registrados")
        st.plotly_chart(fig_conflitos(df_confmun), use_container_width=True, height=400, key="conflitos")

with tabs[3]:
    st.header("Processos Judiciais")
    figs_j = fig_justica(df_proc)
    key_map = {
        "Municípios":"mun",
        "Temporal":"temp",
        "Classes":"class",
        "Assuntos":"ass",
        "Órgãos":"org"
    }

    barras = ["Municípios","Classes","Assuntos","Órgãos"]
    for i in range(0, len(barras), 2):
        cols = st.columns(2, gap="large")
        for col, key in zip(cols, barras[i:i+2]):
            chart_key = key_map[key]
            col.subheader(key)
            col.plotly_chart(
                figs_j[chart_key],
                use_container_width=True,
                height=300,
                key=f"jud_{chart_key}_{i//2}"
            )

    # ── gráfico de linha
    st.subheader("Evolução Mensal de Processos")
    st.plotly_chart(
        figs_j[key_map["Temporal"]],
        use_container_width=True,
        height=500,
        key="jud_temp_full"
    )

with tabs[4]:
    st.header("Focos de Calor")
    files = [
        "focos_municipios_filtrados_part1.csv",
        "focos_municipios_filtrados_part2.csv",
        "focos_municipios_filtrados_part3.csv",
        "focos_municipios_filtrados_part4.csv",
        "focos_municipios_filtrados_part5.csv",
        "focos_municipios_filtrados_part6.csv",
    ]
    df_inpe = load_inpe(files)
    if not df_inpe.empty:
        figs = graficos_inpe(df_inpe)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Risco e Precipitação")
            st.plotly_chart(figs['top_risco'], use_container_width=True)
            st.plotly_chart(figs['top_precip'], use_container_width=True)
        with col2:
            st.subheader("Mapa de Risco de Fogo")
            st.plotly_chart(figs['mapa'], use_container_width=True)
    else:
        st.warning("Nenhum dado disponível após filtros.")
