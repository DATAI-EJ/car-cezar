import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import unicodedata
import os
import numpy as np
import duckdb

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
    df = pd.read_csv(caminho, low_memory=False)
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
        hover_data=[
            "nome_uc", "municipio", "perc_alerta", "perc_sigef",
            "alerta_km2", "sigef_km2", "area_km2"
        ],
        mapbox_style="open-street-map",
        center=centro,
        zoom=4,
        opacity=0.7
    )
    if ids_selecionados:
        ids = list(set(ids_selecionados))
        gdf_sel = gdf_cnuc[gdf_cnuc["id"].isin(ids)]
        fig_sel = px.choropleth_mapbox(
            gdf_sel,
            geojson=gdf_cnuc.__geo_interface__,
            locations="id",
            hover_data=[
                "nome_uc", "municipio", "perc_alerta", "perc_sigef",
                "alerta_km2", "sigef_km2", "area_km2"
            ],
            mapbox_style="open-street-map",
            center=centro,
            zoom=4,
            opacity=0.8
        )
        for trace in fig_sel.data:
            fig.add_trace(trace)
    if invadindo_opcao is not None:
        filtro = (
            gdf_sigef
            if invadindo_opcao.lower() == "todos"
            else gdf_sigef[
                gdf_sigef["invadindo"]
                .str.strip()
                .str.lower() == invadindo_opcao.strip().lower()
            ]
        )
        trace_sigef = go.Choroplethmapbox(
            geojson=filtro.__geo_interface__,
            locations=filtro["id_sigef"],
            z=[1] * len(filtro),
            colorscale=[[0, "#FF4136"], [1, "#FF4136"]],
            marker_opacity=0.5,
            marker_line_width=1,
            showlegend=False,
            showscale=False
        )
        fig.add_trace(trace_sigef)
    df_csv_unique = df_csv.drop_duplicates(subset=["Município"])
    cidades = df_csv_unique["Município"].unique()
    paleta = px.colors.qualitative.Pastel
    mapa_cores = {c: paleta[i % len(paleta)] for i, c in enumerate(cidades)}
    for c in cidades:
        df_c = df_csv_unique[df_csv_unique["Município"] == c]
        base = (df_c["total_ocorrencias"] * 10).tolist()
        outline = [s + 4 for s in base]
        fig.add_trace(go.Scattermapbox(
            lat=df_c["Latitude"],
            lon=df_c["Longitude"],
            mode="markers",
            marker=dict(size=outline, color="black", sizemode="area", opacity=0.8),
            hoverinfo="none",
            showlegend=False
        ))
        fig.add_trace(go.Scattermapbox(
            lat=df_c["Latitude"],
            lon=df_c["Longitude"],
            mode="markers",
            marker=dict(size=base, color=mapa_cores[c], sizemode="area"),
            text=df_c.apply(
                lambda r: (
                    f"Município: {r['Município']}<br>"
                    f"Áreas de conflitos: {r['Áreas de conflitos']}<br>"
                    f"Assassinatos: {r['Assassinatos']}"
                ),
                axis=1
            ),
            hoverinfo="text",
            name=f"<b>Ocorrências – {c}</b>", 
            showlegend=True
        ))
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=centro,
                zoom=4
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
            legend=dict(
                x=0.01,         
                y=0.99,           
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0)", 
                bordercolor="rgba(0,0,0,0)",    
                font=dict(size=10)
            ),
            height=550
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

def fig_car_por_uc_donut(gdf_cnuc_ha: gpd.GeoDataFrame, nome_uc: str, modo_valor: str = "percent") -> go.Figure:
    if nome_uc == "Todas":
        area_total = gdf_cnuc_ha["area_ha"].sum()
        area_car = gdf_cnuc_ha["sigef_ha"].sum()
    else:
        row = gdf_cnuc_ha[gdf_cnuc_ha["nome_uc"] == nome_uc]
        if row.empty:
            raise ValueError(f"Unidade de conservação '{nome_uc}' não encontrada.")
        area_total = row["area_ha"].values[0]
        area_car = row["sigef_ha"].values[0]
    total_chart = max(area_total, area_car)
    restante_chart = total_chart - area_car
    percentual = (area_car / area_total) * 100 if area_total else 0
    if modo_valor == "percent":
        textinfo = "label+percent"
        center_text = f"{percentual:.1f}%"
    else:
        textinfo = "label+value"
        center_text = f"{area_car:,.0f} ha"
    fig = go.Figure(data=[go.Pie(
        labels=["Área CAR", "Área restante"],
        values=[area_car, restante_chart],
        hole=0.6,
        marker_colors=["#2ca02c", "#d9d9d9"],
        textinfo=textinfo,
        hoverinfo="label+value+percent"
    )])
    fig.update_layout(
        title_text=f"Ocupação do CAR em: {nome_uc}",
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=22, showarrow=False)],
        height=400
    )
    return _apply_layout(fig, title=f"Ocupação do CAR em: {nome_uc}", title_size=16)

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

    # Evolução Mensal de Processos — com bolinha e valor destacado
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
    # faz aparecer o texto acima de cada marcador
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

def graficos_inpe(df: pd.DataFrame, ano: int) -> dict:
    df = df[df['DataHora'].dt.year == ano]
    df_indexed = df.set_index('DataHora')
    df_indexed = df_indexed[df_indexed['RiscoFogo'].between(0, 1)]
    monthly = df_indexed['RiscoFogo'].resample('ME').mean().reset_index()
    monthly['RiscoFogo'] = monthly['RiscoFogo'].fillna(0)

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=monthly['DataHora'].dt.to_period('M').astype(str),
        y=monthly['RiscoFogo'],
        name='Risco de Fogo Mensal',
        mode='lines+markers+text',
        marker=dict(size=8, color='#FF4136', line=dict(width=1, color='#444')),
        line=dict(width=2, color='#FF4136'),
        text=[f'{v:.2f}' for v in monthly['RiscoFogo']],
        textposition='top center'
    ))
    fig_temp.update_layout(
        title='Evolução Mensal do Risco de Fogo',
        xaxis_title='Mês',
        yaxis_title='Risco Médio',
        height=400,
        margin=dict(l=60, r=80, t=80, b=40),
        showlegend=True,
        hovermode='x unified'
    )

    top_risco = df.groupby('mun_corrigido')['RiscoFogo'].mean().nlargest(10).sort_values()
    fig_risco = go.Figure(go.Bar(
        y=top_risco.index,
        x=top_risco.values,
        orientation='h',
        marker_color='#FF8C7A',
        text=top_risco.values,
        texttemplate='<b>%{text:.2f}</b>',
        textposition='outside'
    ))
    fig_risco.update_layout(
        title='Top Municípios - Risco de Fogo',
        xaxis_title='Risco Médio',
        height=400,
        margin=dict(l=60, r=80, t=50, b=40)
    )

    top_precip = df.groupby('mun_corrigido')['Precipitacao'].mean().nlargest(10).sort_values()
    fig_precip = go.Figure(go.Bar(
        y=top_precip.index,
        x=top_precip.values,
        orientation='h',
        marker_color='#B3D9FF',
        text=top_precip.values,
        texttemplate='<b>%{text:.1f} mm</b>',
        textposition='outside'
    ))
    fig_precip.update_layout(
        title='Top Municípios - Precipitação',
        xaxis_title='Precipitação Média (mm)',
        height=400,
        margin=dict(l=60, r=80, t=50, b=40)
    )

    df_plot = df.sample(50000, random_state=1) if len(df) > 50000 else df
    lat_min, lat_max = df_plot['Latitude'].min(), df_plot['Latitude'].max()
    lon_min, lon_max = df_plot['Longitude'].min(), df_plot['Longitude'].max()
    centro = {'lat': (lat_min + lat_max) / 2, 'lon': (lon_min + lon_max) / 2}
    span = max(lat_max - lat_min, lon_max - lon_min)
    zoom = 10 if span < 1 else 8 if span < 5 else 6 if span < 10 else 4

    fig_map = px.scatter_map(
        df_plot,
        lat='Latitude',
        lon='Longitude',
        color='RiscoFogo',
        size='Precipitacao',
        hover_name='mun_corrigido',
        size_max=15,
        color_continuous_scale=['#FFE5E5', '#FF4136'],
        zoom=zoom,
        center=centro
    )
    fig_map.update_layout(
        mapbox=dict(style='open-street-map'),
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(title='Risco de Fogo'),
        showlegend=False
    )

    return {
        'temporal': fig_temp,
        'top_risco': fig_risco,
        'top_precip': fig_precip,
        'mapa': fig_map
    }

def mostrar_tabela_unificada(gdf_alertas, gdf_sigef, gdf_cnuc):
    df_a = gdf_alertas[['MUNICIPIO', 'AREAHA']].rename(columns={'MUNICIPIO':'municipio', 'AREAHA':'alerta_ha'})
    df_s = gdf_sigef[['municipio', 'area_km2']].rename(columns={'area_km2':'sigef_ha'})
    df_s['sigef_ha'] = pd.to_numeric(df_s['sigef_ha'], errors='coerce').fillna(0) * 100
    df_c = gdf_cnuc[['municipio', 'ha_total']].rename(columns={'ha_total':'uc_ha'})

    df_alertas_mun = df_a.groupby('municipio', as_index=False)['alerta_ha'].sum()
    df_sigef_mun = df_s.groupby('municipio', as_index=False)['sigef_ha'].sum()
    df_cnuc_mun = df_c.groupby('municipio', as_index=False)['uc_ha'].sum()

    df_merged = df_alertas_mun.merge(df_sigef_mun, on='municipio', how='outer')
    df_merged = df_merged.merge(df_cnuc_mun, on='municipio', how='outer').fillna(0)

    cols = ['alerta_ha', 'sigef_ha', 'uc_ha']
    for c in cols:
        df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(0)
    
    total_alertas = df_merged['alerta_ha'].sum()
    total_sigef = df_merged['sigef_ha'].sum() 
    total_uc = df_merged['uc_ha'].sum()

    df_merged = df_merged[~((df_merged[cols] == 0).all(axis=1))]
    df_merged = df_merged.sort_values('municipio').reset_index(drop=True)
    df_merged = df_merged.rename(columns={
        'municipio': 'MUNICÍPIO',
        'alerta_ha': 'ALERTAS(HA)',
        'sigef_ha': 'SIGEF(HA)', 
        'uc_ha': 'CNUC(HA)'
    })

    total_row = pd.DataFrame([{
        'MUNICÍPIO': 'TOTAL(HA)',
        'ALERTAS(HA)': total_alertas,
        'SIGEF(HA)': total_sigef,
        'CNUC(HA)': total_uc
    }])
    
    df_merged = pd.concat([df_merged, total_row], ignore_index=True)

    styles = []
    colors = {
        'ALERTAS(HA)':'#fde0dd', 
        'SIGEF(HA)':'#e0ecf4', 
        'CNUC(HA)':'#edf8e9'
    }
    for i, c in enumerate(df_merged.columns):
        if c in colors:
            styles.append({'selector': f'td.col{i}', 'props': [('background-color', colors[c])]})
    
    styles.append({
        'selector': 'tr:last-child',
        'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0')]
    })

    styled = (
        df_merged.style
                 .format({c:'{:,.2f}' for c in ['ALERTAS(HA)', 'SIGEF(HA)', 'CNUC(HA)']})
                 .set_table_styles(styles)
                 .set_table_attributes('style="border-collapse:collapse"')
    )

    st.subheader("Tabela Área")
    st.markdown(styled.to_html(), unsafe_allow_html=True)

gdf_alertas = carregar_shapefile(
    r"alertas.shp",
    calcular_percentuais=False
)
gdf_alertas = gdf_alertas.rename(columns={"id":"id_alerta"})

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

caminho_fogo = r"Areas_de_interesse_ordenado.csv"

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

    perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef = criar_cards(None, None)
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
        opcoes_invadindo = ["Selecione", "Todos"] + sorted(gdf_sigef["invadindo"].str.strip().unique().tolist())
        invadindo_opcao_temp = st.selectbox("Tipo de sobreposição:", opcoes_invadindo, index=0, help="Selecione o tipo de área sobreposta para análise")
        invadindo_opcao = None if invadindo_opcao_temp == "Selecione" else invadindo_opcao_temp

        if invadindo_opcao and invadindo_opcao.lower() != "todos":
            gdf_filtrado = gpd.sjoin(gdf_cnuc, gdf_sigef[gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.lower()], how="inner", predicate="intersects")
            ids_selecionados = gdf_filtrado["id"].unique().tolist()
        else:
            ids_selecionados = []

        st.subheader("Mapa de Unidades")
        fig_map = criar_figura(ids_selecionados, invadindo_opcao)
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
        uc_names = ["Todas"] + sorted(gdf_cnuc_ha["nome_uc"].unique())
        nome_uc = st.selectbox("Selecione a Unidade de Conservação:", uc_names)
        modo_input = st.radio("Mostrar valores como:", ["Hectares (ha)", "% da UC"], horizontal=True)
        modo = "absoluto" if modo_input == "Hectares (ha)" else "percent"
        fig = fig_car_por_uc_donut(gdf_cnuc_ha, nome_uc, modo)
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
        st.plotly_chart(fig_sobreposicoes(gdf_cnuc_ha), use_container_width=True, height=350)
        st.caption("Figura 1.3: Distribuição de áreas por unidade de conservação.")
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
        st.plotly_chart(fig_contagens_uc(gdf_cnuc), use_container_width=True, height=350)
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
        <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Visualização unificada dos dados de alertas, SIGEF e CNUC.</p>
    </div>""", unsafe_allow_html=True)
    mostrar_tabela_unificada(gdf_alertas, gdf_sigef, gdf_cnuc)
    st.caption("Tabela 1.1: Dados consolidados por município.")
    with st.expander("Detalhes e Fonte da Tabela 1.1"):
        st.write("""
        **Interpretação:**  
        A tabela apresenta os dados consolidados por município, incluindo:
        - Área de alertas em hectares
        - Área do SIGEF em hectares
        - Área do CNUC em hectares

        **Observações:**
        - Valores em hectares
        - Totais na última linha
        - Células coloridas por tipo de dado

        **Fonte:** MMA - Ministério do Meio Ambiente. *Cadastro Nacional de Unidades de Conservação*. Brasília: MMA, 2025. Disponível em: https://www.gov.br/mma/. Acesso em: maio de 2025.
        """)
    st.divider()

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

    col_fam, col_conf = st.columns(2, gap="large")
    with col_fam:
        st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
            <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Famílias Afetadas</h3>
            <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Distribuição do número de famílias afetadas por conflitos agrários por município.</p>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_familias(df_confmun), use_container_width=True, height=400, key="familias")
        st.caption("Figura 3.1: Distribuição de famílias afetadas por município.")
        with st.expander("Detalhes e Fonte da Figura 3.1"):
            st.write("""
            **Interpretação:**  
            O gráfico apresenta o número total de famílias afetadas por conflitos agrários em cada município.

            **Observações:**
            - Dados agregados por município
            - Valores apresentados em ordem decrescente
            - Inclui todos os tipos de conflitos registrados

            **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
            """)

    with col_conf:
        st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0.5rem;">
            <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Conflitos Registrados</h3>
            <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Número total de conflitos agrários registrados por município.</p>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_conflitos(df_confmun), use_container_width=True, height=400, key="conflitos")
        st.caption("Figura 3.2: Distribuição de conflitos registrados por município.")
        with st.expander("Detalhes e Fonte da Figura 3.2"):
            st.write("""
            **Interpretação:**  
            O gráfico mostra o número total de conflitos agrários registrados em cada município.

            **Observações:**
            - Contagem total de ocorrências por município
            - Ordenação por quantidade de conflitos
            - Inclui todos os tipos de conflitos documentados

            **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
            """)

    st.markdown("""<div style="background-color: #fff; border-radius: 6px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 2rem 0 0.5rem 0;">
        <h3 style="color: #1E1E1E; margin-top: 0; margin-bottom: 0.5rem;">Ocupações Retomadas</h3>
        <p style="color: #666; font-size: 0.95em; margin-bottom:0;">Análise das áreas de conflito com processos de retomada por município.</p>
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(fig_ocupacoes(df_csv), use_container_width=True, height=300, key="ocupacoes")
    st.caption("Figura 3.3: Distribuição de ocupações retomadas por município.")
    with st.expander("Detalhes e Fonte da Figura 3.3"):
        st.write("""
        **Interpretação:**  
        O gráfico apresenta o número de áreas onde houve processos de retomada de ocupações por município.

        **Observações:**
        - Contabiliza áreas com processos de retomada concluídos
        - Ordenação por quantidade de retomadas
        - Permite visualizar concentração geográfica das ações

        **Fonte:** CPT - Comissão Pastoral da Terra. *Conflitos no Campo Brasil*. Goiânia: CPT Nacional, 2025. Disponível em: https://www.cptnacional.org.br/. Acesso em: maio de 2025.
        """)

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

    figs_j = fig_justica(df_proc)
    key_map = {"Municípios": "mun", "Classes": "class", "Assuntos": "ass", "Órgãos": "org", "Temporal": "temp"}
    barras = ["Municípios", "Classes", "Assuntos", "Órgãos"]

    cols = st.columns(2, gap="large")
    for idx, key in enumerate(barras):
        col = cols[idx % 2]
        chart_key = key_map[key]
        col.markdown(f"""
            <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin-bottom:0.5rem;">
                <h3 style="margin:0 0 .5rem 0;">{key}</h3>
                <p style="margin:0;font-size:.95em;color:#666;">Distribuição por {key.lower()}.</p>
            </div>
        """, unsafe_allow_html=True)
        col.plotly_chart(figs_j[chart_key].update_layout(height=350), use_container_width=True, key=f"jud_{chart_key}")
        col.caption(f"Figura 4.{idx+1}: Distribuição por {key.lower()}.")
        with col.expander(f"ℹ️ Detalhes e Fonte da Figura 4.{idx+1}", expanded=False):
            st.write(f"""
            **Interpretação:**  
            Distribuição dos processos por {key.lower()}.

            **Fonte:** TJPA – Tribunal de Justiça do Estado do Pará, 2025.
            """)

    st.markdown("""
        <div style="background:#fff;border-radius:6px;padding:1.5rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);margin:1rem 0 .5rem 0;">
            <h3 style="margin:0 0 .5rem 0;">Evolução Mensal de Processos</h3>
            <p style="margin:0;font-size:.95em;color:#666;">Variação mensal ao longo do período.</p>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(figs_j[key_map["Temporal"]].update_layout(height=400), use_container_width=True, key="jud_temp")
    st.caption("Figura 4.5: Evolução temporal dos processos judiciais.")
    with st.expander("ℹ️ Detalhes e Fonte da Figura 4.5", expanded=False):
        st.write("""
        **Interpretação:**  
        Evolução mensal dos processos.

        **Fonte:** TJPA – Tribunal de Justiça do Estado do Pará, 2025.
        """)

with tabs[3]:
    st.header("Focos de Calor")

    with st.expander("ℹ️ Sobre esta seção", expanded=True):
        st.write("""
        Esta análise apresenta dados sobre focos de calor detectados por satélite, incluindo:
        - Risco de fogo
        - Precipitação acumulada  
        - Distribuição espacial

        Os dados são provenientes do Programa Queimadas do INPE.
        """)
        st.markdown(
            "**Fonte Geral da Seção:** INPE – Instituto Nacional de Pesquisas Espaciais. Programa Queimadas: Monitoramento dos Focos Ativos por Estados. São José dos Campos: INPE, 2025.",
            unsafe_allow_html=True
        )

    files = [
        r"focos_municipios_filtrados_part1.csv",
        r"focos_municipios_filtrados_part2.csv",
        r"focos_municipios_filtrados_part3.csv",
        r"focos_municipios_filtrados_part4.csv",
        r"focos_municipios_filtrados_part5.csv",
        r"focos_municipios_filtrados_part6.csv",
        r"focos_municipios_filtrados_2024_parte_1.csv",
        r"focos_municipios_filtrados_2024_parte_2.csv",
        r"focos_municipios_filtrados_2024_parte_3.csv",
        r"focos_municipios_filtrados_2024_parte_4.csv"
    ]

    @st.cache_data(show_spinner=False)
    def load_inpe_duckdb(filepaths):
        conn = duckdb.connect(database=':memory:')
        queries = []
        for path in filepaths:
            queries.append(f"""
                SELECT 
                    try_cast(DataHora as TIMESTAMP) AS DataHora,
                    try_cast(RiscoFogo AS DOUBLE) AS RiscoFogo,
                    try_cast(Precipitacao AS DOUBLE) AS Precipitacao,
                    try_cast(mun_corrigido AS VARCHAR) AS mun_corrigido,
                    try_cast(DiaSemChuva AS INT) AS DiaSemChuva,
                    try_cast(Latitude AS DOUBLE) AS Latitude,
                    try_cast(Longitude AS DOUBLE) AS Longitude
                FROM read_csv_auto('{path}')
                WHERE 
                    RiscoFogo BETWEEN 0 AND 1 AND 
                    Precipitacao >= 0 AND 
                    DiaSemChuva >= 0 AND
                    Latitude BETWEEN -15 AND 5 AND
                    Longitude BETWEEN -60 AND -45
            """)
        full_query = " UNION ALL ".join(queries)
        df = conn.execute(full_query).fetchdf()
        df = df.dropna(subset=['DataHora'])
        return df

    df_inpe = load_inpe_duckdb(files)

    if not df_inpe.empty:
        anos = sorted(df_inpe['DataHora'].dt.year.dropna().unique())
        ano_selecionado = st.selectbox('Selecione o ano para análise:', anos, index=len(anos) - 1)
        figs = graficos_inpe(df_inpe, ano_selecionado)

        st.subheader("Evolução Temporal do Risco de Fogo")
        st.plotly_chart(figs['temporal'], use_container_width=True)
        st.caption(f"Figura 5.1: Evolução mensal do risco médio de fogo para o ano de {ano_selecionado}.")

        with st.expander("Detalhes e Fonte da Figura 5.1"):
            st.write(f"""
            **Interpretação:**  
            O gráfico mostra como o risco médio de fogo varia mês a mês em {ano_selecionado}, numa escala de 0 (mínimo) a 1 (máximo).

            - **Pontos:** valores médios mensais.
            - **Linha:** tendência ao longo do ano.

            **Observação para {ano_selecionado}:**  
            { {
                2020: "Pico em agosto (0.94).",
                2021: "Pico em julho (0.87).",
                2022: "Pico em julho (0.83).",
                2023: "Pico em agosto (0.69).",
                2024: "Pico em setembro (0.96)."
            }.get(ano_selecionado, "") }

            **Fonte:** INPE. *Programa Queimadas: Dados de Focos de Calor*. São José dos Campos: INPE, 2025.
            """)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Top Municípios por Risco Médio de Fogo")
            st.plotly_chart(figs['top_risco'], use_container_width=True)
            st.caption(f"Figura 5.2: Municípios com maior risco médio de fogo em {ano_selecionado}.")
            with st.expander("Detalhes e Fonte da Figura 5.2"):
                st.write(f"""
                **Interpretação:**  
                Ranking dos municípios com maior risco médio de fogo em {ano_selecionado}.
                {"(Dados disponíveis até " + pd.Timestamp.now().strftime('%B/%Y') + ")" if ano_selecionado == 2024 else ""}

                **Fonte:** INPE. *Programa Queimadas*. INPE, 2025.
                """)

            st.subheader("Top Municípios por Precipitação Acumulada")
            st.plotly_chart(figs['top_precip'], use_container_width=True)
            st.caption(f"Figura 5.3: Municípios com maior precipitação acumulada em {ano_selecionado}.")
            with st.expander("Detalhes e Fonte da Figura 5.3"):
                st.write(f"""
                **Interpretação:**  
                Ranking dos municípios com maior volume de chuva (mm) em {ano_selecionado}.
                {"(Dados disponíveis até " + pd.Timestamp.now().strftime('%B/%Y') + ")" if ano_selecionado == 2024 else ""}

                **Fonte:** INPE. *Programa Queimadas*. INPE, 2025.
                """)

        with col2:
            st.subheader("Mapa de Distribuição dos Focos de Calor")
            st.plotly_chart(figs['mapa'], use_container_width=True)
            st.caption(f"Figura 5.4: Distribuição espacial dos focos de calor em {ano_selecionado}.")
            with st.expander("Detalhes e Fonte da Figura 5.4"):
                st.write(f"""
                **Interpretação:**  
                Cada ponto representa um foco de calor detectado por satélite em {ano_selecionado}.  
                Alta densidade indica maior atividade de queimadas.
                {"(Dados disponíveis até " + pd.Timestamp.now().strftime('%B/%Y') + ")" if ano_selecionado == 2024 else ""}

                **Fonte:** INPE. *Programa Queimadas*. INPE, 2025.
                """)

    else:
        st.warning("Nenhum dado disponível após filtros.")

with tabs[4]:
    st.header("")
