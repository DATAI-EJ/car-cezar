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
  /* Tipografia Times New Roman + tamanhos */
  body, h1, h2, h3, h4, h5, p {
    font-family: 'Times New Roman', serif !important;
  }
  h1 { font-size:32px !important; font-weight:700 !important; }
  h2 { font-size:28px !important; font-weight:600 !important; }

  /* Sidebar estilizada */
  [data-testid="stSidebar"] {
    width: 250px;
    background-color: #FDF5E6;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    padding: 1rem;
  }
  .sidebar .stContainer {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }

  /* Containers e seções com espaçamento */
  .stContainer {
    padding: 1rem 2rem !important;
  }
  .spacer {
    margin-top:20px;
    margin-bottom:20px;
  }

  /* Aba ativa em azul escuro, texto branco */
  .stTabs [role="tablist"] button[aria-selected="true"] {
      background-color: #2F5496 !important;
      color: white       !important;
  }
  /* Abas inativas em cinza claro, texto azul */
  .stTabs [role="tablist"] button[aria-selected="false"] {
      background-color: #F0F0F5 !important;
      color: #2F5496     !important;
  }
  /* Remover borda padrão e arredondar cantos das abas */
  .stTabs [role="tablist"] button {
      border-radius: 8px 8px 0 0 !important;
      border: none             !important;
      padding: 0.5rem 1rem      !important;
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

def fig_justica(df_proc: pd.DataFrame) -> dict[str, go.Figure]:
    figs = {}
    palette = px.defaults.color_discrete_sequence
    bottom_margin = 150

    mapa_classes = {
        "Procedimento Comum Cível": "Proc. Comum",
        "Cumprimento de sentença": "Cumpr. Sentença",
        "Ação Civil Pública": "Ação Civil Púb.",
        "Mandado de Segurança Cível": "MS Cível",
        "Execução Fiscal": "Exec. Fiscal",
        "Ação Ordinária": "Ação Ord.",
        "Ação de Obrigação de Fazer / Não Fazer": "Obrig. Fazer/NF",
        "Outros Procedimentos de Jurisdição Contenciosa": "Outros Contenc.",
        "Petição": "Petição",
        "Ação de Improbidade Administrativa": "Improbidade Adm."
    }

    mapa_assuntos = {
        "Direito Ambiental": "Ambiental",
        "Dano Ambiental": "Dano Amb.",
        "Direitos / Interesses Difusos / Coletivos": "Dir. Difusos",
        "Direito Administrativo e Outras Matérias de Direito Público": "Dir. Admin. / Público",
        "Direito do Consumidor": "Consumidor",
        "Direito Processual Civil e do Trabalho": "Proc. Civil / Trab.",
        "Assuntos Minerários": "Minerários",
        "Indenização por Dano Moral": "Dano Moral"
    }

    mapa_orgaos = {
        "1ª VARA CÍVEL DA COMARCA DE ALTAMIRA": "1ª V. ALTAMIRA",
        "VARA CÍVEL E EMPRESARIAL DE REDENÇÃO": "V. REDENÇÃO",
        "COMARCA DE SÃO FÉLIX DO XINGU": "S.F. XINGU",
        "VARA ÚNICA DE ANAPU": "V. ANAPU",
        "VARA ÚNICA DE ULIANÓPOLIS": "V. ULIANÓPOLIS",
        "VARA ÚNICA DE PACAJÁ": "V. PACAJÁ",
        "VARA ÚNICA DE PLACAS": "V. PLACAS",
        "VARA ÚNICA DE MEDICILÂNDIA": "V. MEDICILÂNDIA",
        "VARA ÚNICA DE CUMARU DO NORTE": "V. CUMARU N."
    }

    top = (
        df_proc['municipio']
            .value_counts().head(10)
            .rename_axis('Municipio')
            .reset_index(name='Quantidade')
    )
    top['label'] = top['Municipio'].apply(lambda x: wrap_label(x, 20))
    fig_mun = px.bar(
        top, x='label', y='Quantidade',
        labels={'label': 'Município', 'Quantidade': 'Quantidade'},
        color_discrete_sequence=palette
    )
    fig_mun.update_traces(texttemplate='%{y}', textposition='outside', cliponaxis=False)
    fig_mun.update_layout(
        margin=dict(l=80, r=60, t=50, b=bottom_margin),
        height=400,
        xaxis=dict(tickangle=45, automargin=True, tickmode='array', tickvals=top['label']),
        yaxis=dict(range=[0, top['Quantidade'].max() * 1.1]),
        xaxis_title=None,
        yaxis_title=None
    )
    figs['mun'] = _apply_layout(fig_mun, title="Top 10 Municípios com Mais Processos", title_size=16)

    df_proc['ano_mes'] = pd.to_datetime(df_proc['data_ajuizamento'], errors='coerce') \
                             .dt.to_period('M').dt.to_timestamp()
    mensal = df_proc.groupby('ano_mes').size().reset_index(name='Quantidade')
    fig_temp = px.line(
        mensal, x='ano_mes', y='Quantidade', markers=True,
        labels={'ano_mes': 'Mês/Ano', 'Quantidade': 'Quantidade'}
    )
    fig_temp.update_layout(
        margin=dict(l=80, r=60, t=50, b=bottom_margin),
        height=400,
        xaxis=dict(tickangle=45, automargin=True)
    )
    figs['temp'] = _apply_layout(fig_temp, title="Evolução Mensal de Processos", title_size=16)

    mappings = [
        ('class', 'classe', 'Top 10 Classes Processuais', mapa_classes),
        ('ass', 'assuntos', 'Top 10 Assuntos', mapa_assuntos),
        ('org', 'orgao_julgador', 'Top 10 Órgãos Julgadores', mapa_orgaos)
    ]
    for key, col, title, mapa in mappings:
        df = (
            df_proc[col]
                .replace(mapa)
                .value_counts().head(10)
                .rename_axis(col)
                .reset_index(name='Quantidade')
        )
        df['label'] = df[col].apply(lambda x: wrap_label(x, 20))

        fig = px.bar(
            df, x='label', y='Quantidade',
            labels={'label': col.upper(), 'Quantidade': 'Quantidade'},
            color_discrete_sequence=palette
        )
        fig.update_traces(texttemplate='%{y}', textposition='outside', cliponaxis=False)
        fig.update_layout(
            margin=dict(l=100, r=60, t=50, b=bottom_margin),
            height=400,
            xaxis=dict(tickangle=45, automargin=True),
            yaxis=dict(range=[0, df['Quantidade'].max() * 1.1])
        )
        figs[key] = _apply_layout(fig, title=title, title_size=16)

    return figs

@st.cache_data
def carregar_dados_fogo(
    caminho_csv: str,
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
    df['hour'] = df['DataHora'].dt.hour
    df['month'] = df['DataHora'].dt.to_period('M').dt.to_timestamp()
    return df

def criar_figuras_fogo(df: pd.DataFrame) -> dict[str, go.Figure]:
    figs = {}
    df = df.dropna(subset=['RiscoFogo','Precipitacao','date','Municipio']).assign(date=lambda d: pd.to_datetime(d['date'], errors='coerce'))
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

    max_date = ts_df['date'].max()

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['count'], mode='markers', name='Diário (pontos)', marker=dict(size=3, color=cores[0], opacity=0.4), visible='legendonly', hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Focos: %{y}<extra></extra>'))
    fig_ts.add_trace(go.Scatter(x=ts_df['date'], y=ts_df['m7'], mode='lines', name='Diário (média 7d)', line=dict(color=cores[1], width=3), visible=True, hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Média7d: %{y:.1f}<extra></extra>'))
    fig_ts.add_trace(go.Scatter(x=ts_month['date'], y=ts_month['count'], mode='markers+lines', name='Mensal (totais)', marker=dict(size=6, color=cores[2], opacity=0.6), line=dict(width=1, color=cores[2]), visible=False, hovertemplate='<b>%{x|%Y-%m}</b><br>Focos: %{y}<extra></extra>'))
    fig_ts.add_trace(go.Scatter(x=ts_month['date'], y=ts_month['m3'], mode='lines', name='Mensal (média 3m)', line=dict(color=cores[3], width=3, dash='dash'), visible=False, hovertemplate='<b>%{x|%Y-%m}</b><br>Média3m: %{y:.1f}<extra></extra>'))
    fig_ts.add_trace(go.Scatter(x=ts_ann['date'], y=ts_ann['count'], mode='markers+lines', name='Anual (totais)', marker=dict(size=8, color=cores[4], opacity=0.6), line=dict(width=1, color=cores[4]), visible=False, hovertemplate='<b>%{x|%Y}</b><br>Focos: %{y}<extra></extra>'))
    fig_ts.add_trace(go.Scatter(x=ts_ann['date'], y=ts_ann['m2'], mode='lines', name='Anual (média 2a)', line=dict(color=cores[5] if len(cores)>5 else cores[0], width=3, dash='dot'), visible=False, hovertemplate='<b>%{x|%Y}</b><br>Média2a: %{y:.1f}<extra></extra>'))
    fig_ts.update_layout(
        updatemenus=[dict(type='buttons', direction='right', x=0.0, y=1.15, pad=dict(r=10, t=10), bgcolor='white', bordercolor='lightgray', borderwidth=1, font=dict(size=12, color='black'), active=0, buttons=[
            dict(label='Diário', method='update', args=[{'visible':[True,True,False,False,False,False]}]),
            dict(label='Mensal', method='update', args=[{'visible':[False,False,True,True,False,False]}]),
            dict(label='Anual', method='update', args=[{'visible':[False,False,False,False,True,True]}])
        ])],
        xaxis=dict(type='date', range=[max_date - pd.Timedelta(days=365*5), max_date], tickformat='%Y', nticks=8, showgrid=False, title='Data'),
        yaxis=dict(title='Número de Focos'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        width=700, height=400, font=dict(size=12),
        title=dict(text='Focos de Calor', x=0.5, xanchor='center', font=dict(size=18))
    )
    figs['ts'] = _apply_layout(fig_ts, title='Focos de Calor', title_size=18)

    top = df['Municipio'].value_counts().head(10).rename_axis('Município').reset_index(name='Focos')
    top['Mun_wrap'] = top['Município'].apply(lambda x: wrap_label(x, 25))
    top['Categoria'] = top['Focos'].rank(method='first', ascending=False).apply(lambda r: 'Top 3' if r <= 3 else 'Outros')
    fig_top = px.bar(top.sort_values('Focos', ascending=True), x='Focos', y='Mun_wrap', orientation='h', color='Categoria', text='Focos', color_discrete_map={'Top 3': cores[2], 'Outros': cores[3]}, labels={'Focos':'Nº de Focos','Mun_wrap':'Município'}, template=None)
    fig_top.update_traces(texttemplate='%{text:.0f}', textposition='outside', hovertemplate='<b>%{y}</b><br>Focos: %{x}<extra></extra>')
    fig_top.update_layout(yaxis=dict(title=''), xaxis=dict(title='Número de Focos'), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=150, r=20, t=60, b=40), height=400)
    figs['top_municipios'] = _apply_layout(fig_top, title='Top 10 Municípios por Focos de Calor', title_size=18)

    df_r = df.query('RiscoFogo >= 0 and RiscoFogo <= 100')
    fig_violin = px.violin(df_r, y='RiscoFogo', box=True, points='all', orientation='h', labels={'RiscoFogo':'Risco de Fogo (%)'}, template=None)
    fig_violin.update_traces(meanline_visible=True, hovertemplate='<b>Risco:</b> %{y:.1f}%<extra></extra>')
    fig_violin.update_layout(xaxis=dict(title='Risco de Fogo (%)'), yaxis=dict(title=''), margin=dict(l=80, r=20, t=60, b=40))
    figs['hist_risco'] = _apply_layout(fig_violin, title='Distribuição de Risco de Fogo (%)', title_size=18)

    df_map = df.dropna(subset=['Latitude','Longitude'])
    p99 = df_map['Precipitacao'].quantile(0.99)
    r99 = df_map['RiscoFogo'].quantile(0.99)
    df_s = df_map[(df_map['Precipitacao']>=0)&(df_map['Precipitacao']<=p99)&(df_map['RiscoFogo']>=0)&(df_map['RiscoFogo']<=r99)]
    fig_map = px.scatter_mapbox(df_s, lat='Latitude', lon='Longitude', hover_name='Municipio', hover_data={'date':True,'RiscoFogo':True}, color_continuous_scale=px.colors.cyclical.IceFire, size_max=6, zoom=5, height=400, template=None)
    centro = {'lat': df_s['Latitude'].mean(), 'lon': df_s['Longitude'].mean()}
    fig_map.update_layout(mapbox=dict(style='open-street-map', center=centro, zoom=5), margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig_map.update_traces(marker=dict(color=cores[3]), marker_showscale=False)
    figs['scatter_prec_risco'] = _apply_layout(fig_map, title='Mapa de Focos de Calor', title_size=18)

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

tabs = st.tabs(["Sobreposições", "Queimadas", "Famílias", "Justiça"])

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

with tabs[1]:
    st.header("Focos de Calor")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Temporal Diário")
        st.plotly_chart(
            figs_fogo['ts'],
            use_container_width=True,
            key="fogo_ts"
        )

    with col2:
        st.subheader("Top 10 Municípios")
        st.plotly_chart(
            figs_fogo['top_municipios'],
            use_container_width=True,
            key="fogo_top10"
        )

with tabs[2]:
    st.header("Impacto Social")
    with st.container():
        col_fam, col_conf = st.columns(2, gap="large")
        with col_fam:
            st.subheader("Famílias Afetadas")
            st.plotly_chart(
                fig_familias(df_confmun),
                use_container_width=True,
                height=400,
                key="familias"
            )
        with col_conf:
            st.subheader("Conflitos Registrados")
            st.plotly_chart(
                fig_conflitos(df_confmun),
                use_container_width=True,
                height=400,
                key="conflitos"
            )

with tabs[3]:
    st.header("Processos Judiciais")
    figs_j = fig_justica(df_proc)
    key_map = {"Municípios":"mun","Temporal":"temp","Classes":"class","Assuntos":"ass","Órgãos":"org"}
    linhas = [("Municípios","Temporal","Classes"),("Assuntos","Órgãos","")]

    for i, linha in enumerate(linhas):
        with st.container():
            cols = st.columns(3, gap="large")
            for j, key in enumerate(linha):
                if not key: continue
                chart_key = key_map[key]
                cols[j].subheader(key)
                cols[j].plotly_chart(
                    figs_j[chart_key],
                    use_container_width=True,
                    height=300,
                    key=f"jud_{chart_key}_{i}"
                )
