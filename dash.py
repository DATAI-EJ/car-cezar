import streamlit as st
st.set_page_config(page_title="Dashboard", layout="wide")
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

@st.cache_data
def carregar_shapefile(caminho, calcular_percentuais=True):
    gdf = gpd.read_file(caminho)
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)
    gdf = gdf[gdf["geometry"].notnull() & gdf["geometry"].is_valid]
    gdf_proj = gdf.to_crs("EPSG:31983")
    gdf_proj["area_calc_km2"] = gdf_proj.geometry.area / 1e6
    if "area_km2" in gdf.columns:
        gdf["area_km2"] = gdf["area_km2"].replace(0, None)
        gdf["area_km2"] = gdf["area_km2"].fillna(gdf_proj["area_calc_km2"])
    else:
        gdf["area_km2"] = gdf_proj["area_calc_km2"]
    if calcular_percentuais:
        if "alerta_km2" in gdf.columns:
            gdf["perc_alerta"] = (gdf["alerta_km2"] / gdf["area_km2"]) * 100
        else:
            gdf["perc_alerta"] = 0
        if "sigef_km2" in gdf.columns:
            gdf["perc_sigef"] = (gdf["sigef_km2"] / gdf["area_km2"]) * 100
        else:
            gdf["perc_sigef"] = 0
    else:
        gdf["perc_alerta"] = 0
        gdf["perc_sigef"] = 0
    gdf["id"] = gdf.index.astype(str)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

gdf_cnuc = carregar_shapefile(r"cnuc.shp")
gdf_sigef = carregar_shapefile(r"sigef.shp", calcular_percentuais=False)
gdf_cnuc["base"] = "cnuc"
gdf_sigef["base"] = "sigef"
gdf_sigef = gdf_sigef.rename(columns={"id": "id_sigef"})
limites = gdf_cnuc.total_bounds
centro = {"lat": (limites[1] + limites[3]) / 2, "lon": (limites[0] + limites[2]) / 2}

@st.cache_data
def load_csv(caminho):
    df = pd.read_csv(caminho)
    df = df.rename(columns={"Unnamed: 0": "Município"})
    colunas_ocorrencias = ["Áreas de conflitos", "Assassinatos", "Conflitos por Terra", "Ocupações Retomadas", "Tentativas de Assassinatos", "Trabalho Escravo"]
    df["total_ocorrencias"] = df[colunas_ocorrencias].sum(axis=1)
    return df

df_csv = load_csv(r"CPT-PA-count.csv")

def criar_figura(ids_selecionados, invadindo_opcao):
    fig = px.choropleth_mapbox(
        gdf_cnuc,
        geojson=gdf_cnuc.__geo_interface__,
        locations="id",
        color_discrete_sequence=["#DDDDDD"],
        hover_data=["nome_uc", "municipio", "perc_alerta", "perc_sigef", "alerta_km2", "sigef_km2", "area_km2"],
        mapbox_style="open-street-map",
        center=centro,
        zoom=4,
        opacity=0.7,
        title="Porcentagem de Área Sobreposta por Alertas e SIGEF",
        template="simple_white"
    )
    if ids_selecionados:
        gdf_sel = gdf_cnuc[gdf_cnuc["id"].isin(ids_selecionados)]
        fig_sel = px.choropleth_mapbox(
            gdf_sel,
            geojson=gdf_cnuc.__geo_interface__,
            locations="id",
            color_discrete_sequence=["#0074D9"],
            hover_data=["nome_uc", "municipio", "perc_alerta", "perc_sigef", "alerta_km2", "sigef_km2", "area_km2"],
            mapbox_style="open-street-map",
            center=centro,
            zoom=4,
            opacity=0.8,
            template="simple_white"
        )
        for trace in fig_sel.data:
            fig.add_trace(trace)
    if invadindo_opcao is not None:
        if invadindo_opcao.lower() == "todos":
            gdf_sigef_filtrado = gdf_sigef
        else:
            gdf_sigef_filtrado = gdf_sigef[gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()]
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
    cidades = df_csv["Município"].unique()
    cores_paleta = px.colors.qualitative.Pastel
    color_map = {cidade: cores_paleta[i % len(cores_paleta)] for i, cidade in enumerate(cidades)}
    for cidade in cidades:
        df_cidade = df_csv[df_csv["Município"] == cidade]
        base_size = list(df_cidade["total_ocorrencias"] * 3)
        outline_size = [s + 4 for s in base_size]
        trace_cpt_outline = go.Scattermapbox(
            lat=df_cidade["Latitude"],
            lon=df_cidade["Longitude"],
            mode="markers",
            marker=dict(size=outline_size, color="black", sizemode="area"),
            hoverinfo="none",
            showlegend=False
        )
        trace_cpt = go.Scattermapbox(
            lat=df_cidade["Latitude"],
            lon=df_cidade["Longitude"],
            mode="markers",
            marker=dict(size=base_size, color=color_map[cidade], sizemode="area"),
            text=df_cidade.apply(lambda linha: (
                f"Município: {linha['Município']}<br>"
                f"Áreas de conflitos: {linha['Áreas de conflitos']}<br>"
                f"Assassinatos: {linha['Assassinatos']}<br>"
                f"Conflitos por Terra: {linha['Conflitos por Terra']}<br>"
                f"Ocupações Retomadas: {linha['Ocupações Retomadas']}<br>"
                f"Tentativas de Assassinatos: {linha['Tentativas de Assassinatos']}<br>"
                f"Trabalho Escravo: {linha['Trabalho Escravo']}"
            ), axis=1),
            hoverinfo="text",
            name=f"Ocorrências - {cidade}",
            showlegend=True
        )
        fig.add_trace(trace_cpt_outline)
        fig.add_trace(trace_cpt)
    fig.update_layout(legend=dict(title="Legenda", x=0, y=1), height=700, margin={"r": 10, "t": 50, "l": 10, "b": 10}, title_font=dict(size=22))
    return fig

def criar_cards(ids_selecionados):
    filtro = gdf_cnuc[gdf_cnuc["id"].isin(ids_selecionados)] if ids_selecionados else gdf_cnuc
    total_area = filtro["area_km2"].sum()
    total_alerta = filtro["alerta_km2"].sum()
    total_sigef = filtro["sigef_km2"].sum()
    perc_total_alerta = (total_alerta / total_area) * 100 if total_area != 0 else 0
    perc_total_sigef = (total_sigef / total_area) * 100 if total_area != 0 else 0
    total_unidades = filtro.shape[0]
    total_c_alertas = filtro["c_alertas"].sum()
    total_c_sigef = filtro["c_sigef"].sum()
    return perc_total_alerta, perc_total_sigef, int(total_unidades), int(total_c_alertas), int(total_c_sigef)

def render_cards(perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef):
    card_style = ("background-color: #0074D9; padding: 5px; border-radius: 5px; text-align: center; width: 120px; height: 120px; margin: 5px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-sizing: border-box; overflow: hidden; font-size: 14px;")
    html = f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 20px;">
        <div style="{card_style}">
            <h4 style="margin:0; font-size:16px; color: white;">Percentual de Alerta</h4>
            <p style="margin:0; font-size:14px; color: white;">{perc_alerta:.2f}%</p>
        </div>
        <div style="{card_style}">
            <h4 style="margin:0; font-size:16px; color: white;">Percentual SIGEF</h4>
            <p style="margin:0; font-size:14px; color: white;">{perc_sigef:.2f}%</p>
        </div>
        <div style="{card_style}">
            <h4 style="margin:0; font-size:16px; color: white;">Total de Unidades</h4>
            <p style="margin:0; font-size:14px; color: white;">{total_unidades}</p>
        </div>
        <div style="{card_style}">
            <h4 style="margin:0; font-size:16px; color: white;">Contagem Alerta</h4>
            <p style="margin:0; font-size:14px; color: white;">{contagem_alerta}</p>
        </div>
        <div style="{card_style}">
            <h4 style="margin:0; font-size:16px; color: white;">Contagem SIGEF</h4>
            <p style="margin:0; font-size:14px; color: white;">{contagem_sigef}</p>
        </div>
    </div>
    """
    return html

bar_fig = px.bar(
    gdf_cnuc,
    x='nome_uc',
    y=['alerta_km2', 'sigef_km2', 'area_km2'],
    labels={'value': "Área (Km²)", "nome_uc": "Nome UC"},
    color_discrete_map={"alerta_km2": 'rgb(251,180,174)', "sigef_km2": 'rgb(179,205,227)', "area_km2": 'rgb(204,235,197)'},
    template="simple_white"
)
bar_fig.update_layout(legend_title_text='Métricas')

contagens_fig = px.bar(
    gdf_cnuc,
    x='nome_uc',
    y=['c_alertas', 'c_sigef'],
    labels={'value': "Contagens", "nome_uc": "Nome UC"},
    color_discrete_map={"c_alertas": 'rgb(251,180,174)', "c_sigef": 'rgb(179,205,227)'},
    template="simple_white"
)
contagens_fig.update_layout(legend_title_text='Contagens de Alertas e SIGEF')

pie_fig = px.pie(
    df_csv,
    values='Áreas de conflitos',
    names='Município',
    title='Áreas de conflitos',
    color_discrete_sequence=px.colors.qualitative.Pastel1,
    template="simple_white",
    hole=0.4
)
pie_fig.update_traces(textposition='inside', textinfo='percent+label')
pie_fig.update_layout(font_size=14)

st.title("Cadastro Nacional de Unidades de Conservação (CNUC)")
opcoes_invadindo = ["Selecione", "Todos"] + sorted(gdf_sigef["invadindo"].str.strip().unique().tolist())
invadindo_opcao = st.sidebar.selectbox("Selecione a área (invadindo)", opcoes_invadindo)
if invadindo_opcao == "Selecione":
    invadindo_opcao = None
if invadindo_opcao is None or invadindo_opcao.lower() == "todos":
    ids_selecionados = []
else:
    gdf_sigef_filtrado = gdf_sigef[gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()]
    gdf_cnuc_filtrado = gpd.sjoin(gdf_cnuc, gdf_sigef_filtrado, how="inner", predicate="intersects")
    ids_selecionados = gdf_cnuc_filtrado["id"].unique().tolist()
fig = criar_figura(ids_selecionados, invadindo_opcao)
perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef = criar_cards(ids_selecionados)
col1, col2 = st.columns([6, 4])
with col1:
    st.plotly_chart(fig, use_container_width=True)
    cards_html = render_cards(perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef)
    st.markdown(cards_html, unsafe_allow_html=True)
with col2:
    st.plotly_chart(bar_fig, use_container_width=True)
    st.plotly_chart(contagens_fig, use_container_width=True)
    st.plotly_chart(pie_fig, use_container_width=True)
