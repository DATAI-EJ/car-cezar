import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unicodedata


df_cpt = pd.DataFrame()

st.set_page_config(page_title="Dashboard", layout="wide")

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

@st.cache_data
def carregar_dados_conflitos_municipio(arquivo_excel):
    df = pd.read_excel(arquivo_excel, sheet_name='Áreas em Conflito', header=0).dropna(how='all')
    df['mun'] = df['mun'].apply(lambda x: [unicodedata.normalize('NFD', str(m).lower()).encode('ascii', 'ignore').decode('ascii').strip().title() for m in str(x).split(',')])
    df_exploded = df.explode('mun')
    df_exploded['Famílias'] = pd.to_numeric(df_exploded['Famílias'], errors='coerce').fillna(0)
    df_exploded['num_municipios'] = df_exploded.groupby('Nome do Conflito')['mun'].transform('nunique')
    df_exploded['Famílias_por_municipio'] = df_exploded['Famílias'] / df_exploded['num_municipios']
    df_conflitos = df_exploded.groupby('mun').agg({'Famílias_por_municipio': 'sum', 'Nome do Conflito': 'count'}).reset_index()
    df_conflitos.columns = ['Município', 'Total_Famílias', 'Número_Conflitos']
    return df_conflitos

df_conflitos_municipio = carregar_dados_conflitos_municipio(r"áreas-selecionadas\CPTF-PA.xlsx")

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
        opacity=0.7
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
    
    cidades = df_csv["Município"].unique()
    cores_paleta = px.colors.qualitative.Pastel
    color_map = {cidade: cores_paleta[i % len(cores_paleta)] for i, cidade in enumerate(cidades)}
    
    for cidade in cidades:
        df_cidade = df_csv[df_csv["Município"] == cidade]
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
            height=800  
        )
    return fig

def criar_graficos_cpt(df):
    figs = []
    
    if 'Ameaçados' in df['Tipo'].values:
        df_ameaca = df[df['Tipo'] == 'Ameaçados']
        fig = px.bar(
            df_ameaca,
            x='Tipo',
            y=['Famílias', 'Área_ha', 'Casas_Destruídas'],
            title='Distribuição de Ameaças',
            labels={'value': 'Quantidade', 'variable': 'Categoria'},
            barmode='group'
        )
        figs.append(fig)
    
    if 'Áreas em Conflito' in df['Tipo'].values:
        df_conflito = df[df['Tipo'] == 'Áreas em Conflito']
        fig = px.scatter(
            df_conflito,
            x='Municípios_Únicos',
            y='Total_Conflitos',
            size='Famílias_Risco',
            title='Relação Conflitos vs Municípios',
            color='Total_Conflitos'
        )
        figs.append(fig)
    
    return figs

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
        text-align: center;">
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

common_layout = {
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Arial", "size": 12},
    "margin": {"t": 40, "b": 20},
    "hoverlabel": {"bgcolor": "white", "font_size": 12}
}

gdf_cnuc_ha = gdf_cnuc.copy()
gdf_cnuc_ha['alerta_ha'] = gdf_cnuc_ha['alerta_km2'] * 100
gdf_cnuc_ha['sigef_ha'] = gdf_cnuc_ha['sigef_km2'] * 100
gdf_cnuc_ha['area_ha']   = gdf_cnuc_ha['area_km2'] * 100

bar_fig = px.bar(
    gdf_cnuc_ha,
    x='nome_uc',
    y=['alerta_ha', 'sigef_ha', 'area_ha'],
    labels={'value': "Área (ha)", "nome_uc": "Nome UC"},
    color_discrete_map={
        "alerta_ha": 'rgb(251,180,174)', 
        "sigef_ha": 'rgb(179,205,227)', 
        "area_ha": 'rgb(204,235,197)'
    },
    barmode='stack',
    text_auto=True
)

for trace in bar_fig.data:
    trace.texttemplate = '%{y:.0f} ha'
    trace.textposition = 'inside'
    trace.hovertemplate = (
        "<b>%{x}</b><br>" +
        trace.name + ": %{y:,.0f}" +
        "<extra></extra>"
    )

bar_fig.update_yaxes(tickformat=",")
bar_fig.update_layout(
    **common_layout,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

contagens_fig = px.bar(
    gdf_cnuc,
    x='nome_uc',
    y=['c_alertas', 'c_sigef'],
    labels={'value': "Contagens", "nome_uc": "Nome UC"},
    color_discrete_map={
        "c_alertas": 'rgb(251,180,174)', 
        "c_sigef": 'rgb(179,205,227)'
    },
    barmode='stack'
)

for trace in contagens_fig.data:
    trace.texttemplate = '%{y:.0f}'
    trace.textposition = 'inside'
    trace.hovertemplate = (
        "<b>%{x}</b><br>" +
        trace.name + ": %{y:,.0f}" +
        "<extra></extra>"
    )

contagens_fig.update_yaxes(tickformat=",")
contagens_fig.update_layout(
    **common_layout,    
    hovermode="x unified",
    showlegend=False
)

st.header("Análise de Conflitos em Áreas Protegidas e Territórios Tradicionais")
st.caption("Monitoramento integrado de sobreposições em Unidades de Conservação, Terras Indígenas e Territórios Quilombolas")
st.divider()

with st.sidebar:
    st.header("Filtros")
    opcoes_invadindo = ["Selecione", "Todos"] + sorted(gdf_sigef["invadindo"].str.strip().unique().tolist())
    invadindo_opcao = st.selectbox("Área de sobreposição:", opcoes_invadindo, help="Selecione o tipo de área sobreposta para análise")
    st.info("ℹ️ Use os filtros para explorar diferentes cenários de sobreposição territorial.")

if invadindo_opcao == "Selecione":
    invadindo_opcao = None

if invadindo_opcao is None or invadindo_opcao.lower() == "todos":
    ids_selecionados = []
else:
    gdf_sigef_filtrado = gdf_sigef[gdf_sigef["invadindo"].str.strip().str.lower() == invadindo_opcao.strip().lower()]
    gdf_cnuc_filtrado = gpd.sjoin(gdf_cnuc, gdf_sigef_filtrado, how="inner", predicate="intersects")
    ids_selecionados = gdf_cnuc_filtrado["id"].unique().tolist()


df = pd.read_csv(r'processos_tjpa_completo_atualizada_pronto.csv',sep = ';', encoding = 'windows-1252')

municipio_counts = df['municipio'].value_counts().head(10).sort_values(ascending=False)
df_municipios = municipio_counts.reset_index()
df_municipios.columns = ['Município', 'Quantidade de Processos']
fig_municipio = px.bar(
    df_municipios,
    x='Quantidade de Processos',
    y='Município',
    orientation='h',
    title='Top 10 Municípios com mais Processos',
    color='Município',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    template='plotly_white'
)
fig_municipio.update_layout(showlegend=False)

df['data_ajuizamento'] = pd.to_datetime(df['data_ajuizamento'], errors='coerce')
if df['data_ajuizamento'].isnull().sum() > 0:
    print("Atenção: Existem datas inválidas que foram convertidas para NaT.")
df['ano_mes'] = df['data_ajuizamento'].dt.to_period('M').dt.to_timestamp()
novembro_2013 = pd.Timestamp('2013-11-01')
if novembro_2013 in df['ano_mes'].unique():
    count_nov_2013 = df[df['ano_mes'] == novembro_2013].shape[0]
    print(f"Número de processos em novembro de 2013: {count_nov_2013}")
processos_por_mes = df.groupby('ano_mes').size().sort_index()
df_mensal = processos_por_mes.reset_index()
df_mensal.columns = ['Ano-Mês', 'Quantidade de Processos']
fig_temporal = px.line(
    df_mensal,
    x='Ano-Mês',
    y='Quantidade de Processos',
    title='Distribuição dos Processos ao Longo do Tempo',
    markers=True,
    labels={'Ano-Mês': 'Ano-Mês', 'Quantidade de Processos': 'Qtd. de Processos'},
    template='plotly_white',
    color_discrete_sequence=['#FEA3AA']
)

classe_df = df['classe'].value_counts().head(10).sort_values(ascending=False).reset_index()
classe_df.columns = ['Classe', 'Quantidade']
fig_classe = px.bar(
    classe_df,
    x='Quantidade',
    y='Classe',
    orientation='h',
    title='Top 10 Classes de Processos',
    color='Classe',
    color_discrete_sequence=px.colors.qualitative.Pastel1,
    template='plotly_white'
)
fig_classe.update_layout(showlegend=False)

assuntos_df = df['assuntos'].value_counts().head(10).sort_values(ascending=False).reset_index()
assuntos_df.columns = ['Assunto', 'Quantidade']
fig_assuntos = px.bar(
    assuntos_df,
    x='Quantidade',
    y='Assunto',
    orientation='h',
    title='Top 10 Assuntos dos Processos',
    color='Assunto',
    color_discrete_sequence=px.colors.qualitative.Pastel2,
    template='plotly_white'
)
fig_assuntos.update_layout(showlegend=False)

orgao_counts = df['orgao_julgador'].value_counts().head(10).sort_values(ascending=False)
df_orgaos = orgao_counts.reset_index()
df_orgaos.columns = ['Órgão Julgador', 'Quantidade de Processos']
fig_orgaos = px.bar(
    df_orgaos,
    x='Quantidade de Processos',
    y='Órgão Julgador',
    orientation='h',
    title='Top 10 Órgãos Julgadores com Mais Processos',
    color='Órgão Julgador',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    template='plotly_white'
)
fig_orgaos.update_layout(showlegend=False)

fig = criar_figura(ids_selecionados, invadindo_opcao)
perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef = criar_cards(ids_selecionados, invadindo_opcao)

col1, col2 = st.columns([8, 4], gap="large")

with col1:
    st.plotly_chart(fig, use_container_width=True, height=700)
    render_cards(perc_alerta, perc_sigef, total_unidades, contagem_alerta, contagem_sigef)

with col2:
    tab1, tab2, tab3, tab4 = st.tabs(["Sobreposições", "Ocupações Retomadas", "Famílias", "Justiça"])
    
    with tab1:
        st.plotly_chart(bar_fig, use_container_width=True)
        st.plotly_chart(contagens_fig, use_container_width=True)
    
    with tab2:
        lollipop_fig = px.bar(
            df_csv.sort_values('Áreas de conflitos', ascending=False),
            x='Áreas de conflitos',
            y='Município',
            orientation='h',
            color='Município',
            color_discrete_sequence=px.colors.qualitative.Pastel1
        )
        lollipop_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        lollipop_fig.update_layout(**common_layout, showlegend=False)
        st.plotly_chart(lollipop_fig, use_container_width=True)
        
    with tab3:
        df_conflitos_municipio['Município'] = df_conflitos_municipio['Município'].apply(lambda x: unicodedata.normalize('NFD', x).encode('ascii', 'ignore').decode('utf-8').strip().title())
        
        df_conflitos_municipio = df_conflitos_municipio.sort_values('Total_Famílias', ascending=False)
        fig_familias = px.bar(
            df_conflitos_municipio,
            x='Total_Famílias',
            y='Município',
            orientation='h',
            title='Total de Famílias Afetadas por Município',
            labels={'Total_Famílias': 'Total de Famílias Afetadas', 'Município': 'Município'},
            text='Total_Famílias',
            color='Município',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_familias.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            hovermode="y unified",
            showlegend=False
        )
        fig_familias.update_traces(texttemplate='<b>%{text:.0f}</b>', textposition='outside')
        st.plotly_chart(fig_familias, use_container_width=True)
    
        df_conflitos_municipio = df_conflitos_municipio.sort_values('Número_Conflitos', ascending=False)
        fig_conflitos = px.bar(
            df_conflitos_municipio,
            x='Número_Conflitos',
            y='Município',
            orientation='h',
            title='Número de Conflitos por Município',
            labels={'Número_Conflitos': 'Número de Conflitos', 'Município': 'Município'},
            text='Número_Conflitos',
            color='Município',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_conflitos.update_layout(
            height=600,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial", size=12),
            hovermode="y unified",
            showlegend=False
        )
        fig_conflitos.update_traces(texttemplate='<b>%{text:.0f}</b>', textposition='outside')
        st.plotly_chart(fig_conflitos, use_container_width=True)
    
    with tab4:
        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
            "Municípios", "Temporal", "Classes", "Assuntos", "Órgãos"
        ])
        
        with subtab1:
            st.plotly_chart(fig_municipio, use_container_width=True)
        
        with subtab2:
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        with subtab3:
            st.plotly_chart(fig_classe, use_container_width=True)
        
        with subtab4:
            st.plotly_chart(fig_assuntos, use_container_width=True)
        
        with subtab5:
            st.plotly_chart(fig_orgaos, use_container_width=True)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f5f7fa;
}
[data-testid="stHeader"] {
    background: rgba(0,116,217,0.1);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e1e4e8;
}
</style>
""", unsafe_allow_html=True)
