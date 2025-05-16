import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colors

st.set_page_config(layout="wide")

# --- FUNÇÕES AUXILIARES ---
def converter_duracao_em_minutos(valor):
    try:
        if isinstance(valor, str) and 'h' in valor.lower():
            partes = valor.lower().split('h')
            horas = int(partes[0].strip()) if partes[0].strip().isdigit() else 0
            minutos = int(partes[1].strip()) if len(partes) > 1 and partes[1].strip().isdigit() else 0
            return horas * 60 + minutos
    except:
        return None
    return None

def converter_duracao_em_horas(valor):
    minutos = converter_duracao_em_minutos(valor)
    return minutos / 60 if minutos else None

# --- CARREGAMENTO DOS DADOS ---
uploaded_file = st.file_uploader("Carregar ficheiro CSV com os cursos", type=[".csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    df['Duração_minutos'] = df['Duração'].apply(converter_duracao_em_minutos)
    df['Duração_horas'] = df['Duração_minutos'] / 60
    df['Preço (€)'] = pd.to_numeric(df['Preço (€)'], errors='coerce')

    plataformas = df['Plataforma'].dropna().unique().tolist()
    anos = df['Data de Criação'].dropna().unique().tolist()

    # --- FILTROS ---
    with st.sidebar:
        st.header("Filtros")

        # Sliders adicionais para Preço e Duração
        preco_min = float(df['Preço (€)'].min(skipna=True))
        preco_max = float(df['Preço (€)'].max(skipna=True))
        preco_range = st.slider("Intervalo de Preço (€)", min_value=0.0, max_value=preco_max, value=(preco_min, preco_max))

        duracao_min = float(df['Duração_horas'].min(skipna=True))
        duracao_max = float(df['Duração_horas'].max(skipna=True))
        duracao_range = st.slider("Intervalo de Duração (horas)", min_value=0.0, max_value=duracao_max, value=(duracao_min, duracao_max))
        plataformas_selecionadas = st.multiselect("Selecionar plataformas", sorted(plataformas), default=sorted(plataformas))
        incluir_sem_ano = st.checkbox("Incluir cursos sem ano", value=True)
        anos_selecionados = st.multiselect("Selecionar anos", sorted(anos), default=sorted(anos))

    if incluir_sem_ano:
        df_filtrado = df[
            df['Plataforma'].isin(plataformas_selecionadas) &
            ((df['Data de Criação'].isin(anos_selecionados)) | (df['Data de Criação'].isna())) &
            (df['Preço (€)'].between(preco_range[0], preco_range[1], inclusive='both')) &
            (df['Duração_horas'].between(duracao_range[0], duracao_range[1], inclusive='both'))
        ]
    else:
        df_filtrado = df[
            df['Plataforma'].isin(plataformas_selecionadas) &
            df['Data de Criação'].isin(anos_selecionados) &
            (df['Preço (€)'].between(preco_range[0], preco_range[1], inclusive='both')) &
            (df['Duração_horas'].between(duracao_range[0], duracao_range[1], inclusive='both'))
        ]

    # --- GRÁFICOS EM ABAS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Cursos por Plataforma", "Duração vs Preço", "Calculadora de Preço", "Duração Média por Plataforma", "Preço Médio por Plataforma", "Preço Médio por Hora", "Cursos por Ano"])

    with tab1:
        st.subheader("Número de Cursos por Plataforma (com 2 ou mais cursos)")
        df_valid = df_filtrado[df_filtrado['Plataforma'].notna() & (df_filtrado['Plataforma'].str.strip() != '')]
        contagem = df_valid['Plataforma'].value_counts()
        contagem_filtrada = contagem[contagem >= 2]

        if not contagem_filtrada.empty:
            cores = plt.cm.tab10.colors
            plataformas = contagem_filtrada.index.tolist()
            num_plataformas = len(plataformas)
            cores_utilizadas = cores * (num_plataformas // len(cores) + 1)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(num_plataformas), contagem_filtrada.values, color=cores_utilizadas[:num_plataformas])

            for i, bar in enumerate(bars):
                altura = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, altura / 2, f'{int(altura)}',
                        ha='center', va='center', color='white', fontsize=12, fontweight='bold')

            ax.set_xticks([])

            for i, plataforma in enumerate(plataformas):
                ax.bar(0, 0, color=cores_utilizadas[i], label=plataforma)

            legenda = ax.legend(
                title="Plataforma",
                title_fontsize=12,
                fontsize=10,
                frameon=True,
                edgecolor='red'
            )
            legenda.get_frame().set_linewidth(1)
            legenda.get_frame().set_alpha(0.5)

            for text in legenda.get_texts():
                text.set_fontweight('bold')

            ax.set_title('Número de Cursos por Plataforma (com 2 ou mais cursos)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Número de Cursos', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelsize=11, labelcolor='black')
            for label in ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Nenhuma plataforma com 2 ou mais cursos encontrada com os filtros selecionados.")

    with tab2:
        st.subheader("Relação entre Duração e Preço dos Cursos (Pagos)")

        df_validos = df_filtrado[
            (df_filtrado['Preço (€)'].notna()) &
            (df_filtrado['Preço (€)'] > 0) &
            (df_filtrado['Duração_horas'].notna())
        ]

        if not df_validos.empty:
            correlacao = df_validos['Duração_horas'].corr(df_validos['Preço (€)'])
            x = df_validos['Duração_horas']
            y = df_validos['Preço (€)']
            a, b = np.polyfit(x, y, 1)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.6}, line_kws={'color': 'red'}, ax=ax)
            equacao = f'y = {a:.2f}x + {b:.2f}'
            ax.text(0.05, max(y) * 0.95, equacao, fontsize=12, color='red')
            ax.set_title(f"Relação entre Duração e Preço dos Cursos (Pagos)\nCorrelação: {correlacao:.2f}", fontsize=14, fontweight='bold')
            ax.set_xlabel('Duração (horas)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Preço (€)', fontsize=12, fontweight='bold')
            ax.tick_params(axis='both', labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("Sem dados suficientes para mostrar a dispersão entre duração e preço com os filtros atuais.")

    with tab3:
        st.subheader("Calculadora de Preço Estimado do Curso")

        if 'a' in locals() and 'b' in locals():
            duracao_input = st.number_input("Introduza a duração do curso (em horas):", min_value=0.0, step=0.5)
            preco_estimado = a * duracao_input + b
            st.write(f"Preço estimado com base na duração: **{preco_estimado:.2f} €**")
        else:
            st.info("A equação de regressão ainda não foi calculada. Vá ao separador 'Duração vs Preço' para gerar os coeficientes.")
    with tab4:
        st.subheader("Duração Média dos Cursos por Plataforma (≥ 2 cursos)")

        cores_por_plataforma = {
            'YouTube': '#FF0000',
            'Udemy': '#A435F0',
            'ISA': '#006633',
            'Coursera': '#0056D2',
            'EDX': '#2A2B2D',
            'ConservationTraining.org': '#3B6E8F',
            'FAO elearning Academy': '#1A5C3D',
            'FCUL': '#0056D2',
            'Future Learn': '#d63384',
            'Convention on Biological Diversity (CBD)': '#006400',
            'Swayam': '#126e82',
            'UAlg': '#002855',
            'FCUP': '#005B82',
            'Learning for Nature': '#38761D',
            'Open Learn': '#660066',
            'Global Biodiversity Information Facility': '#009688'
        }

        df_valid_duracao = df_filtrado[
            (df_filtrado['Plataforma'].notna()) &
            (df_filtrado['Duração_minutos'].notna()) &
            (df_filtrado['Duração_minutos'] > 0)
        ]

        contagem_duracao = df_valid_duracao['Plataforma'].value_counts()
        plataformas_validas_duracao = contagem_duracao[contagem_duracao >= 2].index

        media_duracao = df_valid_duracao[df_valid_duracao['Plataforma'].isin(plataformas_validas_duracao)] \
            .groupby('Plataforma')['Duração_minutos'].mean().sort_values(ascending=False) / 60

        cores_usadas = [cores_por_plataforma.get(p, 'gray') for p in media_duracao.index]

        fig, ax = plt.subplots(figsize=(12, 10))
        bars = ax.bar(media_duracao.index, media_duracao.values, color=cores_usadas)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f} h",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

        ax.set_title('Duração Média dos Cursos por Plataforma (≥ 2 cursos)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Duração Média (horas)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(media_duracao.index)))
        ax.set_xticklabels(media_duracao.index, rotation=90, fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        st.pyplot(fig)
    with tab5:
        st.subheader("Preço Médio dos Cursos Pagos por Plataforma (≥ 2 cursos)")

        df_valid_preco = df_filtrado[
            (df_filtrado['Plataforma'].notna()) &
            (df_filtrado['Preço (€)'].notna()) &
            (df_filtrado['Preço (€)'] > 0)
        ]

        contagem_precos = df_valid_preco.groupby('Plataforma').size()
        plataformas_validas_precos = contagem_precos[contagem_precos >= 2].index

        media_preco = df_valid_preco[df_valid_preco['Plataforma'].isin(plataformas_validas_precos)] \
            .groupby('Plataforma')['Preço (€)'].mean().sort_values(ascending=False)

        cores_usadas = [cores_por_plataforma.get(p, 'gray') for p in media_preco.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(media_preco.index, media_preco.values, color=cores_usadas)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}€",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

        ax.set_title('Preço Médio dos Cursos Pagos por Plataforma (≥ 2 cursos)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Preço Médio (€)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(media_preco.index)))
        ax.set_xticklabels(media_preco.index, fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        st.pyplot(fig)
    with tab6:
        st.subheader("Preço Médio por Hora dos Cursos Pagos por Plataforma (≥ 2 cursos)")

        df_valid_hora = df_filtrado[
            (df_filtrado['Plataforma'].notna()) &
            (df_filtrado['Preço (€)'].notna()) &
            (df_filtrado['Preço (€)'] > 0) &
            (df_filtrado['Duração_minutos'].notna()) &
            (df_filtrado['Duração_minutos'] > 0)
        ]

        df_valid_hora['Preço_por_hora'] = df_valid_hora['Preço (€)'] / (df_valid_hora['Duração_minutos'] / 60)

        contagem_precos_hora = df_valid_hora.groupby('Plataforma').size()
        plataformas_validas_hora = contagem_precos_hora[contagem_precos_hora >= 2].index

        media_preco_hora = df_valid_hora[df_valid_hora['Plataforma'].isin(plataformas_validas_hora)] \
            .groupby('Plataforma')['Preço_por_hora'].mean().sort_values(ascending=False)

        cores_usadas = [cores_por_plataforma.get(p, 'gray') for p in media_preco_hora.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(media_preco_hora.index, media_preco_hora.values, color=cores_usadas)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f} €/h",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')

        ax.set_title('Preço Médio por Hora dos Cursos Pagos por Plataforma (≥ 2 cursos)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Preço Médio por Hora (€)', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(media_preco_hora.index)))
        ax.set_xticklabels(media_preco_hora.index, fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelsize=12)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        st.pyplot(fig)
    with tab7:
        st.subheader("Número de Cursos por Ano de Criação")

        df_ano = df[df['Data de Criação'].notna()].copy()
        df_ano['Data de Criação'] = df_ano['Data de Criação'].astype(int)
        cursos_por_ano = df_ano['Data de Criação'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(cursos_por_ano.index.astype(str), cursos_por_ano.values, color='green')

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title('Número de Cursos por Ano de Criação', fontsize=14, fontweight='bold')
        ax.set_xlabel('Ano', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Cursos', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("Por favor, carrega um ficheiro CSV com os dados dos cursos.")

