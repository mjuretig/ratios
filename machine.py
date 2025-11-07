import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("Visualizador de Ratios de Opciones")

# --- Subir múltiples archivos ---
uploaded_files = st.file_uploader("📂 Subí los archivos Excel (.xlsb)", type=["xlsb"], accept_multiple_files=True)

if uploaded_files:
    # Diccionario para almacenar dataframes por archivo
    dfs = {}
    sheet_names_dict = {}
    hojas_encontradas = {}

    posibles_meses = ["Febrero", "Abril", "Junio", "Agosto", "Octubre", "Diciembre","Opex Oct","Jun","AB"]
    posibles_anios = ["2022", "2023", "2024", "2025"]

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        # Leer nombres de hojas
        xl = pd.ExcelFile(uploaded_file, engine="pyxlsb")
        sheet_names = xl.sheet_names
        sheet_names_dict[file_name] = sheet_names

        # --- Buscar hoja válida ---
        hoja_encontrada = None
        for s in sheet_names:
            if "Lotes" in s:
                for mes in posibles_meses:
                    for anio in posibles_anios:
                        if f"{mes} {anio}" in s:
                            hoja_encontrada = s
                            break
                    if hoja_encontrada:
                        break
            if hoja_encontrada:
                break

        # Si no se encontró, usar la segunda hoja
        if not hoja_encontrada:
            hoja_encontrada = sheet_names[1] if len(sheet_names) > 1 else sheet_names[0]

        hojas_encontradas[file_name] = hoja_encontrada

        # --- Leer solo columnas A:T ---
        df = pd.read_excel(
            uploaded_file,
            engine="pyxlsb",
            sheet_name=hoja_encontrada,
            usecols="A:T"
        )

        # --- Limpieza ---
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["BASE", "ÚLTIMO"], how="any")

        dfs[file_name] = df

    st.success(f"✅ Archivos cargados: {len(uploaded_files)}")

    # --- DEBUG: información por archivo ---
    with st.expander("🧩 Debug: Información de DataFrames por archivo"):
        for file_name, df in dfs.items():
            st.write(f"**Archivo: {file_name} | Hoja: {hojas_encontradas[file_name]}**")
            st.write("Columnas detectadas:", list(df.columns))
            st.write("Tipos de datos:")
            st.write(df.dtypes)
            st.write("Primeras filas:")
            st.dataframe(df.head())

    # --- Selector de Ratio global ---
    ratio_sel = st.selectbox("Seleccionar Ratio a graficar", ["Ratio1", "Ratio2", "Ratio3"])

    # --- Procesamiento por archivo: selección de BASE y cálculo de ratios ---
    dfs_processed = {}
    for file_name, df in dfs.items():
        with st.expander(f"📊 Configuración para {file_name}"):
            bases_unicas = sorted(df["BASE"].unique())
            bases_sel = st.multiselect(f"Seleccionar BASE para {file_name}", bases_unicas)

            if bases_sel:
                df_filt = df[df["BASE"].isin(bases_sel)].copy()

                # --- Calcular ratios para Calls ---
                df_calls = df_filt[df_filt["TIPO"].str.lower() == "call"].copy()
                if not df_calls.empty:
                    df_calls = df_calls.sort_values(by=["FECHA", "BASE"], ascending=[True, True]).reset_index(drop=True)
                    df_calls["Ratio1"] = df_calls.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-1))
                    df_calls["Ratio2"] = df_calls.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-2))
                    df_calls["Ratio3"] = df_calls.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-3))
                    df_calls = df_calls.replace([float("inf"), float("-inf")], None)

                    # Merge ratios for calls
                    df_filt = df_filt.merge(
                        df_calls[["FECHA", "BASE", "Ratio1", "Ratio2", "Ratio3"]],
                        on=["FECHA", "BASE"],
                        how="left"
                    )

                # --- Calcular ratios para Puts (sort descending for BASE to make ratios >1) ---
                df_puts = df_filt[df_filt["TIPO"].str.lower() == "put"].copy()
                if not df_puts.empty:
                    df_puts = df_puts.sort_values(by=["FECHA", "BASE"], ascending=[True, False]).reset_index(drop=True)
                    df_puts["Ratio1"] = df_puts.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-1))
                    df_puts["Ratio2"] = df_puts.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-2))
                    df_puts["Ratio3"] = df_puts.groupby("FECHA")["ÚLTIMO"].transform(lambda x: x / x.shift(-3))
                    df_puts = df_puts.replace([float("inf"), float("-inf")], None)

                    # Merge ratios for puts with suffixes to avoid conflict
                    df_filt = df_filt.merge(
                        df_puts[["FECHA", "BASE", "Ratio1", "Ratio2", "Ratio3"]],
                        on=["FECHA", "BASE"],
                        how="left",
                        suffixes=('', '_put')
                    )

                    # Combine the ratios
                    for ratio in ["Ratio1", "Ratio2", "Ratio3"]:
                        if f"{ratio}_put" in df_filt.columns:
                            df_filt[ratio] = df_filt[ratio].combine_first(df_filt[f"{ratio}_put"])
                            df_filt.drop(f"{ratio}_put", axis=1, inplace=True)

                # --- Mostrar preview con ratios ---
                st.subheader(f"Vista previa con Ratios para {file_name}")
                preview_cols = ["FECHA", "BASE", "ÚLTIMO", "DÍAS AL VTO.", "TIPO", "PRECIO GGAL", "VI %"]
                ratio_cols = [col for col in ["Ratio1", "Ratio2", "Ratio3"] if col in df_filt.columns]
                st.dataframe(df_filt[preview_cols + ratio_cols].head(30))

                dfs_processed[file_name] = df_filt  # Almacenar el full con ratios

    # --- Sección para Gráfico 2D ---
    if dfs_processed:
        st.markdown("---")
        st.subheader("Sonrisa de Volatilidad + Ratio (Doble Eje) - Superposición de Archivos")

        # --- Checkboxes para seleccionar archivos para 2D ---
        selected_files_2d = []
        for file_name in dfs_processed.keys():
            if st.checkbox(f"Incluir {file_name} en el gráfico 2D", value=True, key=f"chk_2d_{file_name}"):
                selected_files_2d.append(file_name)

        # --- Selector de modo para eje X ---
        x_mode = st.radio("Eje X", ["Absolute Strike", "Moneyness (K/S)"], horizontal=True)

        # --- Checkboxes para visibilidad de series ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            show_vi_calls = st.checkbox("Mostrar VI Calls", value=True)
        with col2:
            show_vi_puts = st.checkbox("Mostrar VI Puts", value=True)
        with col3:
            show_ratio_calls = st.checkbox("Mostrar Ratio Calls", value=True)
        with col4:
            show_ratio_puts = st.checkbox("Mostrar Ratio Puts", value=True)

        if selected_files_2d:
            # --- Asignar colores por archivo ---
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            color_map = {file: colors[i % len(colors)] for i, file in enumerate(selected_files_2d)}

            # --- Combinar días únicos ---
            all_dias = set()
            for file_name in selected_files_2d:
                all_dias.update(dfs_processed[file_name]["DÍAS AL VTO."].unique())
            dias_unicos = sorted(list(all_dias))

            if dias_unicos:
                dias_sel = st.select_slider(
                    "Días al Vencimiento",
                    options=dias_unicos,
                    value=dias_unicos[len(dias_unicos)//2]
                )

                fig2d = go.Figure()

                for file_name in selected_files_2d:
                    df_vi = dfs_processed[file_name].copy()
                    df_vi["VI %"] = pd.to_numeric(df_vi["VI %"], errors='coerce') / 100
                    df_vi["VI %"] = df_vi["VI %"].clip(lower=0)

                    df_dia = df_vi[df_vi["DÍAS AL VTO."] == dias_sel].copy()

                    if df_dia.empty:
                        st.info(f"No hay datos para {dias_sel} días en {file_name}.")
                        continue

                    df_dia[ratio_sel] = pd.to_numeric(df_dia[ratio_sel], errors='coerce')

                    # --- Eje X: Strike o Moneyness ---
                    if x_mode == "Moneyness (K/S)":
                        df_dia["x_plot"] = df_dia["BASE"] / df_dia["PRECIO GGAL"]
                    else:
                        df_dia["x_plot"] = df_dia["BASE"]

                    base_color = color_map[file_name]

                    # --- VI % (Calls) - línea sólida ---
                    calls = df_dia[df_dia["TIPO"].str.lower() == "call"]
                    if show_vi_calls and not calls.empty:
                        fig2d.add_trace(go.Scatter(
                            x=calls["x_plot"],
                            y=calls["VI %"] * 100,
                            mode='lines+markers',
                            name=f'VI Call - {file_name}',
                            line=dict(color=base_color, width=3),
                            marker=dict(size=6),
                            yaxis="y"
                        ))

                    # --- VI % (Puts) - línea sólida, mismo color ---
                    puts = df_dia[df_dia["TIPO"].str.lower() == "put"]
                    if show_vi_puts and not puts.empty:
                        fig2d.add_trace(go.Scatter(
                            x=puts["x_plot"],
                            y=puts["VI %"] * 100,
                            mode='lines+markers',
                            name=f'VI Put - {file_name}',
                            line=dict(color=base_color, width=3),
                            marker=dict(size=6),
                            yaxis="y"
                        ))

                    # --- Ratio (Calls) - línea punteada, color más claro ---
                    calls_ratio = calls.dropna(subset=[ratio_sel])
                    if show_ratio_calls and not calls_ratio.empty:
                        fig2d.add_trace(go.Scatter(
                            x=calls_ratio["x_plot"],
                            y=calls_ratio[ratio_sel],
                            mode='lines+markers',
                            name=f'{ratio_sel} Call - {file_name}',
                            line=dict(color=base_color, width=2, dash='dot'),
                            marker=dict(size=6, symbol='circle-open', color=base_color),
                            yaxis="y2"
                        ))

                    # --- Ratio (Puts) - línea punteada ---
                    puts_ratio = puts.dropna(subset=[ratio_sel])
                    if show_ratio_puts and not puts_ratio.empty:
                        fig2d.add_trace(go.Scatter(
                            x=puts_ratio["x_plot"],
                            y=puts_ratio[ratio_sel],
                            mode='lines+markers',
                            name=f'{ratio_sel} Put - {file_name}',
                            line=dict(color=base_color, width=2, dash='dot'),
                            marker=dict(size=6, symbol='circle-open', color=base_color),
                            yaxis="y2"
                        ))

                    # --- GGAL: línea vertical (misma lógica) ---
                    if x_mode == "Absolute Strike":
                        ggal_precio = df_dia["PRECIO GGAL"].mean()
                        fig2d.add_vline(
                            x=ggal_precio,
                            line=dict(color=base_color, width=2, dash="dash"),
                            annotation_text=f"GGAL {file_name}: {ggal_precio:.2f}",
                            annotation_position="top left"
                        )

                # --- ATM si moneyness ---
                if x_mode == "Moneyness (K/S)":
                    fig2d.add_vline(
                        x=1,
                        line=dict(color="black", width=3, dash="dash"),
                        annotation_text="ATM",
                        annotation_position="top left"
                    )

                # --- Layout ---
                x_title = "Strike (BASE)" if x_mode == "Absolute Strike" else "Moneyness (Strike / Spot)"
                fig2d.update_layout(
                    title=f"Sonrisa VI + {ratio_sel} | {int(dias_sel)} días",
                    xaxis_title=x_title,
                    yaxis=dict(
                        title="Volatilidad Implícita (%)",
                        titlefont=dict(color="green"),
                        tickfont=dict(color="green"),
                        side="left"
                    ),
                    yaxis2=dict(
                        title=ratio_sel,
                        titlefont=dict(color="blue"),
                        tickfont=dict(color="blue"),
                        overlaying="y",
                        side="right",
                        position=0.95
                    ),
                    height=550,
                    hovermode='x unified',
                    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"),
                    margin=dict(l=50, r=70, t=60, b=50)
                )

                st.plotly_chart(fig2d, use_container_width=True)

                # --- Caption ---
                st.caption(
                    f"**Línea sólida**: VI % (Calls/Puts). "
                    f"**Línea punteada**: {ratio_sel} (Calls/Puts). "
                    f"**Color por archivo**. "
                    f"**Línea negra**: ATM (si moneyness) o GGAL (si strike)."
                )

                # --- Sección de Estadísticas ---
                st.markdown("---")
                st.subheader("Estadísticas de Ratios")

                ratio_type = st.radio("Estadísticas para", ["Calls", "Puts"], horizontal=True)

                if ratio_type == "Calls":
                    levels = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
                else:
                    levels = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]

                # --- Estadísticas para el día seleccionado (across files) ---
                stats_day_dict = defaultdict(list)
                for file_name in selected_files_2d:
                    df_vi = dfs_processed[file_name]
                    df_dia = df_vi[df_vi["DÍAS AL VTO."] == dias_sel]
                    if df_dia.empty:
                        continue
                    spot = df_dia["PRECIO GGAL"].mean()
                    tipo_lower = "call" if ratio_type == "Calls" else "put"
                    df_type = df_dia[df_dia["TIPO"].str.lower() == tipo_lower]
                    if df_type.empty:
                        continue
                    df_type["moneyness"] = df_type["BASE"] / spot
                    df_type = df_type.dropna(subset=[ratio_sel, "moneyness"])
                    if df_type.empty:
                        continue
                    for level in levels:
                        idx_closest = (df_type["moneyness"] - level).abs().argmin()
                        diff = abs(df_type.iloc[idx_closest]["moneyness"] - level)
                        if diff < 0.03:  # Umbral de cercanía
                            ratio = df_type.iloc[idx_closest][ratio_sel]
                            stats_day_dict[level].append(ratio)

                # Construir DataFrame para el día
                data_day = {}
                for level in levels:
                    vals = stats_day_dict[level]
                    if vals:
                        s = pd.Series(vals)
                        min_val = s[s > 0].min() if (s > 0).any() else np.nan
                        data_day[level] = [
                            s.mean(),
                            s.median(),
                            min_val,
                            s.max(),
                            s.std() if len(s) > 1 else 0
                        ]
                if data_day:
                    df_stats_day = pd.DataFrame(data_day, index=['Mean', 'Median', 'Min', 'Max', 'Std'])
                    df_stats_day = df_stats_day.round(3)
                    st.subheader(f"Estadísticas para {dias_sel} días ({ratio_type})")
                    st.dataframe(df_stats_day)
                else:
                    st.info(f"No hay datos suficientes para estadísticas en {dias_sel} días ({ratio_type}).")

                # --- Estadísticas acumuladas (días >= dias_sel, across files and days) ---
                stats_cum_dict = defaultdict(list)
                for file_name in selected_files_2d:
                    df_vi = dfs_processed[file_name]
                    df_period = df_vi[df_vi["DÍAS AL VTO."] >= dias_sel]
                    period_days = sorted(df_period["DÍAS AL VTO."].unique())
                    for day in period_days:
                        df_day = df_period[df_period["DÍAS AL VTO."] == day]
                        if df_day.empty:
                            continue
                        spot = df_day["PRECIO GGAL"].mean()
                        tipo_lower = "call" if ratio_type == "Calls" else "put"
                        df_type = df_day[df_day["TIPO"].str.lower() == tipo_lower]
                        if df_type.empty:
                            continue
                        df_type["moneyness"] = df_type["BASE"] / spot
                        df_type = df_type.dropna(subset=[ratio_sel, "moneyness"])
                        if df_type.empty:
                            continue
                        for level in levels:
                            idx_closest = (df_type["moneyness"] - level).abs().argmin()
                            diff = abs(df_type.iloc[idx_closest]["moneyness"] - level)
                            if diff < 0.03:
                                ratio = df_type.iloc[idx_closest][ratio_sel]
                                stats_cum_dict[level].append(ratio)

                # Construir DataFrame acumulado
                data_cum = {}
                for level in levels:
                    vals = stats_cum_dict[level]
                    if vals:
                        s = pd.Series(vals)
                        min_val = s[s > 0].min() if (s > 0).any() else np.nan
                        data_cum[level] = [
                            s.mean(),
                            s.median(),
                            min_val,
                            s.max(),
                            s.std() if len(s) > 1 else 0
                        ]
                if data_cum:
                    df_stats_cum = pd.DataFrame(data_cum, index=['Mean', 'Median', 'Min', 'Max', 'Std'])
                    df_stats_cum = df_stats_cum.round(3)
                    st.subheader(f"Estadísticas acumuladas (días >= {dias_sel}) ({ratio_type})")
                    st.dataframe(df_stats_cum)
                else:
                    st.info(f"No hay datos suficientes para estadísticas acumuladas ({ratio_type}).")

                # --- Sección de Gráfico de Serie de Tiempo ---
                st.markdown("---")
                st.subheader("Serie de Tiempo de Estadísticas de Ratios por Moneyness")

                stat_sel = st.selectbox("Seleccionar Estadística para la Serie", ['Mean', 'Median', 'Min', 'Max', 'Std'])

                # Recopilar datos por día y nivel
                data_by_day = defaultdict(lambda: defaultdict(list))
                period_days = sorted([day for day in dias_unicos if day >= dias_sel], reverse=True)  # Descending: far to near exp

                for file_name in selected_files_2d:
                    df_vi = dfs_processed[file_name]
                    for day in period_days:
                        df_day = df_vi[df_vi["DÍAS AL VTO."] == day]
                        if df_day.empty:
                            continue
                        spot = df_day["PRECIO GGAL"].mean()
                        tipo_lower = "call" if ratio_type == "Calls" else "put"
                        df_type = df_day[df_day["TIPO"].str.lower() == tipo_lower]
                        if df_type.empty:
                            continue
                        df_type["moneyness"] = df_type["BASE"] / spot
                        df_type = df_type.dropna(subset=[ratio_sel, "moneyness"])
                        if df_type.empty:
                            continue
                        for level in levels:
                            idx_closest = (df_type["moneyness"] - level).abs().argmin()
                            diff = abs(df_type.iloc[idx_closest]["moneyness"] - level)
                            if diff < 0.03:
                                ratio = df_type.iloc[idx_closest][ratio_sel]
                                data_by_day[day][level].append(ratio)

                # Construir el gráfico
                fig_ts = go.Figure()

                colors_ts = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2"
                ]
                color_map_ts = {level: colors_ts[i % len(colors_ts)] for i, level in enumerate(levels)}

                has_data = False
                for level in levels:
                    x_vals = []
                    y_vals = []
                    for day in period_days:
                        vals = data_by_day[day][level]
                        if vals:
                            s = pd.Series(vals)
                            if stat_sel == 'Mean':
                                stat_val = s.mean()
                            elif stat_sel == 'Median':
                                stat_val = s.median()
                            elif stat_sel == 'Min':
                                stat_val = s[s > 0].min() if (s > 0).any() else np.nan
                            elif stat_sel == 'Max':
                                stat_val = s.max()
                            elif stat_sel == 'Std':
                                stat_val = s.std() if len(s) > 1 else 0
                            if not np.isnan(stat_val):
                                x_vals.append(day)
                                y_vals.append(stat_val)
                    if x_vals:
                        has_data = True
                        fig_ts.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_vals,
                            mode='lines+markers',
                            name=f'Moneyness {level:.2f}',
                            line=dict(color=color_map_ts[level]),
                            marker=dict(size=6)
                        ))

                if has_data:
                    fig_ts.update_layout(
                        title=f"Serie de Tiempo: {stat_sel} de {ratio_sel} por Moneyness ({ratio_type}) | Días >= {dias_sel}",
                        xaxis_title="Días al Vencimiento (de mayor a menor)",
                        yaxis_title=f"{stat_sel} de {ratio_sel}",
                        height=550,
                        hovermode='x unified',
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.9)"),
                        margin=dict(l=50, r=70, t=60, b=50)
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info(f"No hay datos suficientes para la serie de tiempo ({ratio_type}).")

            else:
                st.info("No hay días disponibles.")
        else:
            st.info("Selecciona al menos un archivo para el gráfico 2D.")