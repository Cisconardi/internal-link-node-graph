import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import io # Per gestire l'upload del file

# La funzione crea_grafo_link_interni adattata per Streamlit
def crea_grafo_link_interni_streamlit(
    df_input, 
    stringhe_url_da_escludere_selezionate=None, 
    dominio_da_includere=None,
    nome_file_export_csv="report_nodi_grafo.csv", 
    colonna_anchor_text="Anchor",
    colonna_tipo_record="Type", 
    valore_tipo_da_includere=None,
    abilita_evidenziazione_stringa_url=False,
    stringa_url_da_evidenziare="",          
    colore_evidenziazione_url="#00FF00",     
    abilita_evidenziazione_anchor_arco=False, # Nuovo parametro per archi
    stringa_anchor_da_evidenziare="",        # Nuovo parametro per archi
    colore_evidenziazione_anchor_arco="#FF03EC" # Nuovo parametro per archi (default viola)
):
    """
    Genera una rappresentazione a grafo navigabile dei link interni da un DataFrame,
    restituisce la figura Plotly e salva un report CSV.
    Aggiunta funzionalità per evidenziare nodi e archi.
    """
    df = df_input.copy() 
    log_messages = []

    if df.empty: 
        return None, ["DataFrame di input vuoto."]

    col_source, col_dest = 'Source', 'Destination'
    col_anchor = colonna_anchor_text
    
    cols_base_req = [col_source, col_dest]
    for col in cols_base_req:
        if col not in df.columns:
            msg = f"Errore: Colonna base '{col}' mancante nel file CSV."
            return None, [msg]
            
    if col_anchor and col_anchor not in df.columns:
        log_messages.append(f"Avviso: Colonna anchor '{col_anchor}' specificata ma non trovata. Gli archi non avranno info sugli anchor.")
        col_anchor = None 

    if colonna_tipo_record and colonna_tipo_record not in df.columns:
        log_messages.append(f"Avviso: Colonna tipo record '{colonna_tipo_record}' specificata ma non trovata. Il filtro per tipo di record verrà saltato.")
        colonna_tipo_record = None

    log_messages.append("Inizio pre-processing DataFrame...")
    righe_iniziali_totali = len(df)
    log_messages.append(f"Numero di righe iniziali nel DataFrame: {righe_iniziali_totali}")

    df[col_source] = df[col_source].astype(str).str.strip()
    df[col_dest] = df[col_dest].astype(str).str.strip()
    if col_anchor and col_anchor in df.columns: 
        df[col_anchor] = df[col_anchor].astype(str).str.strip().replace('', np.nan)

    righe_prima_del_filtro_corrente = len(df)
    if colonna_tipo_record and valore_tipo_da_includere and valore_tipo_da_includere != "Nessuno":
        log_messages.append(f"Applicazione filtro tipo record: mantenimento righe se '{colonna_tipo_record}' è '{valore_tipo_da_includere}' (case-insensitive).")
        if colonna_tipo_record in df.columns:
            df[colonna_tipo_record] = df[colonna_tipo_record].astype(str).str.strip()
            df = df[df[colonna_tipo_record].str.lower() == valore_tipo_da_includere.lower()]
            log_messages.append(f"  Righe dopo filtro tipo record: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    df.dropna(subset=[col_source, col_dest], inplace=True)
    df = df[(df[col_source].str.len() > 0) & (df[col_dest].str.len() > 0)]
    log_messages.append(f"Righe dopo rimozione NA/vuoti in Source/Destination: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    str_exclude_lower = [s.lower().strip() for s in stringhe_url_da_escludere_selezionate if s and s.strip()] if stringhe_url_da_escludere_selezionate else []
    dom_include_lower = dominio_da_includere.lower().strip() if dominio_da_includere and dominio_da_includere.strip() else None

    if str_exclude_lower:
        log_messages.append(f"Filtro esclusione URL (Source/Dest): {', '.join(str_exclude_lower)}")
        m_src = pd.Series([True]*len(df),index=df.index); m_dst = pd.Series([True]*len(df),index=df.index)
        for s_ex in str_exclude_lower:
            if s_ex: 
                m_src &= ~df[col_source].str.lower().str.contains(s_ex, na=False, regex=False)
                m_dst &= ~df[col_dest].str.lower().str.contains(s_ex, na=False, regex=False)
        df = df[m_src & m_dst] 
        log_messages.append(f"  Righe dopo filtro esclusione URL: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    if dom_include_lower:
        log_messages.append(f"Filtro inclusione dominio (Dest): '{dom_include_lower}'")
        df = df[df[col_dest].str.lower().str.contains(dom_include_lower, na=False, regex=False)]
        log_messages.append(f"  Righe dopo filtro inclusione dominio: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)
        
    df = df[df[col_source] != df[col_dest]]
    log_messages.append(f"Righe dopo rimozione auto-link: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    
    if df.empty: 
        log_messages.append("DataFrame vuoto post-filtri. Nessun grafo da generare."); 
        return None, log_messages
        
    log_messages.append("Aggregazione anchor text...")
    
    def aggregate_anchors_custom(series):
        valid_anchors = series.dropna().astype(str).str.strip().unique()
        cleaned_anchors = sorted([anchor for anchor in valid_anchors if anchor]) 
        if not cleaned_anchors:
            return ["Vuoto"] 
        return cleaned_anchors

    if col_anchor and col_anchor in df.columns:
        df_agg = df.groupby([col_source, col_dest]).agg(
            Unique_Anchors=(col_anchor, aggregate_anchors_custom)
        ).reset_index()
    else:
        df_agg = df[[col_source, col_dest]].drop_duplicates().reset_index(drop=True)
        df_agg['Unique_Anchors'] = [["Vuoto"] for _ in range(len(df_agg))]

    if df_agg.empty: 
        log_messages.append("DataFrame aggregato vuoto. Nessun grafo da generare."); 
        return None, log_messages
    log_messages.append(f"Archi unici per grafo: {len(df_agg)}")

    G = nx.DiGraph()
    for _, riga in df_agg.iterrows():
        G.add_edge(riga[col_source], riga[col_dest], anchors=riga['Unique_Anchors'])

    log_messages.append(f"Nodi nel grafo: {G.number_of_nodes()}, Archi nel grafo: {G.number_of_edges()}")
    if G.number_of_nodes() == 0: 
        log_messages.append("Grafo vuoto."); 
        return None, log_messages

    try:
        pd.DataFrame(
            [{'Nodo_URL': n, 'OutDegree': G.out_degree(n), 'InDegree': G.in_degree(n)} for n in G.nodes()]
        ).to_csv(nome_file_export_csv, index=False, encoding='utf-8-sig')
        log_messages.append(f"Dati nodi esportati in '{nome_file_export_csv}'. (Salvato sul server)")
    except Exception as e: 
        log_messages.append(f"Errore esportazione CSV: {e}")

    log_messages.append("Calcolo layout...")
    pos = None; layout_3d = False
    node_count = G.number_of_nodes()
    if 0 < node_count < 1500:
        try: 
            k_val = (0.5/np.sqrt(node_count) if node_count >0 else 0.5)
            pos = nx.spring_layout(G, dim=3, k=k_val, iterations=50, seed=42, scale=3)
            layout_3d = True; log_messages.append("Layout 3D calcolato.")
        except Exception as e_3d: log_messages.append(f"Errore Layout 3D ({e_3d}), fallback a 2D."); pos = None
    
    if pos is None and node_count > 0:
        try: 
            k_val = (0.8/np.sqrt(node_count) if node_count >0 else 0.8)
            pos = nx.spring_layout(G, dim=2, k=k_val, iterations=50, seed=42, scale=3)
            layout_3d = False; log_messages.append("Layout 2D (spring) calcolato.")
        except Exception as e_2d_s: 
            log_messages.append(f"Errore Layout 2D spring ({e_2d_s}), fallback a kamada_kawai.")
            try: 
                pos = nx.kamada_kawai_layout(G, dim=2, scale=3)
                layout_3d = False; log_messages.append("Layout 2D (kamada_kawai) calcolato.")
            except Exception as e_2d_k: 
                log_messages.append(f"Errore tutti i layout ({e_2d_k}). Visualizzazione annullata."); 
                return None, log_messages
    
    if pos is None: 
        log_messages.append("Layout non calcolato. Visualizzazione annullata."); 
        return None, log_messages

    log_messages.append("Preparazione visualizzazione Plotly...")
    
    # Preparazione tracce archi (default e evidenziati)
    default_edge_x, default_edge_y, default_edge_z, default_edge_hover_texts = [], [], [], []
    highlight_edge_x, highlight_edge_y, highlight_edge_z, highlight_edge_hover_texts = [], [], [], []
    
    stringa_anchor_lower = stringa_anchor_da_evidenziare.lower().strip() if abilita_evidenziazione_anchor_arco and stringa_anchor_da_evidenziare else ""

    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            pos_u, pos_v = pos[u], pos[v]
            anchors = data.get('anchors', ["Vuoto"]) 
            
            hover_text_content = "N/A" 
            if anchors: 
                display_anchors = anchors[:7]
                hover_text_content = "<br>".join(f"- {str(a)}" for a in display_anchors)
                if len(anchors) > 7: 
                    hover_text_content += f"<br>... e altri {len(anchors) - 7} anchor(s)"
                elif not any(a for a in anchors if a != "Vuoto") and "Vuoto" in anchors : 
                     hover_text_content = "- Vuoto"
            full_hover_text = f"<b>Link</b><br>Da: {u}<br>A: {v}<br>--- Anchor Texts ---<br>{hover_text_content}"

            # Determina se l'arco deve essere evidenziato
            evidenzia_arco = False
            if abilita_evidenziazione_anchor_arco and stringa_anchor_lower:
                for anchor in anchors:
                    if stringa_anchor_lower in str(anchor).lower():
                        evidenzia_arco = True
                        break
            
            if evidenzia_arco:
                highlight_edge_x.extend([pos_u[0], pos_v[0], None])
                highlight_edge_y.extend([pos_u[1], pos_v[1], None])
                if layout_3d: highlight_edge_z.extend([pos_u[2], pos_v[2], None])
                highlight_edge_hover_texts.extend([full_hover_text, full_hover_text, None])
            else:
                default_edge_x.extend([pos_u[0], pos_v[0], None])
                default_edge_y.extend([pos_u[1], pos_v[1], None])
                if layout_3d: default_edge_z.extend([pos_u[2], pos_v[2], None])
                default_edge_hover_texts.extend([full_hover_text, full_hover_text, None])

    traces = []
    # Traccia archi di default
    trace_default_edges_args = dict(line=dict(width=0.7, color='#888'), mode='lines', hoverinfo='text', text=default_edge_hover_texts, opacity=0.7, name='Archi')
    if layout_3d: traces.append(go.Scatter3d(x=default_edge_x, y=default_edge_y, z=default_edge_z, **trace_default_edges_args))
    else: traces.append(go.Scatter(x=default_edge_x, y=default_edge_y, **trace_default_edges_args))

    # Traccia archi evidenziati (se ce ne sono)
    if highlight_edge_x: # Aggiungi solo se ci sono archi da evidenziare
        trace_highlight_edges_args = dict(line=dict(width=1.5, color=colore_evidenziazione_anchor_arco), mode='lines', hoverinfo='text', text=highlight_edge_hover_texts, opacity=0.9, name='Archi Evidenziati')
        if layout_3d: traces.append(go.Scatter3d(x=highlight_edge_x, y=highlight_edge_y, z=highlight_edge_z, **trace_highlight_edges_args))
        else: traces.append(go.Scatter(x=highlight_edge_x, y=highlight_edge_y, **trace_highlight_edges_args))


    # Nodi (logica di colorazione URL e per categoria)
    node_degrees = dict(G.degree())
    node_values = list(node_degrees.values()) if node_degrees else []
    # node_traces_list = [] # Verrà aggiunta a `traces`
    default_node_colors_by_category = {'Basso Grado': 'blue', 'Medio Grado': 'orange', 'Alto Grado': 'red'}
    stringa_url_da_evidenziare_lower = stringa_url_da_evidenziare.lower().strip() if abilita_evidenziazione_stringa_url and stringa_url_da_evidenziare else ""

    if node_values:
        q33, q67 = (np.percentile(node_values,33), np.percentile(node_values,67)) if len(set(node_values))>2 else (min(node_values, default=0),max(node_values, default=0))
        if q33 == q67 and len(set(node_values)) > 1: q33 = min(node_values, default=0); q67 = np.percentile(node_values, 50)
        if q33 == q67 and q67 < max(node_values, default=q67): q67 = max(node_values, default=q67)

        nodes_cat_data = {'Basso Grado': [], 'Medio Grado': [], 'Alto Grado': []}
        for n_id, deg in node_degrees.items():
            cat = 'Medio Grado';
            if deg <= q33: cat = 'Basso Grado'
            elif deg > q67 or (deg > q33 and q33 == q67): cat = 'Alto Grado'
            nodes_cat_data[cat].append(n_id)
        
        for cat_name, nodes_in_cat in nodes_cat_data.items():
            if not nodes_in_cat: continue
            cat_x, cat_y, cat_z, cat_text, cat_size, cat_customdata = [], [], [], [], [], []
            node_marker_colors_list = [] 

            for node_id in nodes_in_cat:
                if node_id not in pos: continue
                cat_x.append(pos[node_id][0]); cat_y.append(pos[node_id][1])
                if layout_3d: cat_z.append(pos[node_id][2])
                in_degree = G.in_degree(node_id); out_degree = G.out_degree(node_id)
                cat_text.append(f"URL: {node_id}<br>Grado Totale: {node_degrees[node_id]}<br>Link Entranti: {in_degree}<br>Link Uscenti: {out_degree}")
                size_val = node_degrees.get(node_id, 0)
                cat_size.append(max(5, min(15, size_val * 1.5 if size_val > 0 else 5)))
                cat_customdata.append(node_id) 

                current_node_color = default_node_colors_by_category[cat_name] 
                if abilita_evidenziazione_stringa_url and stringa_url_da_evidenziare_lower:
                    if stringa_url_da_evidenziare_lower in node_id.lower():
                        current_node_color = colore_evidenziazione_url
                node_marker_colors_list.append(current_node_color)

            marker_style = dict(color=node_marker_colors_list, size=cat_size, opacity=0.9, line=dict(width=0.5, color='#333')) 
            scatter_args = dict(name=cat_name, mode='markers', hoverinfo='text', text=cat_text, marker=marker_style, customdata=cat_customdata) 
            if layout_3d: traces.append(go.Scatter3d(x=cat_x, y=cat_y, z=cat_z, **scatter_args))
            else: traces.append(go.Scatter(x=cat_x, y=cat_y, **scatter_args))
            
    common_layout_properties = dict(
        showlegend=True, 
        legend_title_text='Legenda', # Titolo legenda più generico
        hovermode='closest', 
        margin=dict(b=40,l=5,r=5,t=60)
    ) 
    title_font_properties = dict(size=18)
    title_alignment_properties = dict(x=0.5)

    if layout_3d: 
        fig_layout = go.Layout(
            title=dict(text='<b>Grafo Link Interni (3D) con Anchor Text</b>', font=title_font_properties, **title_alignment_properties),
            scene=dict(bgcolor="rgba(240,240,240,0.95)"), **common_layout_properties
        )
    else: 
        fig_layout = go.Layout(
            title=dict(text='<b>Grafo Link Interni (2D) con Anchor Text</b>', font=title_font_properties, **title_alignment_properties), 
            plot_bgcolor='rgba(245,245,245,1)', **common_layout_properties
        )
    
    figura = go.Figure(data=traces, layout=fig_layout) # Usa la lista `traces` combinata
    log_messages.append("Visualizzazione grafico completata.")
    return figura, log_messages


# --- Applicazione Streamlit ---
st.set_page_config(layout="wide", page_title="Visualizzatore Grafo Link Interni")

st.title("Visualizzatore Interattivo Grafo Link Interni")
st.markdown("""
Carica un file CSV con i link interni (colonne richieste: `Source`, `Destination`; opzionali: `Anchor`, `Type`) 
e personalizza i filtri per visualizzare la struttura del grafo.
""")

log_placeholder = st.empty() 

with st.sidebar: 
    st.header("Opzioni di Filtro e Controllo")
    uploaded_file = st.file_uploader("Carica il tuo file CSV dei link", type=["csv"])

    default_stringhe_da_escludere = [     
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 
        '.css', '.js', '.svg', '.ico', 
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar.gz', 
        '.mp3', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.wmv', 
        '.xml', '.json', '.txt', 
        'mailto:', 'tel:', '#', 'javascript:void(0)',
        'googleads.g.doubleclick.net', 'facebook.com', 
        '/autodiscover/autodiscover.xml'
    ]

    dominio_input = st.text_input("Dominio da Includere (es. 'sitoesempio.com')", value="sitoesempio.com") 

    tipo_record_opzioni = ["Nessuno", "Hyperlink", "HTTP Redirect", "HTML Canonical", "Image"] 
    tipo_record_selezionato = st.selectbox(
        "Valore Tipo Record da Includere (opzionale, colonna 'Type')", 
        options=tipo_record_opzioni, 
        index=1 
    )
    valore_tipo_da_usare = None if tipo_record_selezionato == "Nessuno" else tipo_record_selezionato

    st.markdown("---")
    st.subheader("Filtri URL da Escludere (seleziona per escludere):")
    
    if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = {s_escl: True for s_escl in default_stringhe_da_escludere}

    stringhe_escluse_selezionate_ui = []
    for s_escl in default_stringhe_da_escludere:
        checkbox_key = f"cb_{s_escl.replace('.', '_dot_').replace('/', '_slash_').replace(':', '_colon_').replace('#','_hash_')}"
        if checkbox_key not in st.session_state.checkbox_states:
            st.session_state.checkbox_states[checkbox_key] = True 

        is_checked = st.checkbox(
            s_escl, 
            value=st.session_state.checkbox_states[checkbox_key], 
            key=checkbox_key 
        )
        st.session_state.checkbox_states[checkbox_key] = is_checked 
        if is_checked: 
            stringhe_escluse_selezionate_ui.append(s_escl)
    
    custom_exclusions_input = st.text_input("Altri filtri URL da escludere (separati da virgola):", key="custom_exclusions_text")
    if custom_exclusions_input:
        custom_exclusions_list = [item.strip() for item in custom_exclusions_input.split(',') if item.strip()]
        stringhe_escluse_selezionate_ui.extend(custom_exclusions_list) 
        stringhe_escluse_selezionate_ui = sorted(list(set(stringhe_escluse_selezionate_ui)))

    st.caption(f"Stringhe URL totali per l'esclusione: {stringhe_escluse_selezionate_ui if stringhe_escluse_selezionate_ui else 'Nessuna'}")
    
    st.markdown("---")
    st.subheader("Evidenziazione URL Nodi")
    abilita_evidenziazione_stringa_url_ui = st.checkbox("Abilita evidenziazione URL per stringa", key="enable_highlight_url_str")
    stringa_da_cercare_url_ui = ""
    colore_scelto_url_ui = "#00FF00" 
    if abilita_evidenziazione_stringa_url_ui:
        stringa_da_cercare_url_ui = st.text_input("Stringa da cercare nell'URL del nodo (case-insensitive):", key="highlight_url_str_text")
        colore_scelto_url_ui = st.color_picker("Colore di evidenziazione per URL nodo:", value="#00FF00", key="highlight_url_color")

    st.markdown("---")
    st.subheader("Evidenziazione Anchor Text Archi")
    abilita_evidenziazione_anchor_ui = st.checkbox("Abilita evidenziazione anchor text per archi", key="enable_highlight_anchor_str")
    stringa_da_cercare_anchor_ui = ""
    colore_scelto_anchor_ui = "#800080" # Viola di default
    if abilita_evidenziazione_anchor_ui:
        stringa_da_cercare_anchor_ui = st.text_input("Stringa da cercare nell'anchor text (case-insensitive):", key="highlight_anchor_str_text")
        colore_scelto_anchor_ui = st.color_picker("Colore di evidenziazione per anchor arco:", value="#800080", key="highlight_anchor_color")


    st.markdown("---")
    st.info("Modifica i filtri e il grafico si aggiornerà automaticamente al caricamento di un nuovo file o al cambio di un'opzione (se il file è già caricato).")


if uploaded_file is not None:
    try:
        df_caricato = pd.read_csv(uploaded_file, dtype=str)
        st.success(f"File '{uploaded_file.name}' caricato con successo. ({len(df_caricato)} righe)")

        with st.spinner("Generazione del grafo in corso... Questo potrebbe richiedere alcuni istanti."):
            figura_plotly, log_output = crea_grafo_link_interni_streamlit(
                df_input=df_caricato,
                stringhe_url_da_escludere_selezionate=stringhe_escluse_selezionate_ui, 
                dominio_da_includere=dominio_input if dominio_input else None,
                nome_file_export_csv="report_nodi_grafo_streamlit.csv",
                colonna_anchor_text="Anchor", 
                colonna_tipo_record="Type",   
                valore_tipo_da_includere=valore_tipo_da_usare,
                abilita_evidenziazione_stringa_url=abilita_evidenziazione_stringa_url_ui, 
                stringa_url_da_evidenziare=stringa_da_cercare_url_ui,        
                colore_evidenziazione_url=colore_scelto_url_ui,
                abilita_evidenziazione_anchor_arco=abilita_evidenziazione_anchor_ui, # Passa nuovo parametro
                stringa_anchor_da_evidenziare=stringa_da_cercare_anchor_ui,       # Passa nuovo parametro
                colore_evidenziazione_anchor_arco=colore_scelto_anchor_ui         # Passa nuovo parametro
            )
        
        log_placeholder.text_area("Log di Pre-processing", "\n".join(log_output), height=250)

        if figura_plotly:
            st.plotly_chart(figura_plotly, use_container_width=True, height=800)
            st.caption("Interagisci con il grafo: zoom, pan, rotazione (3D), hover per dettagli.")
            try:
                with open("report_nodi_grafo_streamlit.csv", "rb") as fp:
                    st.download_button(
                        label="Scarica Report Nodi (CSV)",
                        data=fp,
                        file_name="report_nodi_grafo.csv", 
                        mime="text/csv"
                    )
            except FileNotFoundError:
                st.warning("File report nodi ('report_nodi_grafo_streamlit.csv') non trovato. Potrebbe non essere stato ancora generato o c'è stato un errore.")
            except Exception as e_dl:
                 st.warning(f"Errore nel preparare il download del report nodi: {e_dl}")
        else:
            pass

    except Exception as e:
        st.error(f"Errore critico durante l'elaborazione del file o la generazione del grafo: {e}")
        st.exception(e) 
else:
    log_placeholder.info("Attendo il caricamento di un file CSV per visualizzare il grafo e i log.")

