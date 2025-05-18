import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import io # Per gestire l'upload del file

# La funzione crea_grafo_link_interni adattata per Streamlit
# (principalmente cambia figura.show() in return figura)
def crea_grafo_link_interni_streamlit(
    df_input, # DataFrame in input invece del percorso file
    stringhe_url_da_escludere_selezionate=None, 
    dominio_da_includere=None,
    nome_file_export_csv="report_nodi_grafo.csv", # Questo verrà salvato sul server
    colonna_anchor_text="Anchor",
    colonna_tipo_record="Type", 
    valore_tipo_da_includere=None # Modificato per accettare None
):
    """
    Genera una rappresentazione a grafo navigabile dei link interni da un DataFrame,
    restituisce la figura Plotly e salva un report CSV.
    """
    df = df_input.copy() # Lavora su una copia per evitare modifiche all'originale caricato

    # Lista di default completa, usata se stringhe_url_da_escludere_selezionate è None
    # ma in questa app, l'utente farà sempre una selezione (anche se è la selezione di default)
    # stringhe_url_da_escludere_default = [     
    #     '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 
    #     '.css', '.js', '.svg', '.ico', 
    #     # ... (lista completa come nella versione precedente)
    # ]
    # Le stringhe effettive da escludere verranno passate come argomento

    if df.empty: 
        st.warning("Il DataFrame fornito è vuoto.")
        return None

    col_source, col_dest = 'Source', 'Destination'
    col_anchor = colonna_anchor_text
    
    cols_base_req = [col_source, col_dest]
    for col in cols_base_req:
        if col not in df.columns:
            st.error(f"Errore: Colonna base '{col}' mancante nel file CSV.")
            return None
            
    if col_anchor and col_anchor not in df.columns:
        st.warning(f"Avviso: Colonna anchor '{col_anchor}' specificata ma non trovata. Gli archi non avranno info sugli anchor.")
        col_anchor = None 

    if colonna_tipo_record and colonna_tipo_record not in df.columns:
        st.warning(f"Avviso: Colonna tipo record '{colonna_tipo_record}' specificata ma non trovata. Il filtro per tipo di record verrà saltato.")
        colonna_tipo_record = None

    st.subheader("Log di Pre-processing:")
    log_messages = []

    log_messages.append("Inizio pre-processing DataFrame...")
    righe_ini = len(df)
    log_messages.append(f"Numero di righe iniziali nel DataFrame: {righe_ini}")

    df[col_source] = df[col_source].astype(str).str.strip()
    df[col_dest] = df[col_dest].astype(str).str.strip()
    if col_anchor: 
        df[col_anchor] = df[col_anchor].astype(str).str.strip().replace('', np.nan)

    if colonna_tipo_record and valore_tipo_da_includere and valore_tipo_da_includere != "Nessuno":
        log_messages.append(f"Applicazione filtro tipo record: mantenimento righe se '{colonna_tipo_record}' è '{valore_tipo_da_includere}' (case-insensitive).")
        if colonna_tipo_record in df.columns:
            df[colonna_tipo_record] = df[colonna_tipo_record].astype(str).str.strip()
            df = df[df[colonna_tipo_record].str.lower() == valore_tipo_da_includere.lower()]
            log_messages.append(f"  Righe dopo filtro tipo record: {len(df)}")
    righe_ini_post_tipo = len(df)

    df.dropna(subset=[col_source, col_dest], inplace=True)
    df = df[(df[col_source].str.len() > 0) & (df[col_dest].str.len() > 0)]
    log_messages.append(f"Righe dopo rimozione NA/vuoti in Source/Destination: {len(df)} ({righe_ini_post_tipo - len(df)} rimosse)")

    str_exclude_lower = [s.lower() for s in stringhe_url_da_escludere_selezionate if s] if stringhe_url_da_escludere_selezionate else []
    dom_include_lower = dominio_da_includere.lower() if dominio_da_includere else None

    if str_exclude_lower:
        log_messages.append(f"Filtro esclusione URL (Source/Dest): {', '.join(str_exclude_lower)}")
        righe_pre_filt_url = len(df)
        m_src = pd.Series([True]*len(df),index=df.index); m_dst = pd.Series([True]*len(df),index=df.index)
        for s_ex in str_exclude_lower:
            m_src &= ~df[col_source].str.lower().str.contains(s_ex, na=False, regex=False)
            m_dst &= ~df[col_dest].str.lower().str.contains(s_ex, na=False, regex=False)
        df = df[m_src & m_dst]
        log_messages.append(f"  Righe dopo filtro esclusione URL: {len(df)} ({righe_pre_filt_url - len(df)} rimosse)")

    if dom_include_lower:
        righe_pre_filt_dom = len(df)
        log_messages.append(f"Filtro inclusione dominio (Dest): '{dom_include_lower}'")
        df = df[df[col_dest].str.lower().str.contains(dom_include_lower, na=False, regex=False)]
        log_messages.append(f"  Righe dopo filtro inclusione dominio: {len(df)} ({righe_pre_filt_dom - len(df)} rimosse)")
        
    righe_pre_filt_auto = len(df)
    df = df[df[col_source] != df[col_dest]]
    log_messages.append(f"Righe dopo rimozione auto-link: {len(df)} ({righe_pre_filt_auto - len(df)} rimosse)")
    
    if df.empty: 
        log_messages.append("DataFrame vuoto post-filtri. Nessun grafo da generare."); 
        st.text_area("Log Pre-processing", "\n".join(log_messages), height=200)
        st.warning("Nessun dato da visualizzare dopo i filtri.")
        return None
        
    log_messages.append("Aggregazione anchor text...")
    if col_anchor and col_anchor in df.columns:
        df_agg = df.groupby([col_source, col_dest]).agg(
            Unique_Anchors=(col_anchor, lambda s: sorted(list(s.dropna().astype(str).str.strip().unique())))
        ).reset_index()
    else:
        df_agg = df[[col_source, col_dest]].drop_duplicates().reset_index(drop=True)
        df_agg['Unique_Anchors'] = [[] for _ in range(len(df_agg))]

    if df_agg.empty: 
        log_messages.append("DataFrame aggregato vuoto. Nessun grafo da generare."); 
        st.text_area("Log Pre-processing", "\n".join(log_messages), height=200)
        st.warning("Nessun dato aggregato da visualizzare.")
        return None
    log_messages.append(f"Archi unici per grafo: {len(df_agg)}")

    G = nx.DiGraph()
    for _, riga in df_agg.iterrows():
        G.add_edge(riga[col_source], riga[col_dest], anchors=riga['Unique_Anchors'])

    log_messages.append(f"Nodi nel grafo: {G.number_of_nodes()}, Archi nel grafo: {G.number_of_edges()}")
    if G.number_of_nodes() == 0: 
        log_messages.append("Grafo vuoto."); 
        st.text_area("Log Pre-processing", "\n".join(log_messages), height=200)
        st.warning("Il grafo risultante è vuoto.")
        return None

    try:
        pd.DataFrame(
            [{'Nodo_URL': n, 'OutDegree': G.out_degree(n), 'InDegree': G.in_degree(n)} for n in G.nodes()]
        ).to_csv(nome_file_export_csv, index=False, encoding='utf-8-sig')
        log_messages.append(f"Dati nodi esportati in '{nome_file_export_csv}'. (Salvato sul server)")
    except Exception as e: 
        log_messages.append(f"Errore esportazione CSV: {e}")

    st.text_area("Log Pre-processing", "\n".join(log_messages), height=200)

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
                st.error("Errore nel calcolo del layout del grafo.")
                return None
    
    if pos is None: 
        log_messages.append("Layout non calcolato. Visualizzazione annullata."); 
        st.error("Impossibile calcolare il layout del grafo.")
        return None

    log_messages.append("Preparazione visualizzazione Plotly...")
    
    edge_x, edge_y, edge_z, edge_hover_texts = [], [], [], []
    for u, v, data in G.edges(data=True):
        if u in pos and v in pos:
            pos_u, pos_v = pos[u], pos[v]
            anchors = data.get('anchors', [])
            hover_text_content = "N/A"
            if anchors:
                display_anchors = anchors[:7]
                hover_text_content = "<br>".join(f"- {str(a)}" for a in display_anchors)
                if len(anchors) > 7: hover_text_content += f"<br>... e altri {len(anchors) - 7} anchor(s)"
            full_hover_text = f"<b>Link</b><br>Da: {u}<br>A: {v}<br>--- Anchor Texts ---<br>{hover_text_content}"
            edge_x.extend([pos_u[0], pos_v[0], None]); edge_y.extend([pos_u[1], pos_v[1], None])
            if layout_3d: edge_z.extend([pos_u[2], pos_v[2], None])
            edge_hover_texts.extend([full_hover_text, full_hover_text, None])

    trace_edges_args = dict(line=dict(width=0.7, color='#888'), mode='lines', hoverinfo='text', text=edge_hover_texts, opacity=0.7)
    if layout_3d: trace_edges = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, name='Archi', **trace_edges_args)
    else: trace_edges = go.Scatter(x=edge_x, y=edge_y, name='Archi', **trace_edges_args)

    node_degrees = dict(G.degree())
    node_values = list(node_degrees.values()) if node_degrees else []
    node_traces_list = []
    node_colors = {'Basso Grado': 'blue', 'Medio Grado': 'orange', 'Alto Grado': 'red'}
    
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
            for node_id in nodes_in_cat:
                if node_id not in pos: continue
                cat_x.append(pos[node_id][0]); cat_y.append(pos[node_id][1])
                if layout_3d: cat_z.append(pos[node_id][2])
                in_degree = G.in_degree(node_id); out_degree = G.out_degree(node_id)
                cat_text.append(f"URL: {node_id}<br>Grado Totale: {node_degrees[node_id]}<br>Link Entranti: {in_degree}<br>Link Uscenti: {out_degree}")
                size_val = node_degrees.get(node_id, 0)
                cat_size.append(max(5, min(15, size_val * 1.5 if size_val > 0 else 5)))
                cat_customdata.append(node_id) 

            marker_style = dict(color=node_colors[cat_name], size=cat_size, opacity=0.9, line=dict(width=0.5, color='#333'))
            scatter_args = dict(name=cat_name, mode='markers', hoverinfo='text', text=cat_text, marker=marker_style, customdata=cat_customdata) 
            if layout_3d: node_traces_list.append(go.Scatter3d(x=cat_x, y=cat_y, z=cat_z, **scatter_args))
            else: node_traces_list.append(go.Scatter(x=cat_x, y=cat_y, **scatter_args))
            
    layout_args = dict(title_x=0.5, titlefont_size=18, showlegend=True, legend_title_text='Categorie Nodi per Grado', hovermode='closest', margin=dict(b=40,l=5,r=5,t=60)) 
    if layout_3d: fig_layout = go.Layout(title='<b>Grafo Link Interni (3D) con Anchor Text</b>', scene=dict(bgcolor="rgba(240,240,240,0.95)"), **layout_args)
    else: fig_layout = go.Layout(title='<b>Grafo Link Interni (2D) con Anchor Text</b>', plot_bgcolor='rgba(245,245,245,1)', **layout_args)
    
    figura = go.Figure(data=[trace_edges] + node_traces_list, layout=fig_layout)
    log_messages.append("Visualizzazione grafico completata.")
    # Non mostrare il log qui, verrà gestito nell'app principale
    return figura


# --- Applicazione Streamlit ---
st.set_page_config(layout="wide", page_title="Visualizzatore Grafo Link Interni")

st.title("Visualizzatore Interattivo Grafo Link Interni")
st.markdown("""
Carica un file CSV con i link interni (colonne richieste: `Source`, `Destination`; opzionali: `Anchor`, `Type`) 
e personalizza i filtri per visualizzare la struttura del grafo.
""")

# --- Sidebar per i Controlli ---
st.sidebar.header("Opzioni di Filtro e Controllo")

# 1. Caricamento File CSV
uploaded_file = st.sidebar.file_uploader("Carica il tuo file CSV dei link", type=["csv"])

# Valori di default per i filtri (usati se nessun file è caricato o per inizializzazione)
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

# 2. Dominio da Includere
dominio_input = st.sidebar.text_input("Dominio da Includere (es. 'sitoesempio.com')", value="sushisenpai")

# 3. Tipo di Record da Includere
tipo_record_opzioni = ["Nessuno", "Hyperlink", "HTTP Redirect"] # Aggiunto "Nessuno"
tipo_record_selezionato = st.sidebar.selectbox(
    "Valore Tipo Record da Includere (opzionale, colonna 'Type')", 
    options=tipo_record_opzioni, 
    index=1 # Default a "Hyperlink"
)
valore_tipo_da_usare = None if tipo_record_selezionato == "Nessuno" else tipo_record_selezionato


# 4. Filtri Estensioni/Stringhe URL da Escludere
st.sidebar.markdown("---")
st.sidebar.subheader("Filtri URL da Escludere (deseleziona per includere):")
stringhe_escluse_selezionate = []
# Usa uno stato di sessione per mantenere lo stato delle checkbox tra i rerun
if 'checkbox_states' not in st.session_state:
    st.session_state.checkbox_states = {s: True for s in default_stringhe_da_escludere}

for s_escl in default_stringhe_da_escludere:
    # Usa la chiave univoca per ogni checkbox
    is_checked = st.sidebar.checkbox(s_escl, value=st.session_state.checkbox_states[s_escl], key=f"cb_{s_escl}")
    st.session_state.checkbox_states[s_escl] = is_checked # Aggiorna lo stato
    if not is_checked: # Se NON è selezionato, significa che NON vogliamo escluderlo
        pass # Non lo aggiungiamo alla lista di esclusione
    else: # Se è selezionato, lo aggiungiamo alla lista di esclusione
        stringhe_escluse_selezionate.append(s_escl)


# --- Area Principale per il Grafico ---
if uploaded_file is not None:
    try:
        # Leggi il file CSV caricato in un DataFrame
        # Specifica dtype=str per evitare conversioni automatiche che potrebbero dare problemi
        df_caricato = pd.read_csv(uploaded_file, dtype=str)
        st.success(f"File '{uploaded_file.name}' caricato con successo. ({len(df_caricato)} righe)")

        # Genera e visualizza il grafico
        with st.spinner("Generazione del grafo in corso... Questo potrebbe richiedere alcuni istanti."):
            figura_plotly = crea_grafo_link_interni_streamlit(
                df_input=df_caricato,
                stringhe_url_da_escludere_selezionate=stringhe_escluse_selezionate,
                dominio_da_includere=dominio_input if dominio_input else None, # Passa None se vuoto
                nome_file_export_csv="report_nodi_grafo_streamlit.csv", # Nome file per l'export
                colonna_anchor_text="Anchor", # Assicurati che il tuo CSV abbia questa colonna
                colonna_tipo_record="Type",   # Assicurati che il tuo CSV abbia questa colonna
                valore_tipo_da_includere=valore_tipo_da_usare
            )
        
        if figura_plotly:
            st.plotly_chart(figura_plotly, use_container_width=True, height=800) # Aumentata altezza
            st.caption("Puoi interagire con il grafo: zoom, pan, rotazione (se 3D), e hover per dettagli.")
            st.info("Un file CSV con il report dei nodi ('report_nodi_grafo_streamlit.csv') è stato salvato sul server.")
        else:
            st.warning("Impossibile generare il grafico con i filtri correnti o a causa di un errore.")

    except Exception as e:
        st.error(f"Errore durante l'elaborazione del file CSV: {e}")
        st.exception(e) # Mostra il traceback completo per il debug
else:
    st.info("Attendo il caricamento di un file CSV per visualizzare il grafo.")

st.sidebar.markdown("---")
st.sidebar.info("Questa è un'applicazione Streamlit per visualizzare grafi di link interni.")

