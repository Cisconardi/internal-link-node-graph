import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import io # Per gestire l'upload del file
from collections import Counter # Per trovare lo status code più frequente

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
    abilita_evidenziazione_anchor_arco=False, 
    stringa_anchor_da_evidenziare="",        
    colore_evidenziazione_anchor_arco="#800080",
    abilita_colorazione_status_arco=False, # Nuovo
    colonna_status_code_nome="Status Code", # Nuovo
    colori_status_code=None # Nuovo: dizionario con i colori per status
):
    """
    Genera una rappresentazione a grafo navigabile dei link interni da un DataFrame,
    restituisce la figura Plotly e salva un report CSV.
    Aggiunta funzionalità per evidenziare nodi e archi per URL, anchor e status code.
    """
    df = df_input.copy() 
    log_messages = []

    if df.empty: 
        return None, ["DataFrame di input vuoto."]

    col_source, col_dest = 'Source', 'Destination'
    col_anchor = colonna_anchor_text
    col_status = colonna_status_code_nome
    
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
        
    if abilita_colorazione_status_arco and col_status and col_status not in df.columns:
        log_messages.append(f"Avviso: Colorazione archi per status code abilitata, ma colonna '{col_status}' non trovata. La colorazione per status verrà saltata.")
        abilita_colorazione_status_arco = False # Disabilita se la colonna non c'è

    log_messages.append("Inizio pre-processing DataFrame...")
    righe_iniziali_totali = len(df)
    log_messages.append(f"Numero di righe iniziali nel DataFrame: {righe_iniziali_totali}")

    df[col_source] = df[col_source].astype(str).str.strip()
    df[col_dest] = df[col_dest].astype(str).str.strip()
    if col_anchor and col_anchor in df.columns: 
        df[col_anchor] = df[col_anchor].astype(str).str.strip().replace('', np.nan)
    if col_status and col_status in df.columns: # Pulisci anche la colonna status code
        df[col_status] = df[col_status].astype(str).str.strip().replace('', np.nan)


    righe_prima_del_filtro_corrente = len(df)
    if colonna_tipo_record and valore_tipo_da_includere and valore_tipo_da_includere != "Nessuno":
        log_messages.append(f"Applicazione filtro tipo record: mantenimento righe se '{colonna_tipo_record}' è '{valore_tipo_da_includere}' (case-insensitive).")
        if colonna_tipo_record in df.columns:
            df[colonna_tipo_record] = df[colonna_tipo_record].astype(str).str.strip()
            df = df[df[colonna_tipo_record].str.lower() == valore_tipo_da_includere.lower()]
            log_messages.append(f"  Righe dopo filtro tipo record: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    df.dropna(subset=[col_source, col_dest], inplace=True) # Rimuovi NaN solo per source e dest obbligatori
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
        
    log_messages.append("Aggregazione anchor text e status code...")
    
    def aggregate_anchors_custom(series):
        valid_anchors = series.dropna().astype(str).str.strip().unique()
        cleaned_anchors = sorted([anchor for anchor in valid_anchors if anchor]) 
        if not cleaned_anchors: return ["Vuoto"] 
        return cleaned_anchors

    def get_representative_status_code(series):
        # Rimuovi NaN, converti in stringa, togli spazi, filtra stringhe vuote
        valid_statuses = [str(s).strip() for s in series.dropna() if str(s).strip()]
        if not valid_statuses:
            return "Sconosciuto"
        # Trova il più frequente
        count = Counter(valid_statuses)
        most_common = count.most_common(1)
        return most_common[0][0]

    agg_functions = {}
    if col_anchor and col_anchor in df.columns:
        agg_functions['Unique_Anchors'] = (col_anchor, aggregate_anchors_custom)
    
    # Solo se la colonna status esiste e la colorazione è abilitata
    if abilita_colorazione_status_arco and col_status and col_status in df.columns:
        agg_functions['Rep_Status_Code'] = (col_status, get_representative_status_code)

    if not agg_functions: # Se non ci sono colonne opzionali da aggregare
        df_agg = df[[col_source, col_dest]].drop_duplicates().reset_index(drop=True)
        if col_anchor: df_agg['Unique_Anchors'] = [["Vuoto"] for _ in range(len(df_agg))]
        if abilita_colorazione_status_arco and col_status: df_agg['Rep_Status_Code'] = "Sconosciuto"
    else:
        df_agg = df.groupby([col_source, col_dest]).agg(**agg_functions).reset_index()
        if 'Unique_Anchors' not in df_agg.columns and col_anchor: # Assicura che la colonna esista
             df_agg['Unique_Anchors'] = [["Vuoto"] for _ in range(len(df_agg))]
        if 'Rep_Status_Code' not in df_agg.columns and abilita_colorazione_status_arco and col_status:
             df_agg['Rep_Status_Code'] = "Sconosciuto"


    if df_agg.empty: 
        log_messages.append("DataFrame aggregato vuoto. Nessun grafo da generare."); 
        return None, log_messages
    log_messages.append(f"Archi unici per grafo: {len(df_agg)}")

    G = nx.DiGraph()
    for _, riga in df_agg.iterrows():
        anchors_data = riga.get('Unique_Anchors', ["Vuoto"])
        status_data = riga.get('Rep_Status_Code', "Sconosciuto") if abilita_colorazione_status_arco else "Sconosciuto"
        G.add_edge(riga[col_source], riga[col_dest], anchors=anchors_data, status_code=status_data)


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
    
    # Preparazione dati per tracce archi
    stringa_anchor_lower = stringa_anchor_da_evidenziare.lower().strip() if abilita_evidenziazione_anchor_arco and stringa_anchor_da_evidenziare else ""
    
    # Dizionari per raggruppare gli archi per colore/traccia
    edges_by_color_group = {
        'anchor_highlight': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colore_evidenziazione_anchor_arco, 'name': 'Archi Evidenziati (Anchor)'},
        'status_2xx': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('2xx', '#2ca02c'), 'name': 'Archi 2xx (Successo)'},
        'status_301': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('301', '#1f77b4'), 'name': 'Archi 301 (Redirect Perm.)'},
        'status_30x': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('30x', '#aec7e8'), 'name': 'Archi 30x (Altri Redirect)'},
        'status_404': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('404', '#ff7f0e'), 'name': 'Archi 404 (Non Trovato)'},
        'status_4xx': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('4xx', '#d62728'), 'name': 'Archi 4xx (Altri Errori Client)'},
        'status_5xx': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('5xx', '#7f7f7f'), 'name': 'Archi 5xx (Errori Server)'},
        'status_unknown': {'x': [], 'y': [], 'z': [], 'text': [], 'color': colori_status_code.get('Sconosciuto', '#c7c7c7'), 'name': 'Archi (Status Sconosciuto)'},
        'default': {'x': [], 'y': [], 'z': [], 'text': [], 'color': '#888888', 'name': 'Archi (Default)'} # Fallback se la colorazione status non è attiva
    }

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        pos_u, pos_v = pos[u], pos[v]
        anchors = data.get('anchors', ["Vuoto"])
        status_code_str = str(data.get('status_code', "Sconosciuto")).strip()

        hover_text_content = "N/A"
        if anchors:
            display_anchors = anchors[:7]
            hover_text_content = "<br>".join(f"- {str(a)}" for a in display_anchors)
            if len(anchors) > 7: hover_text_content += f"<br>... e altri {len(anchors) - 7} anchor(s)"
            elif not any(a for a in anchors if a != "Vuoto") and "Vuoto" in anchors: hover_text_content = "- Vuoto"
        full_hover_text = f"<b>Link</b><br>Da: {u}<br>A: {v}<br>Status: {status_code_str}<br>--- Anchor Texts ---<br>{hover_text_content}"

        edge_group_key = 'default' # Default se nessun'altra condizione matcha
        is_anchor_highlighted = False

        if abilita_evidenziazione_anchor_arco and stringa_anchor_lower:
            for anchor in anchors:
                if stringa_anchor_lower in str(anchor).lower():
                    edge_group_key = 'anchor_highlight'
                    is_anchor_highlighted = True
                    break
        
        if not is_anchor_highlighted and abilita_colorazione_status_arco:
            if status_code_str.startswith('2'): edge_group_key = 'status_2xx'
            elif status_code_str == '301': edge_group_key = 'status_301'
            elif status_code_str.startswith('3'): edge_group_key = 'status_30x'
            elif status_code_str == '404': edge_group_key = 'status_404'
            elif status_code_str.startswith('4'): edge_group_key = 'status_4xx'
            elif status_code_str.startswith('5'): edge_group_key = 'status_5xx'
            else: edge_group_key = 'status_unknown'
        
        group = edges_by_color_group[edge_group_key]
        group['x'].extend([pos_u[0], pos_v[0], None])
        group['y'].extend([pos_u[1], pos_v[1], None])
        if layout_3d: group['z'].extend([pos_u[2], pos_v[2], None])
        group['text'].extend([full_hover_text, full_hover_text, None])

    traces = []
    for key, group_data in edges_by_color_group.items():
        if group_data['x']: # Solo se ci sono archi in questo gruppo
            line_width = 1.5 if key == 'anchor_highlight' else 0.7
            opacity_val = 0.9 if key == 'anchor_highlight' else 0.7
            trace_args = dict(line=dict(width=line_width, color=group_data['color']), 
                              mode='lines', hoverinfo='text', text=group_data['text'], 
                              opacity=opacity_val, name=group_data['name'])
            if layout_3d: traces.append(go.Scatter3d(x=group_data['x'], y=group_data['y'], z=group_data['z'], **trace_args))
            else: traces.append(go.Scatter(x=group_data['x'], y=group_data['y'], **trace_args))


    # Nodi (logica di colorazione URL e per categoria)
    node_degrees = dict(G.degree())
    node_values = list(node_degrees.values()) if node_degrees else []
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
        legend_title_text='Legenda', 
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
    
    figura = go.Figure(data=traces, layout=fig_layout) 
    log_messages.append("Visualizzazione grafico completata.")
    return figura, log_messages


# --- Applicazione Streamlit ---
st.set_page_config(layout="wide", page_title="Visualizzatore Grafo Link Interni")

st.title("🌐 Visualizzatore Interattivo Grafo Link Interni 🔗")

st.markdown("""
Benvenuto! Questa applicazione ti aiuta a esplorare la struttura dei link interni del tuo sito web. 
Visualizza le connessioni tra le pagine come un grafo interattivo, permettendoti di identificare 
pattern, pagine isolate, e molto altro.

**Come funziona?** 🗺️

1.  **⬆️ Carica il tuo File**: Utilizza la sidebar per caricare un file CSV contenente i dati dei link.
    * **Colonne Richieste**: `Source` (URL di origine), `Destination` (URL di destinazione).
    * **Colonne Opzionali**: 
        * `Anchor` (il testo dell'anchor text del link).
        * `Type` (il tipo di link, es. "Hyperlink", "HTTP Redirect").
        * `Status Code` (il codice di stato HTTP del link di destinazione).
2.  **⚙️ Personalizza i Filtri**: Nella sidebar, puoi:
    * Definire un **dominio specifico** da includere (per concentrarti sui link interni).
    * Filtrare per **tipo di record** (es. visualizzare solo "Hyperlink").
    * **Escludere URL** che contengono stringhe specifiche (es. `.css`, `.jpg`) tramite checkbox o inserendo valori personalizzati.
3.  **🎨 Evidenziazione Personalizzata**:
    * Colora i **nodi** (pagine) la cui URL contiene una stringa a tua scelta.
    * Colora gli **archi** (link) il cui anchor text contiene una stringa a tua scelta.
    * Colora gli **archi** in base al loro Status Code.
4.  **🔎 Esplora il Grafo**: Il grafo verrà visualizzato nell'area principale.
    * Interagisci zoomando, spostandoti e ruotando (se in 3D).
    * Passa il mouse sopra nodi e archi per visualizzare dettagli.
5.  **📊 Analizza i Log**: Un log di pre-processing (espandibile) ti mostrerà come i filtri influenzano i dati.
6.  **💾 Scarica il Report**: Puoi scaricare un riepilogo dei nodi del grafo generato.

Inizia caricando il tuo file e sperimentando con i filtri!
""")


log_placeholder_container = st.empty() 

with st.sidebar: 
    st.header("🛠️ Opzioni di Filtro e Controllo") 
    uploaded_file = st.file_uploader("📤 Carica il tuo file CSV dei link", type=["csv"]) 

    default_stringhe_da_escludere = [     
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 
        '.css', '.js', '.svg', '.ico', 
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar.gz', 
        '.mp3', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.wmv', 
        '.xml', '.json', '.txt', 
        'mailto:', 'tel:', '#', 'javascript:void(0)'
    ]

    dominio_input = st.text_input("🔗 Dominio da Includere (es. 'sitoesempio.com')", value="sitoesempio.com") 

    tipo_record_opzioni = ["Nessuno", "Hyperlink", "HTTP Redirect", "HTML Canonical", "Image"] 
    tipo_record_selezionato = st.selectbox(
        "🏷️ Valore Tipo Record da Includere (opzionale, colonna 'Type')",  
        options=tipo_record_opzioni, 
        index=1 
    )
    valore_tipo_da_usare = None if tipo_record_selezionato == "Nessuno" else tipo_record_selezionato

    st.markdown("---")
    st.subheader("🚫 Filtri URL da Escludere (seleziona per escludere):") 
    
    if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = {s_escl: True for s_escl in default_stringhe_da_escludere}

    stringhe_escluse_selezionate_ui = []
    num_columns = 3
    cols = st.columns(num_columns)
    for idx, s_escl in enumerate(default_stringhe_da_escludere):
        checkbox_key = f"cb_{s_escl.replace('.', '_dot_').replace('/', '_slash_').replace(':', '_colon_').replace('#','_hash_')}"
        if checkbox_key not in st.session_state.checkbox_states:
            st.session_state.checkbox_states[checkbox_key] = True 
        display_label_cb = r"\#" if s_escl == "#" else s_escl
        with cols[idx % num_columns]: 
            is_checked = st.checkbox(
                display_label_cb, 
                value=st.session_state.checkbox_states[checkbox_key], 
                key=checkbox_key 
            )
        st.session_state.checkbox_states[checkbox_key] = is_checked 
        if is_checked: 
            stringhe_escluse_selezionate_ui.append(s_escl)
    
    custom_exclusions_input = st.text_input("➕ Altri filtri URL da escludere (separati da virgola):", key="custom_exclusions_text") 
    if custom_exclusions_input:
        custom_exclusions_list = [item.strip() for item in custom_exclusions_input.split(',') if item.strip()]
        stringhe_escluse_selezionate_ui.extend(custom_exclusions_list) 
        stringhe_escluse_selezionate_ui = sorted(list(set(stringhe_escluse_selezionate_ui)))

    st.caption(f"Stringhe URL totali per l'esclusione: {stringhe_escluse_selezionate_ui if stringhe_escluse_selezionate_ui else 'Nessuna'}")
    
    st.markdown("---")
    st.subheader("🎨 Evidenziazione URL Nodi") 
    abilita_evidenziazione_stringa_url_ui = st.checkbox("Abilita evidenziazione URL per stringa", key="enable_highlight_url_str")
    stringa_da_cercare_url_ui = ""
    colore_scelto_url_ui = "#00FF00" 
    if abilita_evidenziazione_stringa_url_ui:
        stringa_da_cercare_url_ui = st.text_input("Stringa da cercare nell'URL del nodo (case-insensitive):", key="highlight_url_str_text")
        colore_scelto_url_ui = st.color_picker("Colore di evidenziazione per URL nodo:", value="#00FF00", key="highlight_url_color")

    st.markdown("---")
    st.subheader("🌈 Evidenziazione Anchor Text Archi") 
    abilita_evidenziazione_anchor_ui = st.checkbox("Abilita evidenziazione anchor text per archi", key="enable_highlight_anchor_str")
    stringa_da_cercare_anchor_ui = ""
    colore_scelto_anchor_ui = "#800080" 
    if abilita_evidenziazione_anchor_ui:
        stringa_da_cercare_anchor_ui = st.text_input("Stringa da cercare nell'anchor text (case-insensitive):", key="highlight_anchor_str_text")
        colore_scelto_anchor_ui = st.color_picker("Colore di evidenziazione per anchor arco:", value="#800080", key="highlight_anchor_color")

    st.markdown("---")
    st.subheader("🚦 Evidenziazione Archi per Status Code")
    abilita_colorazione_status_ui = st.checkbox("Abilita colorazione archi per Status Code", value=True, key="enable_status_code_coloring")
    colonna_status_code_input = st.text_input("Nome colonna Status Code nel CSV:", value="Status Code", key="status_code_column_name")
    
    colori_status_code_ui = {}
    if abilita_colorazione_status_ui:
        colori_status_code_ui['2xx'] = st.color_picker("Colore Archi 2xx (Successo):", "#2ca02c", key="color_status_2xx")
        colori_status_code_ui['301'] = st.color_picker("Colore Archi 301 (Redirect Perm.):", "#1f77b4", key="color_status_301")
        colori_status_code_ui['30x'] = st.color_picker("Colore Archi 30x (Altri Redirect):", "#aec7e8", key="color_status_30x")
        colori_status_code_ui['404'] = st.color_picker("Colore Archi 404 (Non Trovato):", "#ff7f0e", key="color_status_404")
        colori_status_code_ui['4xx'] = st.color_picker("Colore Archi 4xx (Altri Errori Client):", "#d62728", key="color_status_4xx")
        colori_status_code_ui['5xx'] = st.color_picker("Colore Archi 5xx (Errori Server):", "#7f7f7f", key="color_status_5xx")
        colori_status_code_ui['Sconosciuto'] = st.color_picker("Colore Archi (Status Sconosciuto/Altro):", "#c7c7c7", key="color_status_unknown")


    st.markdown("---")
    st.info("ℹ️ Modifica i filtri e il grafico si aggiornerà automaticamente al caricamento di un nuovo file o al cambio di un'opzione (se il file è già caricato).") 


if uploaded_file is not None:
    try:
        df_caricato = pd.read_csv(uploaded_file, dtype=str)
        st.success(f"✔️ File '{uploaded_file.name}' caricato con successo. ({len(df_caricato)} righe)") 

        with st.spinner("⏳ Generazione del grafo in corso... Questo potrebbe richiedere alcuni istanti."): 
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
                abilita_evidenziazione_anchor_arco=abilita_evidenziazione_anchor_ui, 
                stringa_anchor_da_evidenziare=stringa_da_cercare_anchor_ui,       
                colore_evidenziazione_anchor_arco=colore_scelto_anchor_ui,
                abilita_colorazione_status_arco=abilita_colorazione_status_ui, # Passa nuovo parametro
                colonna_status_code_nome=colonna_status_code_input,         # Passa nuovo parametro
                colori_status_code=colori_status_code_ui                   # Passa nuovo parametro
            )
        
        with log_placeholder_container.expander("📜 Log di Pre-processing", expanded=False): 
            st.text("\n".join(log_output))

        if figura_plotly:
            st.plotly_chart(figura_plotly, use_container_width=True, height=800)
            st.caption("🖱️ Interagisci con il grafo: zoom, pan, rotazione (3D), hover per dettagli.") 
            try:
                with open("report_nodi_grafo_streamlit.csv", "rb") as fp:
                    st.download_button(
                        label="📥 Scarica Report Nodi (CSV)", 
                        data=fp,
                        file_name="report_nodi_grafo.csv", 
                        mime="text/csv"
                    )
            except FileNotFoundError:
                st.warning("⚠️ File report nodi ('report_nodi_grafo_streamlit.csv') non trovato. Potrebbe non essere stato ancora generato o c'è stato un errore.") 
            except Exception as e_dl:
                 st.warning(f"😥 Errore nel preparare il download del report nodi: {e_dl}") 
        else:
            if not log_output: 
                 st.warning("🚫 Impossibile generare il grafico con i filtri correnti o a causa di un errore non specificato nei log.") 


    except Exception as e:
        st.error(f"🆘 Errore critico durante l'elaborazione del file o la generazione del grafo: {e}") 
        st.exception(e) 
else:
    with log_placeholder_container.container(): 
        st.info("⏳ Attendo il caricamento di un file CSV per visualizzare il grafo e i log.") 

st.markdown("---") 
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Made with ❤️ by <a href="https://www.linkedin.com/in/francisco-nardi-212b338b/" target="_blank">Francisco Nardi</a>
    </div>
    """,
    unsafe_allow_html=True
)
