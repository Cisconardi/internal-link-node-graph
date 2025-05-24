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
    abilita_colorazione_status_arco=False, 
    colonna_status_code_nome="Status Code", 
    map_status_code_a_colore=None # Nuovo: dizionario {status_code_str: colore_hex}
):
    """
    Genera una rappresentazione a grafo navigabile dei link interni da un DataFrame,
    restituisce la figura Plotly e salva un report CSV.
    """
    df = df_input.copy() 
    log_messages = []

    if df.empty: 
        return None, ["DataFrame di input vuoto."]

    col_source, col_dest = 'Source', 'Destination'
    col_anchor = colonna_anchor_text
    col_status = colonna_status_code_nome
    
    # ... (Controlli colonne e pre-processing iniziale identici alla versione precedente) ...
    cols_base_req = [col_source, col_dest]
    for col in cols_base_req:
        if col not in df.columns:
            msg = f"Errore: Colonna base '{col}' mancante nel file CSV."
            return None, [msg]
            
    if col_anchor and col_anchor not in df.columns:
        log_messages.append(f"Avviso: Colonna anchor '{col_anchor}' specificata ma non trovata.")
        col_anchor = None 
    if colonna_tipo_record and colonna_tipo_record not in df.columns:
        log_messages.append(f"Avviso: Colonna tipo record '{colonna_tipo_record}' specificata ma non trovata.")
        colonna_tipo_record = None
    if abilita_colorazione_status_arco and col_status and col_status not in df.columns:
        log_messages.append(f"Avviso: Colorazione archi per status abilitata, ma colonna '{col_status}' non trovata.")
        abilita_colorazione_status_arco = False

    log_messages.append("Inizio pre-processing DataFrame...")
    righe_iniziali_totali = len(df)
    log_messages.append(f"Numero di righe iniziali nel DataFrame: {righe_iniziali_totali}")

    df[col_source] = df[col_source].astype(str).str.strip()
    df[col_dest] = df[col_dest].astype(str).str.strip()
    if col_anchor and col_anchor in df.columns: 
        df[col_anchor] = df[col_anchor].astype(str).str.strip().replace('', np.nan)
    if col_status and col_status in df.columns:
        df[col_status] = df[col_status].astype(str).str.strip().replace('', np.nan)

    righe_prima_del_filtro_corrente = len(df)
    if colonna_tipo_record and valore_tipo_da_includere and valore_tipo_da_includere != "Nessuno":
        log_messages.append(f"Filtro tipo record: '{colonna_tipo_record}' == '{valore_tipo_da_includere}'")
        if colonna_tipo_record in df.columns:
            df[colonna_tipo_record] = df[colonna_tipo_record].astype(str).str.strip()
            df = df[df[colonna_tipo_record].str.lower() == valore_tipo_da_includere.lower()]
            log_messages.append(f"  Righe dopo filtro tipo: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    df.dropna(subset=[col_source, col_dest], inplace=True)
    df = df[(df[col_source].str.len() > 0) & (df[col_dest].str.len() > 0)]
    log_messages.append(f"Righe dopo rimozione NA/vuoti Source/Dest: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    str_exclude_lower = [s.lower().strip() for s in stringhe_url_da_escludere_selezionate if s and s.strip()] if stringhe_url_da_escludere_selezionate else []
    dom_include_lower = dominio_da_includere.lower().strip() if dominio_da_includere and dominio_da_includere.strip() else None

    if str_exclude_lower:
        log_messages.append(f"Filtro esclusione URL: {', '.join(str_exclude_lower)}")
        m_src = pd.Series([True]*len(df),index=df.index); m_dst = pd.Series([True]*len(df),index=df.index)
        for s_ex in str_exclude_lower:
            if s_ex: 
                m_src &= ~df[col_source].str.lower().str.contains(s_ex, na=False, regex=False)
                m_dst &= ~df[col_dest].str.lower().str.contains(s_ex, na=False, regex=False)
        df = df[m_src & m_dst] 
        log_messages.append(f"  Righe dopo filtro esclusione URL: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)

    if dom_include_lower:
        log_messages.append(f"Filtro inclusione dominio: '{dom_include_lower}'")
        df = df[df[col_dest].str.lower().str.contains(dom_include_lower, na=False, regex=False)]
        log_messages.append(f"  Righe dopo filtro inclusione dominio: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    righe_prima_del_filtro_corrente = len(df)
        
    df = df[df[col_source] != df[col_dest]]
    log_messages.append(f"Righe dopo rimozione auto-link: {len(df)} ({righe_prima_del_filtro_corrente - len(df)} rimosse)")
    
    if df.empty: 
        log_messages.append("DataFrame vuoto post-filtri."); return None, log_messages
        
    log_messages.append("Aggregazione anchor text e status code...")
    
    def aggregate_anchors_custom(series):
        valid_anchors = series.dropna().astype(str).str.strip().unique()
        cleaned_anchors = sorted([anchor for anchor in valid_anchors if anchor]) 
        if not cleaned_anchors: return ["Vuoto"] 
        return cleaned_anchors

    def get_representative_status_code(series):
        valid_statuses = [str(s).strip() for s in series.dropna() if str(s).strip()]
        if not valid_statuses: return "Sconosciuto"
        count = Counter(valid_statuses); most_common = count.most_common(1)
        return most_common[0][0]

    agg_dict = {}
    if col_anchor and col_anchor in df.columns:
        agg_dict['Unique_Anchors'] = (col_anchor, aggregate_anchors_custom)
    if abilita_colorazione_status_arco and col_status and col_status in df.columns:
         agg_dict['Rep_Status_Code'] = (col_status, get_representative_status_code)

    if agg_dict:
        df_agg = df.groupby([col_source, col_dest]).agg(**agg_dict).reset_index()
    else: # Nessuna colonna opzionale da aggregare
        df_agg = df[[col_source, col_dest]].drop_duplicates().reset_index(drop=True)

    # Assicura che le colonne esistano anche se non aggregate
    if 'Unique_Anchors' not in df_agg.columns:
        df_agg['Unique_Anchors'] = [["Vuoto"] for _ in range(len(df_agg))]
    if 'Rep_Status_Code' not in df_agg.columns:
        df_agg['Rep_Status_Code'] = "Sconosciuto"


    if df_agg.empty: log_messages.append("DataFrame aggregato vuoto."); return None, log_messages
    log_messages.append(f"Archi unici per grafo: {len(df_agg)}")

    G = nx.DiGraph()
    for _, riga in df_agg.iterrows():
        G.add_edge(riga[col_source], riga[col_dest], 
                   anchors=riga['Unique_Anchors'], 
                   status_code=riga['Rep_Status_Code'])

    log_messages.append(f"Nodi: {G.number_of_nodes()}, Archi: {G.number_of_edges()}")
    if G.number_of_nodes() == 0: log_messages.append("Grafo vuoto."); return None, log_messages

    try:
        pd.DataFrame(
            [{'Nodo_URL': n, 'OutDegree': G.out_degree(n), 'InDegree': G.in_degree(n)} for n in G.nodes()]
        ).to_csv(nome_file_export_csv, index=False, encoding='utf-8-sig')
        log_messages.append(f"Dati nodi esportati in '{nome_file_export_csv}'.")
    except Exception as e: log_messages.append(f"Errore esportazione CSV: {e}")

    log_messages.append("Calcolo layout...")
    pos = None; layout_3d = False; node_count = G.number_of_nodes()
    # Aumentata soglia per layout 3D
    if 0 < node_count < 100000: # Soglia aumentata
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
                log_messages.append(f"Errore tutti i layout ({e_2d_k})."); return None, log_messages
    
    if pos is None: log_messages.append("Layout non calcolato."); return None, log_messages

    log_messages.append("Preparazione visualizzazione Plotly...")
    
    # Preparazione dati per tracce archi
    traces = [] # Lista per tutte le tracce (archi e nodi)
    stringa_anchor_lower = stringa_anchor_da_evidenziare.lower().strip() if abilita_evidenziazione_anchor_arco and stringa_anchor_da_evidenziare else ""
    
    edges_data_groups = {
        'anchor_highlight': {'x':[], 'y':[], 'z':[], 'text':[], 'color': colore_evidenziazione_anchor_arco, 'name': 'Archi Evidenziati (Anchor)'}
    }
    if abilita_colorazione_status_arco and map_status_code_a_colore:
        for status_val, color_val in map_status_code_a_colore.items():
            edges_data_groups[f'status_{status_val}'] = {'x':[], 'y':[], 'z':[], 'text':[], 'color': color_val, 'name': f'Archi Status {status_val}'}
    
    edges_data_groups['default'] = {'x':[], 'y':[], 'z':[], 'text':[], 'color': '#888888', 'name': 'Archi (Altri)'}


    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos: continue
        pos_u, pos_v = pos[u], pos[v]
        anchors = data.get('anchors', ["Vuoto"])
        status_code_str = str(data.get('status_code', "Sconosciuto")).strip()

        hover_text_content = "N/A"
        if anchors:
            display_anchors = anchors[:7]
            hover_text_content = "<br>".join(f"- {str(a)}" for a in display_anchors)
            if len(anchors) > 7: hover_text_content += f"<br>... e altri {len(anchors) - 7}"
            elif not any(a for a in anchors if a != "Vuoto") and "Vuoto" in anchors : hover_text_content = "- Vuoto"
        full_hover_text = f"<b>Link</b><br>Da: {u}<br>A: {v}<br>Status: {status_code_str}<br>--- Anchor Texts ---<br>{hover_text_content}"

        target_group_key = 'default'
        is_anchor_highlighted = False

        if abilita_evidenziazione_anchor_arco and stringa_anchor_lower:
            for anchor in anchors:
                if stringa_anchor_lower in str(anchor).lower():
                    target_group_key = 'anchor_highlight'
                    is_anchor_highlighted = True
                    break
        
        if not is_anchor_highlighted and abilita_colorazione_status_arco and map_status_code_a_colore:
            if status_code_str in map_status_code_a_colore: # Verifica se lo status code esatto è nella mappa dei colori personalizzati
                target_group_key = f'status_{status_code_str}'
            # Se non c'è una corrispondenza esatta, non assegnare a un gruppo di status code specifico a meno che non sia 'Sconosciuto' e sia personalizzato
            elif 'Sconosciuto' in map_status_code_a_colore and status_code_str == "Sconosciuto":
                 target_group_key = 'status_Sconosciuto'
            # Altrimenti, rimarrà 'default'

        group = edges_data_groups.get(target_group_key)
        if not group: 
            group = edges_data_groups['default']
            
        group['x'].extend([pos_u[0], pos_v[0], None])
        group['y'].extend([pos_u[1], pos_v[1], None])
        if layout_3d: group['z'].extend([pos_u[2], pos_v[2], None])
        group['text'].extend([full_hover_text, full_hover_text, None])

    for key, group_data in edges_data_groups.items():
        if group_data['x']: 
            line_width = 1.5 if key == 'anchor_highlight' else 0.9 
            opacity_val = 0.9 if key == 'anchor_highlight' else 0.7
            trace_args = dict(line=dict(width=line_width, color=group_data['color']), 
                              mode='lines', hoverinfo='text', text=group_data['text'], 
                              opacity=opacity_val, name=group_data['name'])
            if layout_3d: traces.append(go.Scatter3d(x=group_data['x'], y=group_data['y'], z=group_data['z'], **trace_args))
            else: traces.append(go.Scatter(x=group_data['x'], y=group_data['y'], **trace_args))

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

    title_text = '<b>Grafo Link Interni (3D)</b>' if layout_3d else '<b>Grafo Link Interni (2D)</b>'

    if layout_3d: 
        fig_layout = go.Layout(
            title=dict(text=title_text, font=title_font_properties, **title_alignment_properties),
            scene=dict(bgcolor="rgba(240,240,240,0.95)"), **common_layout_properties
        )
    else: 
        fig_layout = go.Layout(
            title=dict(text=title_text, font=title_font_properties, **title_alignment_properties), 
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
    * Colora gli **archi** in base al loro Status Code, selezionando quali status code personalizzare e scegliendo i colori.
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

    # Prepara la lista di status code unici DOPO il caricamento del file
    unique_status_codes_from_file = []
    # df_temp_for_status_codes = None # Non necessario
    if uploaded_file is not None:
        try:
            # Leggi solo la colonna degli status code per efficienza, se esiste
            # Dobbiamo leggere l'intero file per passarlo alla funzione principale comunque
            # Per ottenere gli status code, è meglio leggere l'intero file una volta e passarlo
            # o leggere solo la colonna status code qui. Per ora, leggiamo solo la colonna.
            # Questo significa che uploaded_file verrà letto due volte se questa sezione è prima del processamento principale.
            # Potrebbe essere ottimizzato passandolo o leggendolo una sola volta.
            
            # Per evitare di leggere due volte il file caricato, potremmo spostare questa logica
            # all'interno del blocco if uploaded_file is not None: principale,
            # ma per ora lo lasciamo qui per semplicità della UI.
            # L'utente potrebbe dover ricaricare o l'app potrebbe rieseguire per vedere gli status code aggiornati.
            
            # Soluzione temporanea per leggere gli status codes:
            # Crea una copia del buffer per non consumarlo
            file_buffer_copy = io.BytesIO(uploaded_file.getvalue())
            temp_df_status = pd.read_csv(file_buffer_copy, dtype=str) # Leggi tutto per ora
            
            status_col_name_to_check = st.session_state.get('status_code_column_name_input', 'Status Code') # Usa lo stato della sessione se disponibile
            if status_col_name_to_check in temp_df_status.columns:
                unique_status_codes_from_file = sorted(list(temp_df_status[status_col_name_to_check].dropna().astype(str).str.strip().unique()))
            else:
                # Stampa un avviso se la colonna non è trovata, ma solo se l'utente ha specificato un nome diverso da quello di default
                # O se il nome di default non è presente.
                if status_col_name_to_check != 'Status Code' or 'Status Code' not in temp_df_status.columns:
                     st.sidebar.warning(f"Colonna Status Code '{status_col_name_to_check}' non trovata nel file caricato.")

        except Exception as e:
            st.sidebar.warning(f"Impossibile leggere gli status code dal file: {e}")
            unique_status_codes_from_file = [] 
        # Non è necessario fare uploaded_file.seek(0) qui perché lo faremo prima della lettura principale


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
    colonna_status_code_input = st.text_input("Nome colonna Status Code nel CSV:", value="Status Code", key="status_code_column_name_input")
    st.session_state.status_code_column_name_input = colonna_status_code_input # Salva per rilettura status codes
    
    map_status_a_colore_ui = {}
    default_colors_for_status_map = { # Mappa più dettagliata per default
        '200': '#2ca02c', # Verde successo
        '301': '#1f77b4', # Blu scuro per redirect permanente
        '302': '#aec7e8', # Azzurro per redirect temporaneo
        '307': '#aec7e8', # Azzurro per redirect temporaneo
        '308': '#1f77b4', # Blu scuro per redirect permanente (nuovo standard)
        '404': '#ff7f0e', # Arancione per non trovato
        '403': '#d62728', # Rosso scuro per proibito
        '410': '#8c564b', # Marrone per Gone
        '500': '#7f7f7f', # Grigio scuro per errore server generico
        '503': '#e377c2', # Rosa per servizio non disponibile
        # Categorie generiche come fallback
        '2xx': '#2ca02c', '3xx': '#17becf', 
        '4xx': '#d62728', '5xx': '#7f7f7f',
        'Sconosciuto': '#c7c7c7' # Grigio chiaro per sconosciuto/altro
    }

    if abilita_colorazione_status_ui:
        if uploaded_file and unique_status_codes_from_file:
            st.write("Seleziona gli Status Code da colorare e personalizza il colore:")
            # Se unique_status_codes_from_file è vuota, il multiselect non mostrerà opzioni
            status_codes_selezionati_per_colore = st.multiselect(
                "Status Code da personalizzare (seleziona dalla lista):",
                options=unique_status_codes_from_file,
                default=[sc for sc in ['200', '301', '404', '500', 'Sconosciuto'] if sc in unique_status_codes_from_file], # Preseleziona alcuni comuni se presenti
                key="status_codes_to_color_multiselect"
            )
            for sc_val in status_codes_selezionati_per_colore:
                # Trova un colore di default sensato
                default_color = default_colors_for_status_map.get(sc_val)
                if not default_color: # Fallback se lo status code esatto non è nella mappa
                    if sc_val.startswith('2'): default_color = default_colors_for_status_map['2xx']
                    elif sc_val.startswith('3'): default_color = default_colors_for_status_map['3xx']
                    elif sc_val.startswith('4'): default_color = default_colors_for_status_map['4xx']
                    elif sc_val.startswith('5'): default_color = default_colors_for_status_map['5xx']
                    else: default_color = default_colors_for_status_map['Sconosciuto']
                
                map_status_a_colore_ui[sc_val] = st.color_picker(f"Colore per Status '{sc_val}':", value=default_color, key=f"color_sc_{sc_val.replace('.', '_')}")
        elif uploaded_file and not unique_status_codes_from_file:
            st.warning(f"Nessun Status Code univoco trovato nella colonna '{colonna_status_code_input}' del file caricato, o la colonna è mancante/vuota.")
        elif not uploaded_file:
             st.info("Carica un file CSV per vedere e personalizzare i colori per Status Code.")


    st.markdown("---")
    st.info("ℹ️ Modifica i filtri e il grafico si aggiornerà automaticamente al caricamento di un nuovo file o al cambio di un'opzione (se il file è già caricato).") 


if uploaded_file is not None:
    try:
        # Assicurati che il puntatore del file sia all'inizio per la lettura principale
        uploaded_file.seek(0)
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
                abilita_colorazione_status_arco=abilita_colorazione_status_ui, 
                colonna_status_code_nome=colonna_status_code_input,         
                map_status_code_a_colore=map_status_a_colore_ui # Passa la mappa dei colori personalizzati                 
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
                st.warning("⚠️ File report nodi ('report_nodi_grafo_streamlit.csv') non trovato.") 
            except Exception as e_dl:
                 st.warning(f"😥 Errore nel preparare il download: {e_dl}") 
        else:
            if not log_output or "Grafo vuoto" not in "".join(log_output) and "DataFrame vuoto" not in "".join(log_output) : 
                 st.warning("🚫 Impossibile generare il grafico. Controlla i log per dettagli.") 

    except Exception as e:
        st.error(f"🆘 Errore critico: {e}") 
        st.exception(e) 
else:
    with log_placeholder_container.container(): 
        st.info("⏳ Attendo il caricamento di un file CSV.") 

st.markdown("---") 
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Made with ❤️ by <a href="https://www.linkedin.com/in/francisco-nardi-212b338b/" target="_blank">Francisco Nardi</a>
    </div>
    """,
    unsafe_allow_html=True
)
