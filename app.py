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
    map_status_code_a_colore=None 
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
    else: 
        df_agg = df[[col_source, col_dest]].drop_duplicates().reset_index(drop=True)

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
    if 0 < node_count < 100000: 
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
    
    traces = [] 
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
            if status_code_str in map_status_code_a_colore: 
                target_group_key = f'status_{status_code_str}'
            elif 'Sconosciuto' in map_status_code_a_colore and status_code_str == "Sconosciuto":
                 target_group_key = 'status_Sconosciuto'

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

with st.expander("Come funziona questa App? 🗺️ (Clicca per espandere)", expanded=True):
    st.markdown("""
    Benvenuto! Questa applicazione ti aiuta a esplorare la struttura dei link interni del tuo sito web. 
    Visualizza le connessioni tra le pagine come un grafo interattivo, permettendoti di identificare 
    pattern, pagine isolate, e molto altro.

    **Passaggi per l'utilizzo:**

    1.  **⬆️ Carica il tuo File**: 
        * Utilizza il pulsante "Carica il tuo file CSV dei link" nella sidebar a sinistra.
        * Il file CSV deve contenere almeno le colonne `Source` (URL di origine del link) e `Destination` (URL di destinazione del link).
        * **Colonne Opzionali Utili**:
            * `Anchor`: Il testo cliccabile del link (anchor text).
            * `Type`: Il tipo di link (es. "Hyperlink", "HTTP Redirect", "Image").
            * `Status Code`: Il codice di stato HTTP della URL di destinazione (es. 200, 301, 404).

    2.  **⚙️ Personalizza i Filtri (Sidebar Sinistra)**:
        * **Dominio da Includere**: Inserisci il tuo dominio (es. `sitoesempio.com`) per focalizzare l'analisi sui link interni. Vengono considerate varianti come `www.`, `http://`, `https://`.
        * **Valore Tipo Record**: Se il tuo CSV ha una colonna "Type", puoi filtrare per tipi specifici come "Hyperlink" o "HTTP Redirect". Seleziona "Nessuno" per non applicare questo filtro.
        * **Filtri URL da Escludere**: 
            * Una lista di checkbox ti permette di escludere rapidamente URL che contengono stringhe comuni (es. estensioni di file immagine come `.jpg`, file di stile come `.css`, ecc.). Deseleziona una checkbox per *includere* le URL che contengono quella stringa.
            * Usa il campo "Altri filtri URL da escludere" per inserire stringhe personalizzate (separate da virgola) da escludere. Qualsiasi URL (sia Source che Destination) che contiene una di queste stringhe verrà rimossa.
    3.  **🎨 Evidenziazione Personalizzata (Sidebar Sinistra)**:
        * **URL Nodi**: Abilita e inserisci una stringa per colorare diversamente i nodi (pagine) la cui URL contiene quella stringa. Scegli il colore che preferisci.
        * **Anchor Text Archi**: Abilita e inserisci una stringa per colorare diversamente gli archi (link) se uno dei loro anchor text contiene quella stringa.
        * **Status Code Archi**: Abilita e definisci il nome della colonna "Status Code" nel tuo CSV. Poi, per ogni status code unico trovato nel tuo file, puoi attivare la colorazione e scegliere un colore personalizzato.
    4.  **🚀 Applica Configurazione**: Dopo aver impostato i filtri e le opzioni di evidenziazione, clicca il pulsante "Applica Configurazione e Genera Grafo" in fondo alla sidebar.
    5.  **🔎 Esplora il Grafo (Area Principale)**:
        * Il grafo generato apparirà sulla destra.
        * Puoi zoomare, spostare (pan) e ruotare (se il layout è 3D) il grafo usando il mouse.
        * Passa il mouse sopra i nodi per vedere l'URL, il grado totale, i link entranti e uscenti.
        * Passa il mouse sopra gli archi per vedere l'URL di origine, di destinazione, lo status code (se disponibile) e gli anchor text unici.
    6.  **📊 Analizza i Log (Sotto il Grafo)**:
        * Un'area espandibile "Log di Pre-processing" ti mostrerà quante righe sono state processate e rimosse da ciascun filtro, aiutandoti a capire come le tue selezioni influenzano il grafo finale.
    7.  **💾 Scarica il Report (Sotto il Grafo)**:
        * Un pulsante "Scarica Report Nodi (CSV)" ti permette di salvare un file CSV con un riepilogo di tutti i nodi presenti nel grafo visualizzato, inclusi i loro gradi di entrata e uscita.

    Inizia caricando il tuo file e sperimentando con i filtri per ottenere gli insight che cerchi!
    """)


log_placeholder_container = st.empty() 
if 'figura_plotly_main' not in st.session_state:
    st.session_state.figura_plotly_main = None
if 'log_output_main' not in st.session_state:
    st.session_state.log_output_main = []


with st.sidebar: 
    st.header("🛠️ Opzioni di Filtro e Controllo") 
    uploaded_file = st.file_uploader("📤 Carica il tuo file CSV dei link", type=["csv"], key="main_file_uploader") 

    if uploaded_file is not None and st.session_state.get('last_uploaded_file_id') != uploaded_file.file_id:
        st.session_state.figura_plotly_main = None
        st.session_state.log_output_main = []
        st.session_state.last_uploaded_file_id = uploaded_file.file_id
        st.session_state.config_applied = False 
    elif uploaded_file is None and st.session_state.get('last_uploaded_file_id') is not None:
        st.session_state.figura_plotly_main = None
        st.session_state.log_output_main = []
        st.session_state.last_uploaded_file_id = None
        st.session_state.config_applied = False

    unique_status_codes_from_file = []
    # Inizializza status_code_column_name_input in session_state se non esiste
    if 'status_code_column_name_input_val' not in st.session_state: 
        st.session_state.status_code_column_name_input_val = "Status Code"


    # Leggi il nome della colonna Status Code dall'input utente
    colonna_status_code_input_val_widget = st.text_input(
        "Nome colonna Status Code nel CSV:", 
        value=st.session_state.status_code_column_name_input_val, # Usa il valore da session_state
        key="status_code_column_name_widget_key" # Chiave per il widget
    )
    # Aggiorna session_state se il valore del widget cambia
    if colonna_status_code_input_val_widget != st.session_state.status_code_column_name_input_val:
        st.session_state.status_code_column_name_input_val = colonna_status_code_input_val_widget
        # Forza un rerun per aggiornare unique_status_codes_from_file se il nome colonna cambia
        # Questo è importante per popolare correttamente le checkbox degli status code.
        if uploaded_file:
            st.rerun()


    if uploaded_file is not None:
        try:
            file_buffer_copy = io.BytesIO(uploaded_file.getvalue()) 
            # Usa il valore da session_state che è stato aggiornato dal widget
            current_status_col_name = st.session_state.status_code_column_name_input_val
            
            # Prova a leggere solo la colonna specificata. Se non esiste, il dataframe sarà vuoto o darà errore.
            try:
                temp_df_status = pd.read_csv(file_buffer_copy, dtype=str, usecols=[current_status_col_name])
                if current_status_col_name in temp_df_status.columns:
                    unique_status_codes_from_file = sorted(list(temp_df_status[current_status_col_name].dropna().astype(str).str.strip().unique()))
                    if not unique_status_codes_from_file:
                        st.sidebar.caption(f"Nessun valore trovato nella colonna '{current_status_col_name}'.")
                else: # Questo caso non dovrebbe accadere con usecols, ma per sicurezza
                    st.sidebar.warning(f"Colonna Status Code '{current_status_col_name}' non trovata.")
            except ValueError: # Accade se la colonna specificata in usecols non esiste
                 st.sidebar.warning(f"Colonna Status Code '{current_status_col_name}' non trovata nel file CSV.")
            uploaded_file.seek(0) 
        except pd.errors.EmptyDataError: st.sidebar.warning("File CSV caricato è vuoto o malformattato.")
        except Exception as e: st.sidebar.error(f"Errore lettura status codes: {e}"); unique_status_codes_from_file = [] 


    default_stringhe_da_escludere = [     
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 
        '.css', '.js', '.svg', '.ico', 
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.zip', '.rar', '.tar.gz', 
        '.mp3', '.wav', '.ogg', '.mp4', '.mov', '.avi', '.wmv', 
        '.xml', '.json', '.txt', 
        'mailto:', 'tel:', '#', 'javascript:void(0)'
    ]

    dominio_input = st.text_input("🔗 Dominio da Includere (es. 'sitoesempio.com')", value="sitoesempio.com", key="dominio_input_key") 

    tipo_record_opzioni = ["Nessuno", "Hyperlink", "HTTP Redirect", "HTML Canonical", "Image"] 
    tipo_record_selezionato = st.selectbox(
        "🏷️ Valore Tipo Record da Includere (opzionale, colonna 'Type')",  
        options=tipo_record_opzioni, 
        index=1, key="tipo_record_key"
    )
    valore_tipo_da_usare = None if tipo_record_selezionato == "Nessuno" else tipo_record_selezionato

    st.markdown("---")
    st.subheader("🚫 Filtri URL da Escludere (seleziona per escludere):") 
    
    if 'checkbox_states' not in st.session_state:
        st.session_state.checkbox_states = {s_escl: True for s_escl in default_stringhe_da_escludere}

    stringhe_escluse_selezionate_ui = []
    num_columns = 3
    cols_checkbox = st.columns(num_columns) 
    for idx, s_escl in enumerate(default_stringhe_da_escludere):
        checkbox_key = f"cb_{s_escl.replace('.', '_dot_').replace('/', '_slash_').replace(':', '_colon_').replace('#','_hash_')}"
        if checkbox_key not in st.session_state.checkbox_states:
            st.session_state.checkbox_states[checkbox_key] = True 
        display_label_cb = r"\#" if s_escl == "#" else s_escl
        with cols_checkbox[idx % num_columns]: 
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
    # colonna_status_code_input è già definita sopra e legata a st.session_state.status_code_column_name_input_val
    
    map_status_a_colore_ui = {}
    default_colors_for_status_map = { 
        '200': '#2ca02c', '301': '#1f77b4', '302': '#aec7e8',
        '307': '#aec7e8', '308': '#1f77b4', '404': '#ff7f0e',
        '403': '#d62728', '410': '#8c564b', '500': '#7f7f7f',
        '503': '#e377c2', 
        '2xx': '#2ca02c', '3xx': '#17becf', 
        '4xx': '#d62728', '5xx': '#7f7f7f',
        'Sconosciuto': '#c7c7c7' 
    }

    if abilita_colorazione_status_ui:
        if uploaded_file and unique_status_codes_from_file:
            st.write("Seleziona e personalizza i colori per gli Status Code:")
            if 'status_checkbox_states' not in st.session_state: # Stato per le checkbox degli status code
                st.session_state.status_checkbox_states = {}
            
            num_status_cols = 2 
            status_cols = st.columns(num_status_cols)
            col_idx = 0

            for sc_val in unique_status_codes_from_file:
                status_checkbox_key = f"cb_status_{sc_val.replace('.', '_').replace('/','_').replace(' ','_')}" # Chiave univoca più robusta
                common_codes_to_default_check = ['200', '301', '302', '404', '500', '503']
                # Inizializza lo stato per questo specifico status code se non esiste
                if status_checkbox_key not in st.session_state.status_checkbox_states:
                    st.session_state.status_checkbox_states[status_checkbox_key] = sc_val in common_codes_to_default_check
                
                with status_cols[col_idx % num_status_cols]:
                    enable_color_for_sc = st.checkbox(
                        f"Colora Status '{sc_val}'", 
                        value=st.session_state.status_checkbox_states[status_checkbox_key],
                        key=status_checkbox_key
                    )
                st.session_state.status_checkbox_states[status_checkbox_key] = enable_color_for_sc

                if enable_color_for_sc:
                    default_color = default_colors_for_status_map.get(sc_val)
                    if not default_color: 
                        if sc_val.startswith('2'): default_color = default_colors_for_status_map['2xx']
                        elif sc_val.startswith('3'): default_color = default_colors_for_status_map['3xx']
                        elif sc_val.startswith('4'): default_color = default_colors_for_status_map['4xx']
                        elif sc_val.startswith('5'): default_color = default_colors_for_status_map['5xx']
                        else: default_color = default_colors_for_status_map['Sconosciuto']
                    
                    with status_cols[col_idx % num_status_cols]: 
                        map_status_a_colore_ui[sc_val] = st.color_picker(
                            f"Colore per '{sc_val}':", 
                            value=default_color, 
                            key=f"color_picker_sc_{sc_val.replace('.', '_').replace('/','_').replace(' ','_')}"
                        )
                col_idx +=1
        elif uploaded_file and not unique_status_codes_from_file:
            st.warning(f"Nessun Status Code univoco trovato nella colonna '{st.session_state.status_code_column_name_input_val}'.")
        elif not uploaded_file:
             st.info("Carica un file CSV per personalizzare i colori per Status Code.")

    st.markdown("---")
    apply_button = st.button("🚀 Applica Configurazione e Genera Grafo", key="apply_config_button")
    st.info("ℹ️ Modifica i filtri e clicca 'Applica' per aggiornare il grafico.") 


if 'config_applied' not in st.session_state:
    st.session_state.config_applied = False

if apply_button and uploaded_file is not None:
    st.session_state.config_applied = True 
    try:
        uploaded_file.seek(0) 
        df_caricato = pd.read_csv(uploaded_file, dtype=str)
        # Non mostrare st.success qui, verrà fatto dopo la generazione del grafo se tutto ok

        with st.spinner("⏳ Generazione del grafo..."): 
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
                colonna_status_code_nome=st.session_state.status_code_column_name_input_val, # Usa il valore da session_state       
                map_status_code_a_colore=map_status_a_colore_ui                
            )
        st.session_state.figura_plotly_main = figura_plotly
        st.session_state.log_output_main = log_output
        if figura_plotly:
             st.success(f"✔️ File '{uploaded_file.name}' elaborato e grafo generato.")
        
    except Exception as e:
        st.error(f"🆘 Errore critico durante l'elaborazione o generazione: {e}") 
        st.exception(e) 
        st.session_state.figura_plotly_main = None
        st.session_state.log_output_main = [f"Errore critico: {e}"]

elif uploaded_file is None and apply_button: # Se si clicca applica ma non c'è file
    st.session_state.figura_plotly_main = None
    st.session_state.log_output_main = []
    st.session_state.config_applied = False 
    st.warning("⚠️ Per favore, carica prima un file CSV.")


# Visualizza i log e il grafico basandosi sullo stato della sessione
if st.session_state.get('log_output_main'):
    with log_placeholder_container.expander("📜 Log di Pre-processing", expanded=st.session_state.get('config_applied', False) ): 
        st.text("\n".join(st.session_state.log_output_main))
elif uploaded_file is not None and not st.session_state.get('config_applied', False):
     with log_placeholder_container.container():
        st.info("⚙️ Configura i filtri e clicca 'Applica Configurazione e Genera Grafo' nella sidebar.")
else: 
    with log_placeholder_container.container(): 
        st.info("⏳ Attendo il caricamento di un file CSV e l'applicazione della configurazione.") 


if st.session_state.get('figura_plotly_main') is not None:
    st.plotly_chart(st.session_state.figura_plotly_main, use_container_width=True, height=800)
    st.caption("🖱️ Interagisci con il grafo.") 
    try:
        # Verifica se il file report esiste prima di offrire il download
        if pd.DataFrame([{'Test': 'test'}]).to_csv("report_nodi_grafo_streamlit.csv", index=False): # Test di scrittura per assicurarsi che la directory sia scrivibile
            # Questo test non garantisce che il file corretto sia stato scritto dalla funzione principale,
            # ma almeno verifica la capacità di scrittura.
            # Sarebbe meglio controllare l'esistenza del file dopo la sua creazione nella funzione principale.
            # Per ora, assumiamo che se figura_plotly esiste, il CSV è stato tentato.
            pass # Il file viene creato nella funzione principale. Il download button lo leggerà.

        with open("report_nodi_grafo_streamlit.csv", "rb") as fp: # Tenta di aprire il file
            st.download_button(
                label="📥 Scarica Report Nodi (CSV)", 
                data=fp,
                file_name="report_nodi_grafo.csv", 
                mime="text/csv",
                key="download_report_button"
            )
    except FileNotFoundError:
        # Questo avviso apparirà solo se il file non è stato creato affatto.
        # Se la funzione crea_grafo_link_interni_streamlit fallisce prima dell'export CSV, 
        # questo blocco potrebbe non essere raggiunto o il file potrebbe non esistere.
        if st.session_state.get('config_applied', False): # Mostra solo se si è tentato di generare
            st.warning("⚠️ File report nodi non trovato (potrebbe non essere stato generato a causa di errori precedenti).") 
    except Exception as e_dl:
         st.warning(f"😥 Errore download report: {e_dl}") 
elif uploaded_file is not None and st.session_state.get('config_applied', False): 
    if not st.session_state.get('log_output_main') or \
       ("Grafo vuoto" not in "".join(st.session_state.get('log_output_main',[])) and \
        "DataFrame vuoto" not in "".join(st.session_state.get('log_output_main',[]))):
        # Mostra questo solo se non ci sono già messaggi di errore specifici nei log
        st.warning("🚫 Impossibile generare il grafico. Controlla i log per dettagli.")


st.markdown("---") 
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Made with ❤️ by <a href="https://www.linkedin.com/in/francisco-nardi-212b338b/" target="_blank">Francisco Nardi</a>
    </div>
    """,
    unsafe_allow_html=True
)
