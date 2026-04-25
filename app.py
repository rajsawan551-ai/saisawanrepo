import json
import graphviz
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

INTERACTIVE_GRAPH_HEIGHT = 520

st.set_page_config(layout="wide")

# ---------------- SESSION ----------------
if "graph" not in st.session_state:
    st.session_state.graph = None


# ---------------- HELPERS ----------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower()
    return df


def find_column(df, options):
    # 1. Prioritize exact column matches
    for opt in options:
        if opt in df.columns:
            return opt
    # 2. Fallback to substring matches
    for opt in options:
        for col in df.columns:
            if opt in col:
                return col
    return None


def load_sheet(xls, possible_names):
    possible_names = [str(p).strip().lower() for p in possible_names]
    for name in xls.sheet_names:
        sheet_name = str(name).strip().lower()
        if any(p in sheet_name for p in possible_names):
            return pd.read_excel(xls, sheet_name=name)
    return None


def load_resource_sheet_fallback(xls):
    for name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=name)
            df = clean_columns(df)
            cols = set(df.columns)
            has_resource_cols = any("resource" in c for c in cols)
            has_id_or_name = any("id" in c for c in cols) or any("name" in c for c in cols)
            if has_resource_cols and has_id_or_name:
                return df
        except Exception:
            continue
    return None


VIEW_ATTRIBUTE_FIELDS = {
    "Engineering": [("cost", "Cost"), ("time_min", "Time (Min)"), ("difficulty", "Difficulty")],
    "Quality": [("quality_grade", "Quality Grade"), ("quality_check", "Quality Check")],
    "Sustainability": [("co2", "CO2 Impact"), ("energy_level", "Energy Level")],
    "Reliability": [("mtbf", "MTBF"), ("availability", "Availability")],
}
NODE_TYPE_OPTIONS = ["Product", "Process", "Resource"]
RESERVED_ATTRIBUTE_KEYS = {"id", "name", "type", "source", "target"}


def normalize_token(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    token = str(value).strip()
    if not token or token.lower() == "nan":
        return ""
    return token.lower()


def normalize_attribute_key(raw_key):
    key = str(raw_key).strip().lower().replace(" ", "_")
    return key


def ensure_node_columns(nodes_df):
    safe_df = nodes_df.copy()
    for col in ["id", "name", "type"]:
        if col not in safe_df.columns:
            safe_df[col] = pd.NA
        safe_df[col] = safe_df[col].astype("object")
    return safe_df


def ensure_edge_columns(edges_df):
    safe_df = edges_df.copy()
    for col in ["source", "target"]:
        if col not in safe_df.columns:
            safe_df[col] = pd.NA
    return safe_df[["source", "target"] + [c for c in safe_df.columns if c not in {"source", "target"}]]


def parse_extra_attributes_json(raw_json):
    text = str(raw_json or "").strip()
    if not text:
        return {}, None
    try:
        payload = json.loads(text)
    except Exception as exc:
        return {}, f"Invalid attributes JSON: {exc}"
    if not isinstance(payload, dict):
        return {}, "Extra attributes must be a JSON object, for example: {\"supplier\": \"ACME\"}"

    normalized = {}
    for key, value in payload.items():
        norm_key = normalize_attribute_key(key)
        if not norm_key or norm_key in RESERVED_ATTRIBUTE_KEYS:
            continue
        normalized[norm_key] = value
    return normalized, None


def coerce_value_for_column(series, raw_value):
    if raw_value is None:
        if pd.api.types.is_numeric_dtype(series.dtype):
            return float("nan")
        return pd.NA

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            if pd.api.types.is_numeric_dtype(series.dtype):
                return float("nan")
            return pd.NA

    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric_value = pd.to_numeric(raw_value, errors="coerce")
        if pd.notna(numeric_value):
            if pd.api.types.is_integer_dtype(series.dtype) and float(numeric_value).is_integer():
                return int(numeric_value)
            return float(numeric_value)
        return raw_value

    if pd.api.types.is_bool_dtype(series.dtype):
        text = str(raw_value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return raw_value

    return raw_value


def safe_set_cell(nodes_df, row_index, column_name, raw_value):
    if column_name not in nodes_df.columns:
        nodes_df[column_name] = pd.NA
    value_to_set = coerce_value_for_column(nodes_df[column_name], raw_value)
    try:
        nodes_df.at[row_index, column_name] = value_to_set
    except (TypeError, ValueError):
        nodes_df[column_name] = nodes_df[column_name].astype("object")
        nodes_df.at[row_index, column_name] = raw_value
    return nodes_df


def node_primary_alias_from_row(row):
    node_id = normalize_token(row.get("id"))
    if node_id:
        return node_id
    return normalize_token(row.get("name"))


def node_aliases_from_row(row):
    aliases = set()
    id_alias = normalize_token(row.get("id"))
    name_alias = normalize_token(row.get("name"))
    if id_alias:
        aliases.add(id_alias)
    if name_alias:
        aliases.add(name_alias)
    return aliases


def format_node_label_from_row(row):
    node_id = str(row.get("id", "")).strip() or "-"
    node_name = str(row.get("name", "")).strip() or "-"
    node_type = str(row.get("type", "")).strip() or "-"
    return f"{node_id} | {node_name} | {node_type}"


def find_node_indices_by_field(nodes_df, field_name, norm_value):
    if not norm_value or field_name not in nodes_df.columns:
        return []
    matches = []
    for idx, row in nodes_df.iterrows():
        if normalize_token(row.get(field_name)) == norm_value:
            matches.append(int(idx))
    return matches


def resolve_existing_node_index(nodes_df, node_id, node_name):
    id_norm = normalize_token(node_id)
    name_norm = normalize_token(node_name)
    id_matches = find_node_indices_by_field(nodes_df, "id", id_norm)
    name_matches = find_node_indices_by_field(nodes_df, "name", name_norm)

    if len(id_matches) > 1:
        return None, f"Multiple nodes share ID '{node_id}'. Please clean duplicates before editing."
    if len(name_matches) > 1:
        return None, f"Multiple nodes share name '{node_name}'. Please clean duplicates before editing."

    idx_id = id_matches[0] if id_matches else None
    idx_name = name_matches[0] if name_matches else None
    if idx_id is not None and idx_name is not None and idx_id != idx_name:
        return None, "ID matches one node while name matches another node. Resolve this conflict first."

    return idx_id if idx_id is not None else idx_name, None


def find_conflicting_node_index(nodes_df, node_id, node_name, exclude_index):
    id_norm = normalize_token(node_id)
    name_norm = normalize_token(node_name)
    for idx, row in nodes_df.iterrows():
        if int(idx) == int(exclude_index):
            continue
        same_id = bool(id_norm) and normalize_token(row.get("id")) == id_norm
        same_name = bool(name_norm) and normalize_token(row.get("name")) == name_norm
        if same_id or same_name:
            return int(idx)
    return None


def apply_view_and_extra_attributes(nodes_df, row_index, view, view_attributes, extra_attributes,
                                    clear_blank_view_fields=False):
    for attr_key, _ in VIEW_ATTRIBUTE_FIELDS.get(view, []):
        if attr_key not in nodes_df.columns:
            nodes_df[attr_key] = pd.NA
        raw_value = view_attributes.get(attr_key, "")
        text_value = str(raw_value).strip() if raw_value is not None else ""
        if text_value:
            nodes_df = safe_set_cell(nodes_df, row_index, attr_key, text_value)
        elif clear_blank_view_fields:
            nodes_df = safe_set_cell(nodes_df, row_index, attr_key, None)

    for attr_key, attr_value in extra_attributes.items():
        if attr_key not in nodes_df.columns:
            nodes_df[attr_key] = pd.NA
        nodes_df = safe_set_cell(nodes_df, row_index, attr_key, attr_value)

    return nodes_df


def remap_edges_for_alias_change(edges_df, old_aliases, new_primary_alias):
    safe_edges = ensure_edge_columns(edges_df)
    if safe_edges.empty or not old_aliases or not new_primary_alias:
        return safe_edges

    old_aliases_norm = {normalize_token(alias) for alias in old_aliases if normalize_token(alias)}
    if not old_aliases_norm:
        return safe_edges

    def remap(value):
        norm_value = normalize_token(value)
        if norm_value in old_aliases_norm:
            return new_primary_alias
        return norm_value

    safe_edges["source"] = safe_edges["source"].apply(remap)
    safe_edges["target"] = safe_edges["target"].apply(remap)
    return safe_edges


def create_or_reuse_node(nodes_df, edges_df, node_id, node_name, node_type, view, view_attributes, extra_attributes):
    safe_nodes = ensure_node_columns(nodes_df)
    safe_edges = ensure_edge_columns(edges_df)

    clean_id = str(node_id or "").strip()
    clean_name = str(node_name or "").strip()
    if not clean_id and not clean_name:
        return safe_nodes, safe_edges, False, "Node ID or Node Name is required."

    if not clean_id:
        clean_id = clean_name
    if not clean_name:
        clean_name = clean_id
    if node_type not in NODE_TYPE_OPTIONS:
        node_type = "Resource"

    existing_index, resolve_error = resolve_existing_node_index(safe_nodes, clean_id, clean_name)
    if resolve_error:
        return safe_nodes, safe_edges, False, resolve_error

    if existing_index is None:
        new_row = {"id": clean_id, "name": clean_name, "type": node_type}
        safe_nodes = pd.concat([safe_nodes, pd.DataFrame([new_row])], ignore_index=True)
        row_index = int(safe_nodes.index[-1])
        created_new = True
        old_aliases = set()
    else:
        row_index = int(existing_index)
        created_new = False
        old_aliases = node_aliases_from_row(safe_nodes.loc[row_index])

    safe_nodes = safe_set_cell(safe_nodes, row_index, "id", clean_id)
    safe_nodes = safe_set_cell(safe_nodes, row_index, "name", clean_name)
    safe_nodes = safe_set_cell(safe_nodes, row_index, "type", node_type)
    safe_nodes = apply_view_and_extra_attributes(
        safe_nodes,
        row_index,
        view,
        view_attributes,
        extra_attributes,
        clear_blank_view_fields=False,
    )

    new_primary_alias = node_primary_alias_from_row(safe_nodes.loc[row_index])
    safe_edges = remap_edges_for_alias_change(safe_edges, old_aliases, new_primary_alias)
    return safe_nodes.reset_index(drop=True), safe_edges.reset_index(drop=True), created_new, None


def update_existing_node(nodes_df, edges_df, row_index, node_id, node_name, node_type, view, view_attributes,
                         extra_attributes, clear_blank_view_fields=False):
    safe_nodes = ensure_node_columns(nodes_df)
    safe_edges = ensure_edge_columns(edges_df)
    if int(row_index) not in set(safe_nodes.index.tolist()):
        return safe_nodes, safe_edges, "Selected node no longer exists."

    clean_id = str(node_id or "").strip()
    clean_name = str(node_name or "").strip()
    if not clean_id and not clean_name:
        return safe_nodes, safe_edges, "Node ID or Node Name is required."
    if not clean_id:
        clean_id = clean_name
    if not clean_name:
        clean_name = clean_id
    if node_type not in NODE_TYPE_OPTIONS:
        node_type = "Resource"

    conflict_index = find_conflicting_node_index(safe_nodes, clean_id, clean_name, exclude_index=row_index)
    if conflict_index is not None:
        return safe_nodes, safe_edges, f"Another node already uses this ID or name (row {conflict_index})."

    old_aliases = node_aliases_from_row(safe_nodes.loc[row_index])
    safe_nodes = safe_set_cell(safe_nodes, row_index, "id", clean_id)
    safe_nodes = safe_set_cell(safe_nodes, row_index, "name", clean_name)
    safe_nodes = safe_set_cell(safe_nodes, row_index, "type", node_type)
    safe_nodes = apply_view_and_extra_attributes(
        safe_nodes,
        row_index,
        view,
        view_attributes,
        extra_attributes,
        clear_blank_view_fields=clear_blank_view_fields,
    )

    new_primary_alias = node_primary_alias_from_row(safe_nodes.loc[row_index])
    safe_edges = remap_edges_for_alias_change(safe_edges, old_aliases, new_primary_alias)
    return safe_nodes.reset_index(drop=True), safe_edges.reset_index(drop=True), None


def delete_node_and_related_edges(nodes_df, edges_df, row_index):
    safe_nodes = ensure_node_columns(nodes_df)
    safe_edges = ensure_edge_columns(edges_df)
    if int(row_index) not in set(safe_nodes.index.tolist()):
        return safe_nodes, safe_edges, "Selected node no longer exists."

    aliases_to_remove = node_aliases_from_row(safe_nodes.loc[row_index])
    safe_nodes = safe_nodes.drop(index=row_index).reset_index(drop=True)

    if not safe_edges.empty and aliases_to_remove:
        source_norm = safe_edges["source"].apply(normalize_token)
        target_norm = safe_edges["target"].apply(normalize_token)
        keep_mask = (~source_norm.isin(aliases_to_remove)) & (~target_norm.isin(aliases_to_remove))
        safe_edges = safe_edges.loc[keep_mask].reset_index(drop=True)
    else:
        safe_edges = safe_edges.reset_index(drop=True)

    return safe_nodes, safe_edges, None


def add_edge_relation(edges_df, source_alias, target_alias):
    safe_edges = ensure_edge_columns(edges_df)
    source_norm = normalize_token(source_alias)
    target_norm = normalize_token(target_alias)
    if not source_norm or not target_norm:
        return safe_edges, "Source and target are required."
    if source_norm == target_norm:
        return safe_edges, "Self-loop edges are not allowed."

    existing_source = safe_edges["source"].apply(normalize_token)
    existing_target = safe_edges["target"].apply(normalize_token)
    already_exists = ((existing_source == source_norm) & (existing_target == target_norm)).any()
    if already_exists:
        return safe_edges, "This edge already exists."

    safe_edges = pd.concat(
        [safe_edges, pd.DataFrame([{"source": source_norm, "target": target_norm}])],
        ignore_index=True,
    )
    return safe_edges.reset_index(drop=True), None


def delete_edges_by_indices(edges_df, edge_indices):
    safe_edges = ensure_edge_columns(edges_df)
    if not edge_indices:
        return safe_edges
    return safe_edges.drop(index=edge_indices, errors="ignore").reset_index(drop=True)


def render_view_attribute_inputs(view, key_prefix, defaults=None):
    values = {}
    for attr_key, label in VIEW_ATTRIBUTE_FIELDS.get(view, []):
        default_value = ""
        if defaults is not None and attr_key in defaults and pd.notna(defaults[attr_key]):
            default_value = str(defaults[attr_key])
        values[attr_key] = st.text_input(label, value=default_value, key=f"{key_prefix}_{attr_key}")
    return values


def render_crud_main_page(view):
    if "nodes_df" not in st.session_state:
        return

    nodes_df = ensure_node_columns(st.session_state.nodes_df)
    edges_df = ensure_edge_columns(st.session_state.edges_df if "edges_df" in st.session_state else pd.DataFrame())

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["➕ Create Node", "✏️ Update Node", "🗑️ Delete Node", "🔗 Add Edge", "✂️ Delete Edge"])

    with tab1:
        st.subheader("Create or Reuse Node")
        with st.form("create_reuse_node_form", clear_on_submit=True):
            node_type = st.selectbox("Assign Type", NODE_TYPE_OPTIONS, key="create_node_type")
            node_id = st.text_input("Node ID", key="create_node_id")
            node_name = st.text_input("Node Name", key="create_node_name")
            st.caption(f"{view} attributes")
            view_attributes = render_view_attribute_inputs(view, key_prefix=f"create_{view.lower()}")
            extra_json = st.text_area("Extra attributes JSON (optional)", key="create_extra_json",
                                      placeholder='{"supplier": "ACME"}')
            create_submit = st.form_submit_button("Save Node")

        if create_submit:
            extra_attributes, parse_error = parse_extra_attributes_json(extra_json)
            if parse_error:
                st.error(parse_error)
            else:
                new_nodes, new_edges, created_new, op_error = create_or_reuse_node(
                    nodes_df, edges_df, node_id, node_name, node_type, view, view_attributes, extra_attributes
                )
                if op_error:
                    st.error(op_error)
                else:
                    st.session_state.nodes_df = new_nodes
                    st.session_state.edges_df = new_edges
                    st.success("Node created." if created_new else "Existing node updated for this view.")
                    st.rerun()

    with tab2:
        st.subheader("Update Existing Node")
        if nodes_df.empty:
            st.info("No nodes available.")
        else:
            selectable_indices = list(nodes_df.index)
            selected_index = st.selectbox("Select Node", options=selectable_indices,
                                          format_func=lambda idx: format_node_label_from_row(nodes_df.loc[idx]),
                                          key="update_node_select")
            selected_row = nodes_df.loc[selected_index]
            current_type = str(selected_row.get("type", "Resource")).strip().title()
            if current_type not in NODE_TYPE_OPTIONS: current_type = "Resource"
            current_view_defaults = {key: selected_row.get(key) for key, _ in VIEW_ATTRIBUTE_FIELDS.get(view, [])}

            with st.form(f"update_node_form_{selected_index}", clear_on_submit=False):
                node_type = st.selectbox("Assign Type", NODE_TYPE_OPTIONS, index=NODE_TYPE_OPTIONS.index(current_type),
                                         key=f"upd_type_{selected_index}")
                node_id = st.text_input("Node ID",
                                        value="" if pd.isna(selected_row.get("id")) else str(selected_row.get("id")),
                                        key=f"upd_id_{selected_index}")
                node_name = st.text_input("Node Name", value="" if pd.isna(selected_row.get("name")) else str(
                    selected_row.get("name")), key=f"upd_name_{selected_index}")
                st.caption(f"{view} attributes")
                view_attributes = render_view_attribute_inputs(view, key_prefix=f"upd_{selected_index}_{view.lower()}",
                                                               defaults=current_view_defaults)
                clear_blank = st.checkbox("Clear blank fields for this view", value=False,
                                          key=f"upd_clear_{selected_index}")
                extra_json = st.text_area("Extra attributes JSON (optional)", value="",
                                          key=f"upd_json_{selected_index}")
                update_submit = st.form_submit_button("Apply Update")

            if update_submit:
                extra_attributes, parse_error = parse_extra_attributes_json(extra_json)
                if parse_error:
                    st.error(parse_error)
                else:
                    new_nodes, new_edges, op_error = update_existing_node(
                        nodes_df, edges_df, selected_index, node_id, node_name, node_type, view, view_attributes,
                        extra_attributes, clear_blank
                    )
                    if op_error:
                        st.error(op_error)
                    else:
                        st.session_state.nodes_df = new_nodes
                        st.session_state.edges_df = new_edges
                        st.success("Node updated.")
                        st.rerun()

    with tab3:
        st.subheader("Delete Node")
        if nodes_df.empty:
            st.info("No nodes available.")
        else:
            with st.form("delete_node_form"):
                selected_index = st.selectbox("Node to delete", options=list(nodes_df.index),
                                              format_func=lambda idx: format_node_label_from_row(nodes_df.loc[idx]))
                confirm_delete = st.checkbox("I understand this also removes connected edges", value=False)
                delete_submit = st.form_submit_button("Delete Node")

            if delete_submit:
                if not confirm_delete:
                    st.warning("Please confirm deletion first.")
                else:
                    new_nodes, new_edges, op_error = delete_node_and_related_edges(nodes_df, edges_df, selected_index)
                    if op_error:
                        st.error(op_error)
                    else:
                        st.session_state.nodes_df = new_nodes
                        st.session_state.edges_df = new_edges
                        st.success("Node and connected edges deleted.")
                        st.rerun()

    with tab4:
        st.subheader("Create Edge")
        if len(nodes_df.index) < 2:
            st.info("At least two nodes are required.")
        else:
            with st.form("create_edge_form"):
                source_index = st.selectbox("Source Node", options=list(nodes_df.index),
                                            format_func=lambda idx: format_node_label_from_row(nodes_df.loc[idx]))
                target_index = st.selectbox("Target Node", options=list(nodes_df.index), index=1,
                                            format_func=lambda idx: format_node_label_from_row(nodes_df.loc[idx]))
                edge_submit = st.form_submit_button("Add Edge")

            if edge_submit:
                source_alias = node_primary_alias_from_row(nodes_df.loc[source_index])
                target_alias = node_primary_alias_from_row(nodes_df.loc[target_index])
                new_edges, op_error = add_edge_relation(edges_df, source_alias, target_alias)
                if op_error:
                    st.error(op_error)
                else:
                    st.session_state.edges_df = new_edges
                    st.success("Edge added.")
                    st.rerun()

    with tab5:
        st.subheader("Delete Edge")
        if edges_df.empty:
            st.info("No edges available.")
        else:
            with st.form("delete_edges_form"):
                selected_edges = st.multiselect("Edges to delete", options=list(edges_df.index), format_func=lambda
                    idx: f"{edges_df.loc[idx, 'source']} -> {edges_df.loc[idx, 'target']}")
                delete_edges_submit = st.form_submit_button("Delete Selected Edges")

            if delete_edges_submit:
                if not selected_edges:
                    st.warning("Select at least one edge.")
                else:
                    st.session_state.edges_df = delete_edges_by_indices(edges_df, selected_edges)
                    st.success("Selected edges deleted.")
                    st.rerun()


def build_label(row, view):
    name = str(row.get("name", "Unknown"))
    label = name

    if view == "Engineering":
        if pd.notna(row.get("cost")): label += f"\nCost: {row.get('cost')}"
        if pd.notna(row.get("time_min")): label += f"\nTime: {row.get('time_min')}m"
        if pd.notna(row.get("difficulty")): label += f"\nDiff: {row.get('difficulty')}"

    elif view == "Quality":
        if pd.notna(row.get("quality_grade")): label += f"\nQ-Grade: {row.get('quality_grade')}"
        if pd.notna(row.get("quality_check")): label += f"\nQ-Check: {row.get('quality_check')}"

    elif view == "Sustainability":
        if pd.notna(row.get("co2")): label += f"\nCO2: {row.get('co2')}"
        if pd.notna(row.get("energy_level")): label += f"\nEnergy: {row.get('energy_level')}"

    elif view == "Reliability":
        if pd.notna(row.get("mtbf")): label += f"\nMTBF: {row.get('mtbf')}"
        if pd.notna(row.get("availability")): label += f"\nAvail: {row.get('availability')}"

    return label


def build_interactive_graph_html(nodes_df, edges_df, view, height=INTERACTIVE_GRAPH_HEIGHT):
    local_nodes = ensure_node_columns(nodes_df)
    local_edges = ensure_edge_columns(edges_df)

    local_nodes["id"] = local_nodes["id"].astype(str).str.strip().str.lower()
    local_nodes["name"] = local_nodes["name"].astype(str).str.strip().str.lower()
    local_nodes["type"] = local_nodes["type"].astype(str).str.strip().str.lower()
    local_edges["source"] = local_edges["source"].astype(str).str.strip().str.lower()
    local_edges["target"] = local_edges["target"].astype(str).str.strip().str.lower()

    alias_to_key = {}
    known_keys = set()
    for _, row in local_nodes.iterrows():
        node_key = row["id"] if row["id"] and row["id"] != "nan" else row["name"]
        if not node_key or node_key == "nan":
            continue
        alias_to_key[row["id"]] = node_key
        alias_to_key[row["name"]] = node_key
        known_keys.add(node_key)

    connected_keys = set()
    mapped_edges = []
    for _, edge in local_edges.iterrows():
        source_key = alias_to_key.get(edge["source"])
        target_key = alias_to_key.get(edge["target"])
        if source_key in known_keys and target_key in known_keys:
            connected_keys.add(source_key)
            connected_keys.add(target_key)
            mapped_edges.append((source_key, target_key))

    local_nodes["node_key"] = local_nodes["id"].where(
        (local_nodes["id"] != "") & (local_nodes["id"] != "nan"), local_nodes["name"]
    )
    filtered = local_nodes[local_nodes["node_key"].isin(known_keys)].copy()
    if filtered.empty:
        return "<div style='padding:14px;border:1px solid #fde68a;background:#fffbeb;color:#92400e;border-radius:10px;'>No connected nodes found to render.</div>"

    type_style = {
        "product": {"bg": "#fecaca", "border": "#b91c1c", "y": -110},
        "process": {"bg": "#fed7aa", "border": "#c2410c", "y": 0},
        "resource": {"bg": "#bbf7d0", "border": "#15803d", "y": 110},
    }
    type_order = ["product", "process", "resource"]

    nodes_payload = []
    for node_type in type_order:
        typed = filtered[filtered["type"] == node_type]
        typed = typed.sort_values("name") if "name" in typed.columns else typed.sort_values("node_key")
        count = len(typed)
        for idx, (_, row) in enumerate(typed.iterrows()):
            node_id = str(row.get("node_key", "")).strip().lower()
            if not node_id or node_id == "nan":
                continue
            style = type_style.get(node_type, {"bg": "#e2e8f0", "border": "#334155", "y": 0})
            label = str(build_label(row, view)).replace("\r\n", "\n")
            if count <= 1:
                x_pos = 0
            else:
                x_step = max(220, min(420, 1800 / (count - 1)))
                x_pos = (idx - (count - 1) / 2.0) * x_step
            nodes_payload.append(
                {
                    "id": node_id,
                    "label": label,
                    "shape": "box",
                    "x": x_pos,
                    "y": style["y"],
                    "color": {
                        "background": style["bg"],
                        "border": style["border"],
                        "highlight": {"background": style["bg"], "border": style["border"]},
                    },
                    "margin": 16,
                    "widthConstraint": {"minimum": 220, "maximum": 300},
                    "font": {"color": "#0f172a", "size": 20, "face": "Segoe UI"},
                }
            )

    edges_payload = []
    for source_key, target_key in mapped_edges:
        if source_key in known_keys and target_key in known_keys:
            edges_payload.append(
                {
                    "from": source_key,
                    "to": target_key,
                    "arrows": "to",
                    "color": {"color": "#64748b", "highlight": "#0f172a"},
                    "smooth": {
                        "enabled": True,
                        "type": "cubicBezier",
                        "forceDirection": "vertical",
                        "roundness": 0.45,
                    },
                }
            )

    options = {
        "autoResize": True,
        "layout": {"improvedLayout": True, "randomSeed": 8},
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "tooltipDelay": 120,
            "navigationButtons": True,
            "zoomSpeed": 0.35,
        },
        "physics": {"enabled": False},
        "nodes": {
            "shape": "box",
            "borderWidth": 2,
            "borderWidthSelected": 3,
            "font": {"multi": True, "size": 20},
        },
        "edges": {
            "width": 2.3,
            "selectionWidth": 3,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.75}},
        },
    }

    template = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body {
      margin: 0;
      padding: 0;
      background:
        radial-gradient(circle at 10% 12%, #dbeafe 0%, transparent 35%),
        radial-gradient(circle at 88% 82%, #dcfce7 0%, transparent 30%),
        linear-gradient(145deg, #f8fafc 0%, #eef2ff 48%, #f0fdf4 100%);
      font-family: "Segoe UI", sans-serif;
    }
    .graph-shell {
      margin: 0;
      padding: 16px;
      border: 1px solid #bfdbfe;
      border-radius: 16px;
      background: linear-gradient(160deg, rgba(255, 255, 255, 0.95) 0%, rgba(241, 245, 249, 0.92) 100%);
      box-shadow: 0 12px 32px rgba(15, 23, 42, 0.10);
    }
    .graph-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
      color: #0f172a;
      font-size: 14px;
      font-weight: 600;
    }
    .graph-badge {
      padding: 4px 10px;
      border-radius: 999px;
      background: #dbeafe;
      color: #1e3a8a;
      font-size: 12px;
      border: 1px solid #bfdbfe;
    }
    #interactive-network {
      width: 100%;
      height: __HEIGHT__px;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background:
        linear-gradient(0deg, rgba(255, 255, 255, 0.96) 0%, rgba(248, 250, 252, 0.96) 100%),
        repeating-linear-gradient(0deg, rgba(148, 163, 184, 0.10) 0px, rgba(148, 163, 184, 0.10) 1px, transparent 1px, transparent 28px),
        repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.10) 0px, rgba(148, 163, 184, 0.10) 1px, transparent 1px, transparent 28px);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
    }
    .graph-hint {
      margin-top: 10px;
      color: #334155;
      font-size: 13px;
      line-height: 1.4;
    }
  </style>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
  <div class="graph-shell">
    <div class="graph-head">
      <span>Interactive Production Graph</span>
      <span class="graph-badge">Drag | Zoom | Pan</span>
    </div>
    <div id="interactive-network"></div>
    <div class="graph-hint">
      Drag nodes to reposition. Use mouse wheel to zoom and drag empty space to pan.
    </div>
  </div>
  <script>
    const container = document.getElementById("interactive-network");
    if (!window.vis) {
      container.innerHTML = "<div style='padding:14px;color:#991b1b;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;'>Interactive library could not load. Switch to Static mode.</div>";
    } else {
      const nodes = new vis.DataSet(__NODES__);
      const edges = new vis.DataSet(__EDGES__);
      const data = { nodes: nodes, edges: edges };
      const options = __OPTIONS__;
      const network = new vis.Network(container, data, options);
      setTimeout(() => network.fit({ animation: true, margin: 20 }), 120);
    }
  </script>
</body>
</html>
"""

    return (
        template.replace("__HEIGHT__", str(int(height)))
        .replace("__NODES__", json.dumps(nodes_payload))
        .replace("__EDGES__", json.dumps(edges_payload))
        .replace("__OPTIONS__", json.dumps(options))
    )


def build_networkx_graph():
    """Converts the session state DataFrames into a NetworkX DiGraph."""
    G = nx.DiGraph()
    if "nodes_df" not in st.session_state or st.session_state.nodes_df.empty:
        return G

    alias_to_key = {}

    # 1. Add Nodes and build the Alias Dictionary
    for _, row in st.session_state.nodes_df.iterrows():
        raw_id = str(row.get("id", "")).strip().lower()
        raw_name = str(row.get("name", "")).strip().lower()

        # Primary key is ID, fallback is Name
        node_key = raw_id if raw_id and raw_id != "nan" else raw_name
        if not node_key or node_key == "nan":
            continue

        # Map both the ID and the Name to the exact same core Node Key
        alias_to_key[raw_id] = node_key
        alias_to_key[raw_name] = node_key

        attr_dict = row.dropna().to_dict()
        G.add_node(node_key, **attr_dict)

    # 2. Add Edges using the mapped keys
    if "edges_df" in st.session_state and not st.session_state.edges_df.empty:
        for _, row in st.session_state.edges_df.iterrows():
            src_raw = str(row.get("source", "")).strip().lower()
            tgt_raw = str(row.get("target", "")).strip().lower()

            # Translate the Excel edge text into the core Node Key
            src = alias_to_key.get(src_raw, src_raw)
            tgt = alias_to_key.get(tgt_raw, tgt_raw)

            if src and tgt and src != "nan" and tgt != "nan":
                # Safety fallback: If edge exists but node wasn't in nodes_df
                if src not in G: G.add_node(src, name=src_raw.title(), type="Unknown")
                if tgt not in G: G.add_node(tgt, name=tgt_raw.title(), type="Unknown")

                G.add_edge(src, tgt)

    return G


# --- Task 6a: Custom Car Requirement ---
def check_process_resource_assignment(G):
    """
    Requirement Check: Every 'Process' node MUST have an incoming edge
    from a 'Resource' node. (You can't assemble a car without a tool/worker).
    """
    violations = []
    for node, data in G.nodes(data=True):
        if str(data.get("type", "")).lower() == "process":
            # Check predecessors (incoming edges)
            has_resource = False
            for pred in G.predecessors(node):
                if str(G.nodes[pred].get("type", "")).lower() == "resource":
                    has_resource = True
                    break
            if not has_resource:
                violations.append(data.get("name", node))
    return violations


# --- Task 6b: View-Specific Dependencies ---
def analyze_downstream_quality_impact(G, start_node_name):
    """
    View Dependency: If a specific process or product fails a 'Quality' check,
    what downstream products are affected? Uses Graph Reachability (Paths).
    """
    # Find node ID from name
    start_node = None
    for n, d in G.nodes(data=True):
        if str(d.get("name", "")).lower() == str(start_node_name).lower():
            start_node = n
            break

    if not start_node or start_node not in G:
        return []

    # Find all nodes reachable from the start node (downstream dependencies)
    reachable = list(nx.descendants(G, start_node))
    affected_items = [G.nodes[n].get("name", n) for n in reachable]
    return affected_items


# --- Task 6c: Identify Similarly Structured Elements ---
def find_similarly_structured_processes(G):
    """
    Identifies processes that require the exact same number of inputs
    and produce the exact same number of outputs (Structural similarity).
    """
    process_signatures = {}
    for node, data in G.nodes(data=True):
        if str(data.get("type", "")).lower() == "process":
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            sig = f"{in_degree}_in_{out_degree}_out"

            if sig not in process_signatures:
                process_signatures[sig] = []
            process_signatures[sig].append(data.get("name", node))

    # Filter for signatures that have more than 1 matching process
    similar_groups = {sig: nodes for sig, nodes in process_signatures.items() if len(nodes) > 1}
    return similar_groups


# --- Task 6d: Identification of Disconnected Segments ---
def find_disconnected_segments(G):
    """
    Uses Weakly Connected Components to find isolated islands in the graph.
    """
    components = list(nx.weakly_connected_components(G))
    if len(components) <= 1:
        return []  # Graph is fully connected

    # Sort by size so the largest component (the main assembly line) is first
    sorted_comps = sorted(components, key=len, reverse=True)

    isolated_segments = []
    # Skip [0] because we do not want to report the main graph as an orphan!
    for comp in sorted_comps[1:]:
        names = [G.nodes[n].get("name", n) for n in comp]
        isolated_segments.append(names)
    return isolated_segments


# ---------------- GRAPH ----------------
def build_graphviz(nodes_df, edges_df, view):
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", splines="polyline", ranksep="1.5", nodesep="0.8", newrank="true")

    local_nodes = ensure_node_columns(nodes_df)
    local_edges = ensure_edge_columns(edges_df)

    # Normalize IDs/names/types and endpoints
    local_nodes["id"] = local_nodes["id"].astype(str).str.strip().str.lower()
    local_nodes["name"] = local_nodes["name"].astype(str).str.strip().str.lower()
    local_nodes["type"] = local_nodes["type"].astype(str).str.strip().str.lower()
    local_edges["source"] = local_edges["source"].astype(str).str.strip().str.lower()
    local_edges["target"] = local_edges["target"].astype(str).str.strip().str.lower()

    # Build canonical node key map (id + name -> same node)
    alias_to_key = {}
    known_keys = set()
    for _, row in local_nodes.iterrows():
        node_key = row["id"] if row["id"] and row["id"] != "nan" else row["name"]
        if not node_key or node_key == "nan":
            continue
        alias_to_key[row["id"]] = node_key
        alias_to_key[row["name"]] = node_key
        known_keys.add(node_key)

    # Keep only nodes that are connected through mapped edges
    connected_keys = set()
    for _, e in local_edges.iterrows():
        s_raw = e["source"]
        t_raw = e["target"]
        s_key = alias_to_key.get(s_raw)
        t_key = alias_to_key.get(t_raw)
        if s_key in known_keys and t_key in known_keys:
            connected_keys.add(s_key)
            connected_keys.add(t_key)

    local_nodes["node_key"] = local_nodes["id"].where(
        (local_nodes["id"] != "") & (local_nodes["id"] != "nan"), local_nodes["name"]
    )
    filtered = local_nodes[local_nodes["node_key"].isin(known_keys)]

    products = filtered[filtered["type"] == "product"]
    processes = filtered[filtered["type"] == "process"]
    resources = filtered[filtered["type"] == "resource"]
    product_ids = []
    process_ids = []
    resource_ids = []

    # PRODUCTS
    with dot.subgraph(name="products_rank") as s:
        s.attr(rank="same")
        products_sorted = products.sort_values("name") if "name" in products.columns else products.sort_values(
            "node_key")
        for _, row in products_sorted.iterrows():
            node_id = str(row.get("node_key", "")).strip().lower()
            if not node_id or node_id == "nan":
                continue
            product_ids.append(node_id)
            s.node(
                str(node_id),
                label=build_label(row, view),
                shape="box",
                style="filled,rounded",
                fillcolor="#fecaca"
            )

    # PROCESSES
    with dot.subgraph(name="processes_rank") as s:
        s.attr(rank="same")
        processes_sorted = processes.sort_values("name") if "name" in processes.columns else processes.sort_values(
            "node_key")
        for _, row in processes_sorted.iterrows():
            node_id = str(row.get("node_key", "")).strip().lower()
            if not node_id or node_id == "nan":
                continue
            process_ids.append(node_id)
            s.node(
                str(node_id),
                label=build_label(row, view),
                shape="box",
                style="filled,rounded",
                fillcolor="#fed7aa"
            )

    # RESOURCES
    with dot.subgraph(name="resources_rank") as s:
        s.attr(rank="same")
        resources_sorted = resources.sort_values("name") if "name" in resources.columns else resources.sort_values(
            "node_key")
        for _, row in resources_sorted.iterrows():
            node_id = str(row.get("node_key", "")).strip().lower()
            if not node_id or node_id == "nan":
                continue
            resource_ids.append(node_id)
            s.node(
                str(node_id),
                label=build_label(row, view),
                shape="box",
                style="filled,rounded",
                fillcolor="#bbf7d0"
            )

    # Keep a strict left->right order inside each row.
    for ids in [product_ids, process_ids, resource_ids]:
        for i in range(len(ids) - 1):
            dot.edge(str(ids[i]), str(ids[i + 1]), style="invis", weight="100")

    # Force top-to-bottom row stacking: Products -> Processes -> Resources.
    if product_ids and process_ids:
        dot.edge(str(product_ids[0]), str(process_ids[0]), style="invis", weight="200")
    if process_ids and resource_ids:
        dot.edge(str(process_ids[0]), str(resource_ids[0]), style="invis", weight="200")

    # EDGES
    for _, row in local_edges.iterrows():
        source = alias_to_key.get(row.get("source", ""))
        target = alias_to_key.get(row.get("target", ""))
        if source in known_keys and target in known_keys:
            dot.edge(str(source), str(target), constraint="false")

    return dot


# ---------------- GRAPH TAB ----------------
def graph_tab():
    st.markdown(
        """
        <style>
          .main .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
          }
          .graph-hero {
            margin: 0 0 0.9rem 0;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            border: 1px solid #bfdbfe;
            background: linear-gradient(135deg, #eff6ff 0%, #f0fdf4 100%);
            color: #0f172a;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
          }
          .graph-hero strong {
            color: #1e3a8a;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Production Graph")
    st.markdown(
        """
        <div class="graph-hero">
          <strong>Interactive Mode</strong> gives you a larger canvas with draggable nodes.
          Use the mode selector to switch back to the original static Graphviz layout anytime.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "nodes_df" not in st.session_state:
        st.warning("Load data first")
        return

    st.session_state.nodes_df = ensure_node_columns(st.session_state.nodes_df)
    st.session_state.edges_df = ensure_edge_columns(
        st.session_state.edges_df if "edges_df" in st.session_state else pd.DataFrame()
    )

    view = st.selectbox(
        "Select View",
        ["Engineering", "Quality", "Sustainability", "Reliability"]
    )

    render_mode = st.selectbox(
        "Graph Mode",
        ["Interactive (drag nodes)", "Static (current Graphviz)"]
    )

    if render_mode == "Interactive (drag nodes)":
        html_graph = build_interactive_graph_html(
            st.session_state.nodes_df,
            st.session_state.edges_df,
            view,
            height=INTERACTIVE_GRAPH_HEIGHT
        )
        components.html(html_graph, height=INTERACTIVE_GRAPH_HEIGHT + 120, scrolling=False)
    else:
        dot = build_graphviz(
            st.session_state.nodes_df,
            st.session_state.edges_df,
            view
        )
        try:
            st.graphviz_chart(dot, use_container_width=True)
        except Exception as e:
            st.error(f"Graphviz Rendering Engine Failed: {e}")

    # ---- TABLE ----
    st.subheader("Extracted Attributes")

    df = st.session_state.nodes_df.copy()

    if view == "Engineering":
        target_cols = ["name", "type", "cost", "time_min", "difficulty"]
    elif view == "Quality":
        target_cols = ["name", "type", "quality_grade", "quality_check"]
    elif view == "Sustainability":
        target_cols = ["name", "type", "co2", "energy_level"]
    elif view == "Reliability":
        target_cols = ["name", "type", "mtbf", "availability"]
    else:
        target_cols = ["name", "type"]

    for col in target_cols:
        if col not in df.columns:
            df[col] = pd.NA

    st.dataframe(df[target_cols], height=170, use_container_width=True)


# TASK 6 - Visualize
# TASK 6 - Visualize
def build_diagnostic_tree(G, violations_req, disconnected_req, impact_req=None):
    """
    Builds TWO highly organized Graphviz trees highlighting system issues.
    Separates orphans into a completely separate graph object so they can be rendered side-by-side.
    """
    # 1. Main Graph (Top-to-Bottom layout)
    dot_main = graphviz.Digraph(engine="dot")
    dot_main.attr(rankdir="TB", splines="polyline", nodesep="0.3", ranksep="0.4")

    # 2. Orphan Graph (Top-to-Bottom layout to stack vertically in a side column)
    dot_iso = graphviz.Digraph(engine="dot")
    dot_iso.attr(rankdir="TB", nodesep="0.2", ranksep="0.2")

    isolated_names = []
    for comp in disconnected_req:
        isolated_names.extend(comp)

    isolated_nodes = []
    main_nodes = []
    for n, data in G.nodes(data=True):
        raw_name = data.get("name", n)
        if raw_name in isolated_names or n in isolated_names:
            isolated_nodes.append(n)
        else:
            main_nodes.append(n)

    def add_styled_node(subgraph, node_id, is_orphan=False):
        data = G.nodes[node_id]
        raw_name = data.get("name", node_id)
        name = str(raw_name).title()
        node_type = str(data.get("type", "")).title()

        if is_orphan:
            # Force orphan styling
            fillcolor, color, fontcolor, penwidth = "#fef08a", "#eab308", "#854d0e", "2"
        else:
            # Normal styling & Failure states
            fillcolor, color, fontcolor, penwidth = "#f8fafc", "#cbd5e1", "#334155", "1"
            if raw_name in violations_req or node_id in violations_req:
                fillcolor, color, fontcolor, penwidth = "#fee2e2", "#dc2626", "#991b1b", "3"
            elif impact_req and (raw_name in impact_req or node_id in impact_req):
                fillcolor, color, fontcolor, penwidth = "#ffedd5", "#ea580c", "#9a3412", "2"

        label = f"{name}\n[{node_type}]"
        subgraph.node(str(node_id), label=label, shape="box", style="filled,rounded",
                      fillcolor=fillcolor, color=color, fontcolor=fontcolor,
                      penwidth=penwidth, fontname="Helvetica", fontsize="10", margin="0.05,0.05")

    # --- BUILD MAIN GRAPH ---
    prods = [n for n in main_nodes if str(G.nodes[n].get("type", "")).lower() == "product"]
    procs = [n for n in main_nodes if str(G.nodes[n].get("type", "")).lower() == "process"]
    res = [n for n in main_nodes if str(G.nodes[n].get("type", "")).lower() == "resource"]

    with dot_main.subgraph(name="rank_res") as s:
        s.attr(rank="same")
        for n in res: add_styled_node(s, n)

    with dot_main.subgraph(name="rank_procs") as s:
        s.attr(rank="same")
        for n in procs: add_styled_node(s, n)

    with dot_main.subgraph(name="rank_prods") as s:
        s.attr(rank="same")
        for n in prods: add_styled_node(s, n)

    for n in main_nodes:
        if n not in prods and n not in procs and n not in res:
            add_styled_node(dot_main, n)

    if res and procs:
        dot_main.edge(str(res[0]), str(procs[0]), style="invis", weight="200")
    if procs and prods:
        dot_main.edge(str(procs[0]), str(prods[0]), style="invis", weight="200")

    for src, tgt in G.edges():
        dot_main.edge(str(src), str(tgt), color="#94a3b8", arrowsize="0.6")

    # --- BUILD ORPHAN GRAPH ---
    if isolated_nodes:
        with dot_iso.subgraph(name="cluster_isolated") as c_iso:
            c_iso.attr(label="Orphaned Elements", style="dashed,rounded",
                       color="#ca8a04", fontcolor="#854d0e", bgcolor="#fef9c3")
            # We skip 'rank="same"' here so they stack nicely in a vertical column
            for n in isolated_nodes:
                add_styled_node(c_iso, n, is_orphan=True)

    # Return both graphs!
    return dot_main, dot_iso

# --- Graph Semantic Auto-Healer ---
def auto_heal_graph(G):
    H = G.copy()
    edges_to_add = []
    edges_to_remove = []

    for u, v in H.edges():
        u_type = str(H.nodes[u].get("type", "")).lower()
        v_type = str(H.nodes[v].get("type", "")).lower()

        # Detect flawed mapping: Product -> Resource
        if u_type == "product" and v_type == "resource":
            for pred in H.predecessors(u):
                if str(H.nodes[pred].get("type", "")).lower() == "process":
                    edges_to_add.append((v, pred))
                    edges_to_remove.append((u, v))

    H.remove_edges_from(edges_to_remove)
    H.add_edges_from(edges_to_add)
    return H


# ---------------- UI ----------------
page = st.sidebar.selectbox(
    "Go to",
    ["Home & Data Input", "Edit Model (CRUD)", "Graph View", "Analysis Report"]
)

# ---------------- HOME & DATA INPUT ----------------
if page == "Home & Data Input":
    st.header("PPR Management Dashboard 🏠")

    # 1. KPI Dashboard Segment
    if "nodes_df" in st.session_state and not st.session_state.nodes_df.empty:
        df = st.session_state.nodes_df
        type_series = df["type"].astype(str).str.strip().str.lower() if "type" in df.columns else pd.Series([],
                                                                                                            dtype=str)

        st.subheader("System Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total Products", int((type_series == "product").sum()))
        col2.metric("⚙️ Total Processes", int((type_series == "process").sum()))
        col3.metric("🛠️ Total Resources", int((type_series == "resource").sum()))

        st.divider()
        st.subheader("Key Performance Indicators (KPIs)")

        total_cost = pd.to_numeric(df.get("cost", pd.Series()), errors="coerce").sum()
        total_time = pd.to_numeric(df.get("time_min", pd.Series()), errors="coerce").sum()
        total_co2 = pd.to_numeric(df.get("co2", pd.Series()), errors="coerce").sum()

        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("💰 Total System Cost", f"${total_cost:,.2f}")
        kpi2.metric("⏱️ Total Prod. Time", f"{total_time:,.1f} mins")
        kpi3.metric("🌱 Est. CO2 Footprint", f"{total_co2:,.2f} kg")

    else:
        st.info("👋 Welcome! Please upload your PPR Excel file below to generate your dashboard.")

    st.divider()

    # 2. Upload Segment
    st.header("Upload Excel Data")
    file = st.file_uploader("Upload your PPR model", type=["xlsx"])

    if file:
        try:
            xls = pd.ExcelFile(file)
            sheet_names = list(xls.sheet_names)

            df_products = load_sheet(xls, ["product"])
            df_processes = load_sheet(xls, ["process"])
            df_resources = load_sheet(xls, ["resource", "resources", "resour"])
            df_mapping = load_sheet(xls, ["mapping", "connection"])

            if df_products is None or df_mapping is None:
                st.error("Missing required sheets")
                st.stop()

            df_products = clean_columns(df_products)
            df_mapping = clean_columns(df_mapping)

            if df_processes is not None:
                df_processes = clean_columns(df_processes)

            if df_resources is not None:
                df_resources = clean_columns(df_resources)
                if df_resources.empty:
                    df_resources = None
            else:
                df_resources = load_resource_sheet_fallback(xls)
                if df_resources is not None:
                    df_resources = clean_columns(df_resources)
                    if df_resources.empty:
                        df_resources = None

            nodes = []

            # ---- Products ----
            for _, r in df_products.iterrows():
                nodes.append(
                    {"id": r.get("product id", r.get("product name")), "name": r.get("product", r.get("product name")),
                     "type": "Product", **r})

            # ---- Processes ----
            if df_processes is not None:
                for _, r in df_processes.iterrows():
                    nodes.append({"id": r.get("process id", r.get("process name")),
                                  "name": r.get("process", r.get("process name")), "type": "Process", **r})

            # ---- Resources ----
            if df_resources is not None:
                resource_id_col = find_column(df_resources, ["resource id", "resource_id", "id"])
                resource_name_col = find_column(df_resources, ["resource name", "resource_name"])

                for _, r in df_resources.iterrows():
                    resource_id = r.get(resource_id_col) if resource_id_col else None
                    resource_name = r.get(resource_name_col) if resource_name_col else None
                    if pd.isna(resource_id) and pd.isna(resource_name): continue

                    # Safely extract the data without overriding the core 'type'
                    row_data = r.dropna().to_dict()
                    excel_type = row_data.pop("type", "Resource")

                    nodes.append({
                        "id": resource_id if pd.notna(resource_id) else resource_name,
                        "name": resource_name if pd.notna(resource_name) else resource_id,
                        "type": "Resource",
                        "resource_category": excel_type,
                        **row_data
                    })

            # ---- Edges Mapping ---- (This is what accidentally got deleted!)
            edges = []
            src = find_column(df_mapping, ["source", "from"])
            tgt = find_column(df_mapping, ["target", "to"])
            if src is None or tgt is None:
                st.error("Mapping sheet must contain source/from and target/to columns.")
                st.stop()

            for _, r in df_mapping.iterrows():
                s_raw, t_raw = r[src], r[tgt]
                if pd.notna(s_raw) and pd.notna(t_raw):
                    s, t = str(s_raw).strip().lower(), str(t_raw).strip().lower()
                    if not s or not t or s == "nan" or t == "nan": continue
                    edges.append({"source": s, "target": t})

            # Fallback for missing resources
            known_aliases = {str(n.get("id", "")).strip().lower() for n in nodes} | {
                str(n.get("name", "")).strip().lower() for n in nodes}
            for endpoint in {e["source"] for e in edges} | {e["target"] for e in edges}:
                ep = str(endpoint).strip().lower()
                if ep and ep != "nan" and ep not in known_aliases:
                    nodes.append({"id": ep, "name": ep, "type": "Resource"})
                    known_aliases.add(ep)

            st.session_state.nodes_df = ensure_node_columns(pd.DataFrame(nodes))
            st.session_state.edges_df = ensure_edge_columns(pd.DataFrame(edges, columns=["source", "target"]))
            st.success("Data loaded successfully! 🚀 Dashboard KPIs are now populated at the top.")

            if st.button("Refresh Dashboard"):
                st.rerun()

        except Exception as e:
            st.error(e)

# ---------------- EDIT MODEL (CRUD) ----------------
elif page == "Edit Model (CRUD)":
    st.header("Modify PPR Model")
    if "nodes_df" not in st.session_state or st.session_state.nodes_df.empty:
        st.warning("Please load data in the 'Home & Data Input' tab first.")
        st.stop()

    view = st.selectbox("Select View Context (for editing specific attributes)",
                        ["Engineering", "Quality", "Sustainability", "Reliability"])
    st.markdown("---")
    render_crud_main_page(view)


# ---------------- GRAPH VIEW ----------------
elif page == "Graph View":
    graph_tab()

# ---------------- ANALYSIS REPORT ----------------
elif page == "Analysis Report":
    st.header("Diagnostic Dashboard & Requirement Analysis")

    if "nodes_df" not in st.session_state or st.session_state.nodes_df is None or st.session_state.nodes_df.empty:
        st.warning("Please load data in the 'Home & Data Input' tab first.")
        st.stop()

    G = build_networkx_graph()

    # --- AUTO-HEAL TOGGLE ---
    st.markdown("---")
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.write("**Graph Semantic Auto-Healer**")
        st.caption(
            "Detects logical errors in Excel mapping (e.g., connecting a Resource to a Product instead of a Process) and dynamically rewires the graph topology.")
    with col_t2:
        heal_toggle = st.checkbox("⚙️ Activate Auto-Heal", value=False)

    if heal_toggle:
        G = auto_heal_graph(G)
        st.success("Graph topology successfully healed! Check the diagnostic updates below.")

    violations_req = check_process_resource_assignment(G)
    disconnected_req = find_disconnected_segments(G)

    node_names = sorted([str(d.get("name", n)).title() for n, d in G.nodes(data=True)])

    st.subheader("Simulate Failure (Downstream Impact)")
    selected_fail = st.selectbox("Select a node to simulate a quality failure:", ["-- None --"] + node_names)

    impact_req = []
    if selected_fail != "-- None --":
        impact_req = analyze_downstream_quality_impact(G, selected_fail)

    st.divider()

    st.subheader("System Health Overview (Visual Tree)")
    st.markdown(
        "**Legend:** 🔴 **Red:** Process missing Resource | 🟡 **Yellow:** Disconnected Segment | 🟠 **Orange:** Downstream Impact | ⚪ **Grey:** Healthy")

    # Unpack both graphs
    diagnostic_tree_main, diagnostic_tree_iso = build_diagnostic_tree(G, violations_req, disconnected_req, impact_req)

    # Create an asymmetrical side-by-side layout (Main gets 75% space, Orphans get 25%)
    col_g1, col_g2 = st.columns([3, 1])

    with col_g1:
        st.write("**Main Assembly Pipeline**")
        try:
            st.graphviz_chart(diagnostic_tree_main, use_container_width=True)
        except Exception as render_err:
            st.error(f"Graphviz Rendering Engine Failed. Error details: {render_err}")

    with col_g2:
        st.write("**Disconnected Components**")
        if disconnected_req:
            try:
                st.graphviz_chart(diagnostic_tree_iso, use_container_width=True)
            except Exception as render_err:
                st.error(f"Graphviz Rendering Engine Failed. Error details: {render_err}")
        else:
            st.success("No orphaned elements detected!")

    st.divider()

    st.subheader("Detailed Diagnostic Reports")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("##### Process-Resource Allocation")
        if violations_req:
            formatted_v = [str(v).title() for v in violations_req]
            st.error(f"Missing Resources for {len(formatted_v)} processes: {', '.join(formatted_v)}")
        else:
            st.success("All processes have resources correctly assigned.")

        st.write("##### Network Connectivity")
        if disconnected_req:
            st.warning(f"Found {len(disconnected_req)} isolated network segments.")
            for i, comp in enumerate(disconnected_req):
                formatted_c = [str(name).title().replace("Nan", "NAN") for name in comp]
                st.caption(f"Segment {i + 1}: {', '.join(formatted_c)}")
        else:
            st.success("System is fully connected.")

    with col_b:
        st.write("##### Downstream Impact Analysis")
        if selected_fail != "-- None --":
            if impact_req:
                formatted_i = [str(i).title() for i in impact_req]
                st.warning(f"Failure in '{selected_fail}' impacts {len(formatted_i)} nodes: {', '.join(formatted_i)}")
            else:
                st.info("No downstream impact detected. (End of the line)")
        else:
            st.info("Select a node above to simulate failure.")

        st.write("##### Structural Similarity")
        similar = find_similarly_structured_processes(G)
        if similar:
            for sig, nodes in similar.items():
                formatted_n = [str(n).title() for n in nodes]
                st.info(f"Structure [{sig.replace('_', ' ')}]: {', '.join(formatted_n)}")
        else:
            st.write("No structurally identical processes found.")
