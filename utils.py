import random
import numpy as np
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
import csv
import re
import pandas as pd


def compute_fbc(adj_matrix, flows, K=2, normalise=True):
    """
    Near-shortest Flow-Betweenness Centrality (FBC).

    Parameters
    ----------
    adj_matrix : array-like (N×N)
        Binary / weight-agnostic adjacency matrix.
    flows : list[(s, t)]
        Ordered source–target node pairs to evaluate.
    K : int, default 2
        Allow paths up to (d_min + K) hops.
    normalise : bool, default True
        If True, divide final scores by the number of valid (s, t) pairs
        so every value lies in [0, 1].

    Returns
    -------
    dict {node: score}
        FBC score for every vertex in the graph.
    """
    # ---------- 1) Build an unweighted graph ----------
    A = np.asarray(adj_matrix)
    N = A.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:                       # non-zero → edge
                G.add_edge(i, j)

    # ---------- 2) Initialise output ----------
    flow_bc = {v: 0.0 for v in G.nodes()}
    valid_pairs = 0                           # we'll count only pairs with a path

    # ---------- 3) Process each source–target pair ----------
    for (s, t) in flows:
        if s == t:
            continue

        # 3a) hop-count shortest path length
        try:
            d_min = nx.shortest_path_length(G, s, t)
        except nx.NetworkXNoPath:
            continue                          # unreachable pair → skip

        valid_pairs += 1

        # 3b) enumerate simple paths in non-decreasing hop length order
        from networkx.algorithms.simple_paths import shortest_simple_paths
        all_paths = shortest_simple_paths(G, s, t)

        # 3c) collect near-shortest paths and their weights
        path_list, weight_list = [], []
        for pi in all_paths:
            hops = len(pi) - 1
            if hops > d_min + K:
                break                         # generator is already ordered
            w = 1.0 / ((hops - d_min) + 1)    # geometric decay
            path_list.append(pi)
            weight_list.append(w)

        if not path_list:                     # defensive, shouldn't happen
            continue

        # 3d) normalise weights so pair contributes exactly 1.0 in total
        total_w = sum(weight_list)
        norm_w = [w / total_w for w in weight_list]

        # 3e) add contribution to interior vertices
        for pi, w in zip(path_list, norm_w):
            for v in pi[1:-1]:
                flow_bc[v] += w

    # ---------- 4) Optional normalisation ----------
    if normalise and valid_pairs:
        flow_bc = {v: bc / valid_pairs for v, bc in flow_bc.items()}

    return flow_bc


def changeAdj(actions, original_adj_matrix):
    new_adj = [row.copy() for row in original_adj_matrix]
    for index, action in enumerate(actions):
        if action:
            new_adj[index] = [0] * len(new_adj[index])
            for row in new_adj:
                row[index] = 0
    return new_adj


def get_gw(adj_matrix, n_clients, n_servers):
    n = len(adj_matrix)
    m = n_clients 
    if m > n or (n <= 1 and m > 0):
        return [], []
    
    nodes = list(range(n))
    random.shuffle(nodes)
    
    clients_gw = [nodes[i] for i in range(m)]
    servers_gw = [nodes[(i + 1) % n] for i in range(m)]
    
    return clients_gw, servers_gw


# Global variable to store RTT table - loaded once
_RTT_TABLE_CACHE = None


def load_rtt_table_from_csv(filename='ping.csv'):
    """
    Load RTT table from CSV file and cache it globally

    Parameters:
    -----------
    filename : str
        Input CSV filename

    Returns:
    --------
    dict : RTT table in the format {(src, dst): {'avg': float, 'min': float, 'max': float, 'rtts': list}}
    """
    global _RTT_TABLE_CACHE

    if _RTT_TABLE_CACHE is not None:
        return _RTT_TABLE_CACHE

    rtt_table = {}

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            src = int(row['src'])
            dst = int(row['dst'])

            # Parse RTT values
            rtts = []
            i = 0
            while f'rtt_{i}' in row and row[f'rtt_{i}']:
                rtts.append(float(row[f'rtt_{i}']))
                i += 1

            # Handle infinity values
            avg_val = float('inf') if row['avg_rtt'] == 'inf' else float(
                row['avg_rtt'])
            min_val = float('inf') if row['min_rtt'] == 'inf' else float(
                row['min_rtt'])
            max_val = float('inf') if row['max_rtt'] == 'inf' else float(
                row['max_rtt'])

            rtt_table[(src, dst)] = {
                'avg': avg_val,
                'min': min_val,
                'max': max_val,
                'rtts': rtts
            }

    _RTT_TABLE_CACHE = rtt_table
    return rtt_table


def extract_rtt_features_from_table(rtt_table, adj_matrix, gateways=None, fbc_scores=None, fbc_threshold=0, norm_mode=0, timeout_ms=1000.0):
    """
    Extract RTT features using precomputed RTT table

    Parameters:
    -----------
    rtt_table : dict
        Precomputed RTT measurements
    adj_matrix : array-like
        Adjacency matrix to determine neighbors
    gateways : dict
        Dict with 'source' and 'dest' keys for separated mode
    fbc_scores : dict
        Flow Betweenness Centrality scores for each node
    fbc_threshold : float
        Threshold for FBC - nodes with FBC <= threshold get penalty values
    norm_mode : int
        0 = No normalization
        1 = Min-Max normalization
        2 = ICMP timeout normalization
    timeout_ms : float
        ICMP timeout in milliseconds (used only when norm_mode=2)

    Returns:
    --------
    dict : Features for each node
    """
    num_nodes = len(adj_matrix)
    features = {}

    for node in range(num_nodes):
        # Get neighbor RTTs from table
        neighbors = []
        neighbor_rtts = []

        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] == 1:
                neighbors.append(neighbor)

                # Get RTTs from precomputed table
                if (node, neighbor) in rtt_table and rtt_table[(node, neighbor)]['rtts']:
                    neighbor_rtts.extend(rtt_table[(node, neighbor)]['rtts'])

        # Calculate neighbor features
        if neighbor_rtts:
            features[node] = {
                'avg_rtt_neigh': np.mean(neighbor_rtts),
                'min_rtt_neigh': np.min(neighbor_rtts),
                'max_rtt_neigh': np.max(neighbor_rtts),
                'neighbors': neighbors,
                'num_neighbors': len(neighbors)
            }
        else:
            features[node] = {
                'avg_rtt_neigh': None,
                'min_rtt_neigh': None,
                'max_rtt_neigh': None,
                'neighbors': neighbors,
                'num_neighbors': 0
            }

        # Add FBC value if computed
        if fbc_scores is not None:
            features[node]['fbc'] = fbc_scores.get(node, 0.0)

        # Calculate gateway features if gateways are provided
        if gateways is not None and 'source' in gateways and 'dest' in gateways:
            source_gateways = gateways['source']
            dest_gateways = gateways['dest']

            # Check if this node is a gateway itself
            is_source_gateway = node in source_gateways
            is_dest_gateway = node in dest_gateways

            # Calculate source gateway features
            if is_source_gateway and not (fbc_scores is not None and fbc_scores.get(node, 0.0) > 0):
                # Pure source gateway
                features[node]['min_rtt_to_src'] = 0.0
                features[node]['max_rtt_to_src'] = 0.0
                features[node]['rtt_ratio_src'] = 1.0
            else:
                # Calculate RTT to source gateways from table
                src_rtts = []
                for gw in source_gateways:
                    if gw != node and (node, gw) in rtt_table and rtt_table[(node, gw)]['rtts']:
                        src_rtts.extend(rtt_table[(node, gw)]['rtts'])

                if src_rtts:
                    min_rtt_src = np.min(src_rtts)
                    max_rtt_src = np.max(src_rtts)
                    features[node]['min_rtt_to_src'] = min_rtt_src
                    features[node]['max_rtt_to_src'] = max_rtt_src
                    features[node]['rtt_ratio_src'] = min_rtt_src / \
                        max_rtt_src if max_rtt_src > 0 else 1.0
                else:
                    features[node]['min_rtt_to_src'] = 0.0 if is_source_gateway else 1000.0
                    features[node]['max_rtt_to_src'] = 0.0 if is_source_gateway else 1000.0
                    features[node]['rtt_ratio_src'] = 1.0

            # Calculate destination gateway features
            if is_dest_gateway and not (fbc_scores is not None and fbc_scores.get(node, 0.0) > 0):
                # Pure destination gateway
                features[node]['min_rtt_to_dst'] = 0.0
                features[node]['max_rtt_to_dst'] = 0.0
                features[node]['rtt_ratio_dst'] = 1.0
            else:
                # Calculate RTT to destination gateways from table
                dst_rtts = []
                for gw in dest_gateways:
                    if gw != node and (node, gw) in rtt_table and rtt_table[(node, gw)]['rtts']:
                        dst_rtts.extend(rtt_table[(node, gw)]['rtts'])

                if dst_rtts:
                    min_rtt_dst = np.min(dst_rtts)
                    max_rtt_dst = np.max(dst_rtts)
                    features[node]['min_rtt_to_dst'] = min_rtt_dst
                    features[node]['max_rtt_to_dst'] = max_rtt_dst
                    features[node]['rtt_ratio_dst'] = min_rtt_dst / \
                        max_rtt_dst if max_rtt_dst > 0 else 1.0
                else:
                    features[node]['min_rtt_to_dst'] = 0.0 if is_dest_gateway else 1000.0
                    features[node]['max_rtt_to_dst'] = 0.0 if is_dest_gateway else 1000.0
                    features[node]['rtt_ratio_dst'] = 1.0

            # Apply FBC threshold penalties
            if fbc_scores is not None:
                node_fbc = fbc_scores.get(node, 0.0)
                if node_fbc <= fbc_threshold and not (is_source_gateway or is_dest_gateway):
                    features[node]['min_rtt_to_src'] = 1000.0
                    features[node]['max_rtt_to_src'] = 1000.0
                    features[node]['rtt_ratio_src'] = 1.0
                    features[node]['min_rtt_to_dst'] = 1000.0
                    features[node]['max_rtt_to_dst'] = 1000.0
                    features[node]['rtt_ratio_dst'] = 1.0

    # Apply normalization if requested
    if norm_mode == 1:
        return normalize_rtt_features(features, fbc_threshold, mode=0, timeout_ms=timeout_ms)
    elif norm_mode == 2:
        return normalize_rtt_features(features, fbc_threshold, mode=1, timeout_ms=timeout_ms)
    else:
        return features


def normalize_rtt_features(features, fbc_threshold=0, mode=1, timeout_ms=1000.0):
    """
    Normalize RTT features for GNN input

    Parameters:
    -----------
    features : dict
        Raw features from extract_rtt_features_from_table
    fbc_threshold : float
        FBC threshold used in feature extraction
    mode : int
        0 = Min-Max normalization across nodes
        1 = Timeout-based normalization (default)
    timeout_ms : float
        ICMP timeout in milliseconds (used only when mode=1)

    Returns:
    --------
    dict : Normalized features
    """
    normalized = {}

    if mode == 0:
        # Min-Max Normalization Mode
        # Collect all values for normalization (excluding None and penalty values)
        all_avg_neigh = []
        all_min_neigh = []
        all_max_neigh = []
        all_min_gw = []
        all_max_gw = []
        all_ratio_gw = []
        all_fbc = []

        for node, f in features.items():
            # Neighbor features (always included)
            if f['avg_rtt_neigh'] is not None:
                all_avg_neigh.append(f['avg_rtt_neigh'])
            if f['min_rtt_neigh'] is not None:
                all_min_neigh.append(f['min_rtt_neigh'])
            if f['max_rtt_neigh'] is not None:
                all_max_neigh.append(f['max_rtt_neigh'])

            # Gateway features (source and dest)
            for key in ['min_rtt_to_src', 'max_rtt_to_src', 'min_rtt_to_dst', 'max_rtt_to_dst']:
                if key in f and f[key] is not None and f[key] < 1000:
                    all_min_gw.append(
                        f[key]) if 'min' in key else all_max_gw.append(f[key])

            # Ratio features
            for key in ['rtt_ratio_src', 'rtt_ratio_dst']:
                if key in f and f[key] is not None:
                    all_ratio_gw.append(f[key])

            # FBC scores
            if 'fbc' in f:
                all_fbc.append(f['fbc'])

        # Calculate min/max for normalization
        def get_min_max(values):
            if not values:
                return 0, 1
            return 0, max(values)  # Always use 0 as minimum

        min_avg_n, max_avg_n = get_min_max(all_avg_neigh)
        min_min_n, max_min_n = get_min_max(all_min_neigh)
        min_max_n, max_max_n = get_min_max(all_max_neigh)

        # Use combined max for all gateway RTTs
        all_gw_values = all_min_gw + all_max_gw
        max_gw_overall = max(all_gw_values) if all_gw_values else 1

        # For ratio, keep the original min/max since it's already 0-1
        min_ratio = min(all_ratio_gw) if all_ratio_gw else 0
        max_ratio = max(all_ratio_gw) if all_ratio_gw else 1

        # For FBC, use 0 as minimum
        min_fbc = 0
        max_fbc = max(all_fbc) if all_fbc else 1

        # Normalize each node
        for node, f in features.items():
            norm_f = {
                'neighbors': f['neighbors'],
                'num_neighbors': f['num_neighbors']
            }

            # Normalize neighbor features (0-1 range)
            if f['avg_rtt_neigh'] is not None:
                norm_f['avg_rtt_neigh'] = f['avg_rtt_neigh'] / \
                    max_avg_n if max_avg_n > 0 else 0.0
            else:
                norm_f['avg_rtt_neigh'] = 0.5

            if f['min_rtt_neigh'] is not None:
                norm_f['min_rtt_neigh'] = f['min_rtt_neigh'] / \
                    max_min_n if max_min_n > 0 else 0.0
            else:
                norm_f['min_rtt_neigh'] = 0.5

            if f['max_rtt_neigh'] is not None:
                norm_f['max_rtt_neigh'] = f['max_rtt_neigh'] / \
                    max_max_n if max_max_n > 0 else 0.0
            else:
                norm_f['max_rtt_neigh'] = 0.5

            # Normalize FBC (0 to max)
            if 'fbc' in f:
                norm_f['fbc'] = f['fbc'] / max_fbc if max_fbc > 0 else 0.0
                node_fbc = f['fbc']
            else:
                norm_f['fbc'] = 0.0
                node_fbc = 0.0

            # Handle separated source/dest features
            if 'min_rtt_to_src' in f:
                # Source gateway features
                if f['min_rtt_to_src'] == 0.0 and f['max_rtt_to_src'] == 0.0:
                    norm_f['min_rtt_to_src'] = 0.0
                    norm_f['max_rtt_to_src'] = 0.0
                    norm_f['rtt_ratio_src'] = 1.0
                elif f['min_rtt_to_src'] >= 1000:
                    norm_f['min_rtt_to_src'] = 1.0
                    norm_f['max_rtt_to_src'] = 1.0
                    norm_f['rtt_ratio_src'] = 1.0
                else:
                    norm_f['min_rtt_to_src'] = f['min_rtt_to_src'] / \
                        max_gw_overall
                    norm_f['max_rtt_to_src'] = f['max_rtt_to_src'] / \
                        max_gw_overall
                    norm_f['rtt_ratio_src'] = f['rtt_ratio_src'] if f.get(
                        'rtt_ratio_src') is not None else 1.0

            if 'min_rtt_to_dst' in f:
                # Destination gateway features
                if f['min_rtt_to_dst'] == 0.0 and f['max_rtt_to_dst'] == 0.0:
                    norm_f['min_rtt_to_dst'] = 0.0
                    norm_f['max_rtt_to_dst'] = 0.0
                    norm_f['rtt_ratio_dst'] = 1.0
                elif f['min_rtt_to_dst'] >= 1000:
                    norm_f['min_rtt_to_dst'] = 1.0
                    norm_f['max_rtt_to_dst'] = 1.0
                    norm_f['rtt_ratio_dst'] = 1.0
                else:
                    norm_f['min_rtt_to_dst'] = f['min_rtt_to_dst'] / \
                        max_gw_overall
                    norm_f['max_rtt_to_dst'] = f['max_rtt_to_dst'] / \
                        max_gw_overall
                    norm_f['rtt_ratio_dst'] = f['rtt_ratio_dst'] if f.get(
                        'rtt_ratio_dst') is not None else 1.0

            normalized[node] = norm_f

    else:  # mode == 1 (Timeout-based normalization)
        # Similar logic for timeout-based normalization
        # ... (implementation similar to above but using timeout_ms for normalization)
        pass

    return normalized


def get_state(adj_matrix, client_gw, servers_gw, original, flows_by_qos=None):
    graph_metrics = collect_graph_metrics(
        adj_matrix, original, client_gw, servers_gw, flows_by_qos=flows_by_qos)
    all_node_state = []
    for node_idx in range(len(adj_matrix)):
        node_state = {
            'is_client_server': 1 if node_idx in client_gw or node_idx in servers_gw else 0,
            'graph_metrics': {
                'betweenness_centrality': {
                    'original': graph_metrics['betweenness_centrality']['original'].get(node_idx, 0),
                    'current': graph_metrics['betweenness_centrality']['current'].get(node_idx, 0)
                },
                'degree_centrality': {
                    'original': graph_metrics['degree_centrality']['original'].get(node_idx, 0),
                    'current': graph_metrics['degree_centrality']['current'].get(node_idx, 0)
                },
                'clustering_coefficient': {
                    'original': graph_metrics['clustering_coefficient']['original'].get(node_idx, 0),
                    'current': graph_metrics['clustering_coefficient']['current'].get(node_idx, 0)
                },
                'eigenvector_centrality': {
                    'original': graph_metrics.get('eigenvector_centrality', {}).get('original', {}).get(node_idx, 0),
                    'current': graph_metrics.get('eigenvector_centrality', {}).get('current', {}).get(node_idx, 0)
                },
                'is_articulation_point': {
                    'original': node_idx in graph_metrics['articulation_points']['original'],
                    'current': node_idx in graph_metrics['articulation_points']['current']
                },
                'flow_betweenness_centrality': {
                    'original': graph_metrics['flow_betweenness_centrality']['original'].get(node_idx, 0),
                    'current': graph_metrics['flow_betweenness_centrality']['current'].get(node_idx, 0)
                },
                # Per-QoS FBC features (4 new features)
                'fbc_Interactive_Web': graph_metrics['fbc_per_qos'].get('Interactive_Web', {}).get(node_idx, 0.0),
                'fbc_Streaming_Media': graph_metrics['fbc_per_qos'].get('Streaming_Media', {}).get(node_idx, 0.0),
                'fbc_Background_Sync': graph_metrics['fbc_per_qos'].get('Background_Sync', {}).get(node_idx, 0.0),
                'fbc_Real_Time_Interactive': graph_metrics['fbc_per_qos'].get('Real_Time_Interactive', {}).get(node_idx, 0.0),
                # Add RTT features
                'avg_rtt_neigh': graph_metrics['rtt_features'].get(node_idx, {}).get('avg_rtt_neigh', 1),
                'min_rtt_neigh': graph_metrics['rtt_features'].get(node_idx, {}).get('min_rtt_neigh', 1),
                'max_rtt_neigh': graph_metrics['rtt_features'].get(node_idx, {}).get('max_rtt_neigh', 1),
                'min_rtt_to_src': graph_metrics['rtt_features'].get(node_idx, {}).get('min_rtt_to_src', 1.0),
                'max_rtt_to_src': graph_metrics['rtt_features'].get(node_idx, {}).get('max_rtt_to_src', 1.0),
                'rtt_ratio_src': graph_metrics['rtt_features'].get(node_idx, {}).get('rtt_ratio_src', 1.0),
                'min_rtt_to_dst': graph_metrics['rtt_features'].get(node_idx, {}).get('min_rtt_to_dst', 1.0),
                'max_rtt_to_dst': graph_metrics['rtt_features'].get(node_idx, {}).get('max_rtt_to_dst', 1.0),
                'rtt_ratio_dst': graph_metrics['rtt_features'].get(node_idx, {}).get('rtt_ratio_dst', 1.0),
            }
        }
        all_node_state.append(node_state)
    return all_node_state


def generate_ip_node_mappings(adj_matrix, n_clients, n_servers):
    ip_to_node = {}
    node_to_ip = {}

    # Pre-allocate node_to_ip for all nodes
    n_total = len(adj_matrix) + n_clients + n_servers
    for i in range(n_total):
        node_to_ip[i] = []

    # Router-to-router mappings
    for i in range(len(adj_matrix)):
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                ip_i = f"1.{i}.{j}.1"
                ip_j = f"1.{i}.{j}.2"

                ip_to_node[ip_i] = i
                ip_to_node[ip_j] = j

                node_to_ip[i].append(ip_i)
                node_to_ip[j].append(ip_j)

    # Client mappings
    for client_id in range(len(adj_matrix), len(adj_matrix)+n_clients):
        for gateway_idx in range(len(adj_matrix)):
            client_ip = f"111.111.{gateway_idx}.1"
            gateway_ip = f"111.111.{gateway_idx}.2"

            ip_to_node[client_ip] = client_id
            ip_to_node[gateway_ip] = gateway_idx

            node_to_ip[client_id].append(client_ip)
            node_to_ip[gateway_idx].append(gateway_ip)

    # Server mappings
    for server_id in range(len(adj_matrix)+n_clients, len(adj_matrix)+n_clients+n_servers):
        for gateway_idx in range(len(adj_matrix)):
            server_ip = f"222.222.{gateway_idx}.1"
            gateway_ip = f"222.222.{gateway_idx}.2"

            ip_to_node[server_ip] = server_id
            ip_to_node[gateway_ip] = gateway_idx

            node_to_ip[server_id].append(server_ip)
            node_to_ip[gateway_idx].append(gateway_ip)

    return ip_to_node, node_to_ip


def collect_graph_metrics(adj_matrix, original_adj_matrix, client_gateways=None, server_gateways=None, flows_by_qos=None):
    # Convert to numpy arrays once
    current_array = np.array(adj_matrix)
    original_array = np.array(original_adj_matrix)

    # Cache graph creation
    current_graph = nx.from_numpy_array(current_array)
    original_graph = nx.from_numpy_array(original_array)

    # Pre-calculate connected status
    original_connected = nx.is_connected(original_graph)
    current_connected = nx.is_connected(current_graph)

    # Generate flows from client gateways to server gateways (paired)
    flows = []
    if client_gateways is not None and server_gateways is not None:
        n_clients = len(client_gateways)
        n_servers = len(server_gateways)
        for i in range(n_clients):
            # Each client i connects to server (i % n_servers)
            client = client_gateways[i]
            server_idx = i % n_servers
            server = server_gateways[server_idx]
            if client != server:  # Avoid self-loops
                flows.append((client, server))

    # Calculate FBC for both graphs (total across all flows)
    if flows:
        fbc_original = compute_fbc(original_adj_matrix, flows, K=2)
        fbc_current = compute_fbc(adj_matrix, flows, K=2)
    else:
        # If no flows provided, initialize with zeros
        fbc_original = {i: 0.0 for i in range(len(original_adj_matrix))}
        fbc_current = {i: 0.0 for i in range(len(adj_matrix))}

    # Calculate per-QoS FBC (only for original topology - current changes each step)
    if flows_by_qos is not None:
        fbc_per_qos_original = compute_fbc_per_qos(original_adj_matrix, flows_by_qos, K=2)
    else:
        # Default: all zeros for each QoS type
        fbc_per_qos_original = {qos: {i: 0.0 for i in range(len(original_adj_matrix))} for qos in QOS_TYPES}

    # Load RTT table and extract features
    rtt_table = load_rtt_table_from_csv('ping.csv')

    # Prepare gateways in the expected format
    gateways = {
        'source': client_gateways if client_gateways else [],
        'dest': server_gateways if server_gateways else []
    }

    # Extract RTT features with normalization
    rtt_features = extract_rtt_features_from_table(
        rtt_table=rtt_table,
        adj_matrix=original_array,  # Use current topology for neighbor detection
        gateways=gateways,
        fbc_scores=fbc_original,  # Use current FBC scores
        fbc_threshold=0.0,
        norm_mode=1  # Min-max normalization
    )

    metrics = {
        'betweenness_centrality': {
            'original': dict(nx.betweenness_centrality(original_graph)),
            'current': dict(nx.betweenness_centrality(current_graph))
        },
        'degree_centrality': {
            'original': dict(nx.degree_centrality(original_graph)),
            'current': dict(nx.degree_centrality(current_graph))
        },
        'clustering_coefficient': {
            'original': dict(nx.clustering(original_graph)),
            'current': dict(nx.clustering(current_graph))
        },
        'articulation_points': {
            'original': list(nx.articulation_points(original_graph)),
            'current': list(nx.articulation_points(current_graph))
        },
        'graph_metrics': {
            'original': {
                'diameter': nx.diameter(original_graph) if original_connected else float('inf'),
                'radius': nx.radius(original_graph) if original_connected else float('inf'),
                'is_connected': original_connected,
                'number_of_components': nx.number_connected_components(original_graph)
            },
            'current': {
                'diameter': nx.diameter(current_graph) if current_connected else float('inf'),
                'radius': nx.radius(current_graph) if current_connected else float('inf'),
                'is_connected': current_connected,
                'number_of_components': nx.number_connected_components(current_graph)
            }
        },
        'flow_betweenness_centrality': {
            'original': fbc_original,
            'current': fbc_current
        },
        'fbc_per_qos': fbc_per_qos_original,  # Per-QoS FBC scores
        'rtt_features': rtt_features  # Add RTT features to metrics
    }
    return metrics


# =============================================================================
# Traffic Feature Extraction for Traffic-Aware Agent
# =============================================================================

# QoS type mapping (must match mawi_q_list keys in sim/utils.py)
QOS_TYPES = ['Interactive_Web', 'Streaming_Media', 'Background_Sync', 'Real_Time_Interactive']


def compute_fbc_per_qos(adj_matrix, flows_by_qos, K=2):
    """
    Compute Flow Betweenness Centrality separately for each QoS type.

    Parameters
    ----------
    adj_matrix : array-like (N×N)
        Binary adjacency matrix.
    flows_by_qos : dict {qos_type: [(src, dst), ...]}
        Flows grouped by QoS type.
    K : int
        Allow paths up to (d_min + K) hops.

    Returns
    -------
    dict {qos_type: {node: fbc_score}}
        FBC scores per node for each QoS type.
    """
    fbc_per_qos = {}

    for qos_type in QOS_TYPES:
        flows = flows_by_qos.get(qos_type, [])
        if flows:
            fbc_per_qos[qos_type] = compute_fbc(adj_matrix, flows, K=K, normalise=True)
        else:
            # No flows of this type - all nodes get 0
            N = len(adj_matrix)
            fbc_per_qos[qos_type] = {i: 0.0 for i in range(N)}

    return fbc_per_qos


def extract_flows_by_qos(row, client_gateways, server_gateways):
    """
    Extract flows grouped by QoS type from a MAWI CSV row.

    Parameters
    ----------
    row : pd.Series
        Processed row from MAWI CSV (after process_row())
    client_gateways : list
        List of client gateway nodes
    server_gateways : list
        List of server gateway nodes

    Returns
    -------
    dict {qos_type: [(src, dst), ...]}
        Flows (as gateway pairs) grouped by QoS type.
    """
    flows_by_qos = {qos: [] for qos in QOS_TYPES}

    # Find all flow columns
    flow_cols = [c for c in row.index if c.startswith('F') and c.endswith('/T')]

    for col in flow_cols:
        flow_num = col.replace('/T', '')  # e.g., 'F1'
        flow_idx = int(flow_num[1:]) - 1  # 'F1' -> 0, 'F2' -> 1, etc.

        # Skip if flow index exceeds gateways
        if flow_idx >= len(client_gateways) or flow_idx >= len(server_gateways):
            continue

        throughput = row.get(col, 0)
        packets = row.get(f'{flow_num}/P', 0)

        # Skip zero flows
        if throughput == 0 or packets == 0:
            continue

        q_type_col = f'{flow_num}/q_type'
        if q_type_col not in row.index:
            continue

        q_type = row[q_type_col]
        if q_type not in QOS_TYPES:
            continue

        # Map flow to gateway pair
        src_gw = client_gateways[flow_idx]
        dst_gw = server_gateways[flow_idx % len(server_gateways)]

        if src_gw != dst_gw:
            flows_by_qos[q_type].append((src_gw, dst_gw))

    return flows_by_qos


def load_traffic_norm_constants(json_path='./traffic_norm_constants.json'):
    """
    Load pre-computed normalization constants from JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file with normalization constants

    Returns
    -------
    dict : Normalization constants for each feature per QoS type
    """
    import json

    with open(json_path, 'r') as f:
        norm_constants = json.load(f)

    return norm_constants


def compute_traffic_norm_constants(csv_path, percentile=99):
    """
    Scan MAWI CSV once to compute normalization constants using percentile values.

    Parameters
    ----------
    csv_path : str
        Path to the MAWI CSV file (e.g., TL_MAWI_balanced.csv)
    percentile : int
        Percentile to use for normalization (default 99 to avoid outliers)

    Returns
    -------
    dict : Normalization constants for each feature per QoS type
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Initialize collectors per QoS type
    stats = {qos: {
        'n_flows': [],
        'n_packets': [],
        'avg_packet_size': [],
        'packet_rate': []
    } for qos in QOS_TYPES}

    # Also track total packets for traffic intensity calculation
    total_packets_per_row = []

    # Process each row
    for idx, row in df.iterrows():
        # Find all flow columns
        flow_cols = [c for c in df.columns if c.startswith('F') and c.endswith('/T')]

        # Count flows per QoS type for this row
        qos_counts = {qos: 0 for qos in QOS_TYPES}
        qos_packets = {qos: 0 for qos in QOS_TYPES}
        qos_packet_sizes = {qos: [] for qos in QOS_TYPES}
        qos_intervals = {qos: [] for qos in QOS_TYPES}

        row_total_packets = 0

        for col in flow_cols:
            flow_num = col.replace('/T', '')  # e.g., 'F1'
            throughput = row.get(col, 0)
            packets = row.get(f'{flow_num}/P', 0)

            # Skip zero/NaN flows
            if pd.isna(throughput) or pd.isna(packets) or throughput == 0 or packets == 0:
                continue

            q_type_col = f'{flow_num}/q_type'
            if q_type_col not in row:
                continue

            q_type = row[q_type_col]
            if pd.isna(q_type) or q_type not in QOS_TYPES:
                continue

            qos_counts[q_type] += 1
            qos_packets[q_type] += int(packets)
            row_total_packets += int(packets)

            # Packet size
            pkt_size = row.get(f'{flow_num}/Avg_packet_size', 0)
            if not pd.isna(pkt_size) and pkt_size > 0:
                qos_packet_sizes[q_type].append(pkt_size)

            # Interval (for packet rate)
            interval = row.get(f'{flow_num}/interval', 0)
            if not pd.isna(interval) and interval > 0:
                qos_intervals[q_type].append(interval)

        total_packets_per_row.append(row_total_packets)

        # Aggregate stats for this row
        for qos in QOS_TYPES:
            if qos_counts[qos] > 0:
                stats[qos]['n_flows'].append(qos_counts[qos])
                stats[qos]['n_packets'].append(qos_packets[qos])

                if qos_packet_sizes[qos]:
                    stats[qos]['avg_packet_size'].append(np.mean(qos_packet_sizes[qos]))

                if qos_intervals[qos]:
                    # Packet rate = 1 / average_interval
                    avg_interval = np.mean(qos_intervals[qos])
                    if avg_interval > 0:
                        stats[qos]['packet_rate'].append(1.0 / avg_interval)

    # Compute percentile values for normalization
    norm_constants = {}

    for qos in QOS_TYPES:
        norm_constants[qos] = {}
        for feature in ['n_flows', 'n_packets', 'avg_packet_size', 'packet_rate']:
            values = stats[qos][feature]
            if values:
                norm_constants[qos][feature] = float(np.percentile(values, percentile))
            else:
                # Default values if no data
                norm_constants[qos][feature] = 1.0

    # Add total packets normalization for traffic intensity
    if total_packets_per_row:
        norm_constants['max_total_packets'] = float(np.percentile(total_packets_per_row, percentile))
    else:
        norm_constants['max_total_packets'] = 1.0

    # Add strictness normalization (max is 1/min_sla_delay)
    # Real_Time_Interactive has sla_delay=0.080, so max strictness = 1/0.08 = 12.5
    norm_constants['max_strictness'] = 12.5

    return norm_constants


def get_traffic_features(row, norm_constants, qos_profiles):
    """
    Extract normalized traffic features from a MAWI CSV row.

    Parameters
    ----------
    row : pd.Series
        Processed row from MAWI CSV (after process_row())
    norm_constants : dict
        Normalization constants from compute_traffic_norm_constants()
    qos_profiles : dict
        QoS profiles dict (mawi_q_list from sim/utils.py)

    Returns
    -------
    torch.Tensor : [4, 8] tensor of normalized traffic features
        Rows: QoS types (Interactive_Web, Streaming_Media, Background_Sync, Real_Time_Interactive)
        Cols: n_flows, n_packets, avg_packet_size, packet_rate, w_delay, w_jitter, w_loss, strictness
    """
    import torch

    # Initialize feature matrix [4 QoS types, 8 features]
    features = torch.zeros(4, 8, dtype=torch.float32)

    # Find all flow columns in this row
    flow_cols = [c for c in row.index if c.startswith('F') and c.endswith('/T')]

    # Aggregate per QoS type
    qos_data = {qos: {
        'n_flows': 0,
        'n_packets': 0,
        'packet_sizes': [],
        'intervals': []
    } for qos in QOS_TYPES}

    for col in flow_cols:
        flow_num = col.replace('/T', '')  # e.g., 'F1'
        throughput = row.get(col, 0)
        packets = row.get(f'{flow_num}/P', 0)

        # Skip NaN or zero flows
        if pd.isna(throughput) or pd.isna(packets) or throughput == 0 or packets == 0:
            continue

        q_type_col = f'{flow_num}/q_type'
        if q_type_col not in row.index:
            continue

        q_type = row[q_type_col]
        if q_type not in QOS_TYPES:
            continue

        qos_data[q_type]['n_flows'] += 1
        qos_data[q_type]['n_packets'] += int(packets)

        # Packet size
        pkt_size = row.get(f'{flow_num}/Avg_packet_size', 0)
        if not pd.isna(pkt_size) and pkt_size > 0:
            qos_data[q_type]['packet_sizes'].append(pkt_size)

        # Interval
        interval = row.get(f'{flow_num}/interval', 0)
        if not pd.isna(interval) and interval > 0:
            qos_data[q_type]['intervals'].append(interval)

    # Build feature tensor
    for qos_idx, qos in enumerate(QOS_TYPES):
        data = qos_data[qos]
        qos_norm = norm_constants.get(qos, {})
        qos_profile = qos_profiles.get(qos, {})

        # Feature 0: n_flows (normalized)
        max_flows = qos_norm.get('n_flows', 1.0)
        features[qos_idx, 0] = min(1.0, data['n_flows'] / max_flows) if max_flows > 0 else 0.0

        # Feature 1: n_packets (normalized)
        max_packets = qos_norm.get('n_packets', 1.0)
        features[qos_idx, 1] = min(1.0, data['n_packets'] / max_packets) if max_packets > 0 else 0.0

        # Feature 2: avg_packet_size (normalized)
        if data['packet_sizes']:
            avg_pkt_size = np.mean(data['packet_sizes'])
            max_pkt_size = qos_norm.get('avg_packet_size', 1500.0)
            features[qos_idx, 2] = min(1.0, avg_pkt_size / max_pkt_size) if max_pkt_size > 0 else 0.0
        else:
            features[qos_idx, 2] = 0.0

        # Feature 3: packet_rate (normalized)
        if data['intervals']:
            avg_interval = np.mean(data['intervals'])
            packet_rate = 1.0 / avg_interval if avg_interval > 0 else 0.0
            max_rate = qos_norm.get('packet_rate', 1.0)
            features[qos_idx, 3] = min(1.0, packet_rate / max_rate) if max_rate > 0 else 0.0
        else:
            features[qos_idx, 3] = 0.0

        # Feature 4: w_delay (already in [0, 1])
        features[qos_idx, 4] = float(qos_profile.get('w_d', 0.0))

        # Feature 5: w_jitter (already in [0, 1])
        features[qos_idx, 5] = float(qos_profile.get('w_j', 0.0))

        # Feature 6: w_loss (already in [0, 1])
        features[qos_idx, 6] = float(qos_profile.get('w_l', 0.0))

        # Feature 7: strictness = 1/sla_delay (normalized)
        sla_delay = qos_profile.get('sla_delay', 1.0)
        strictness = 1.0 / sla_delay if sla_delay > 0 else 0.0
        max_strictness = norm_constants.get('max_strictness', 12.5)
        features[qos_idx, 7] = min(1.0, strictness / max_strictness)

    return features


def compute_traffic_intensity(row, norm_constants):
    """
    Compute traffic intensity for adaptive reward calculation.

    Parameters
    ----------
    row : pd.Series
        Processed row from MAWI CSV
    norm_constants : dict
        Normalization constants

    Returns
    -------
    float : Traffic intensity in [0, 1] range
    """
    # Sum all packets in this row
    flow_cols = [c for c in row.index if c.startswith('F') and c.endswith('/P')]
    total_packets = sum(row.get(col, 0) for col in flow_cols if row.get(col, 0) > 0)

    max_packets = norm_constants.get('max_total_packets', 1.0)
    return min(1.0, total_packets / max_packets) if max_packets > 0 else 0.0
