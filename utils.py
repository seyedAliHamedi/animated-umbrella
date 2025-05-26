import random
import numpy as np
import networkx as nx


def changeAdj(actions, original_adj_matrix):
    new_adj = [row.copy() for row in original_adj_matrix]
    for index, action in enumerate(actions):
        if action:
            new_adj[index] = [0] * len(new_adj[index])
            for row in new_adj:
                row[index] = 0
    return new_adj


def get_gw(adj_matrix, n_clients, n_servers):
    available_gateways = list(range(len(adj_matrix)))

    client_gateways = random.sample(available_gateways, n_clients)
    remaining_gateways = [
        gw for gw in available_gateways if gw not in client_gateways]
    server_gateways = random.sample(remaining_gateways, n_servers)

    return client_gateways, server_gateways


def get_state(adj_matrix, client_gw, servers_gw, original):
    graph_metrics = collect_graph_metrics(adj_matrix, original)
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
                }
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


def collect_graph_metrics(adj_matrix, original_adj_matrix):
    # Convert to numpy arrays once
    current_array = np.array(adj_matrix)
    original_array = np.array(original_adj_matrix)

    # Cache graph creation
    current_graph = nx.from_numpy_array(current_array)
    original_graph = nx.from_numpy_array(original_array)

    # Pre-calculate connected status
    original_connected = nx.is_connected(original_graph)
    current_connected = nx.is_connected(current_graph)

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
    }
    return metrics
