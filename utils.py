import random


def fc_graph(n):
    adj_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0 if i == j else 1)
        adj_matrix.append(row)
    return adj_matrix


def changeAdj(actions, original_adj_matrix):
    new_adj = [row.copy() for row in original_adj_matrix]
    for index, action in enumerate(actions):
        if action:
            new_adj[index] = [0] * len(new_adj[index])
            for row in new_adj:
                row[index] = 0
    return new_adj


def get_gw(adj_matrix, n_clients, n_servers):
    available_gateways = list(range(len(adj_matrix)))[:-1]

    client_gateways = random.sample(available_gateways, n_clients)
    remaining_gateways = [
        gw for gw in available_gateways if gw not in client_gateways]
    server_gateways = random.sample(remaining_gateways, n_servers)

    return client_gateways, server_gateways


def get_state(adj_matrix, client_gw, servers_gw):
    all_node_state = []
    for index in range(len(adj_matrix)):
        node_state = {
            # 'is_active': sum(adj_matrix[index]) > 0,
            'is_client_server': 1 if index in client_gw or index in servers_gw else 0,
        }
        all_node_state.append(node_state)
    return all_node_state
