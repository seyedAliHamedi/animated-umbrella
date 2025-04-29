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


def generate_ip_node_mappings(adj_matrix, n_clients, n_servers):
    ip_to_node = {}
    node_to_ip = {}

    for i in range(len(adj_matrix)):
        node_to_ip[i] = []
        for j in range(i, len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                ip_i = f"1.{i}.{j}.1"
                ip_j = f"1.{i}.{j}.2"

                ip_to_node[ip_i] = i
                ip_to_node[ip_j] = j

                node_to_ip[i].append(ip_i)

                if j not in node_to_ip:
                    node_to_ip[j] = []
                node_to_ip[j].append(ip_j)

    for client_id in range(len(adj_matrix), len(adj_matrix)+n_clients):
        for gateway_idx in range(len(adj_matrix)):

            client_ip = f"111.111.{gateway_idx}.1"
            gateway_ip = f"111.111.{gateway_idx}.2"

            ip_to_node[client_ip] = client_id
            ip_to_node[gateway_ip] = gateway_idx

            if client_id not in node_to_ip:
                node_to_ip[client_id] = []
            node_to_ip[client_id].append(client_ip)

            if gateway_idx not in node_to_ip:
                node_to_ip[gateway_idx] = []
            node_to_ip[gateway_idx].append(gateway_ip)

    for server_id in range(len(adj_matrix)+n_clients, len(adj_matrix)+n_clients+n_servers):
        for gateway_idx in range(len(adj_matrix)):

            server_ip = f"222.222.{gateway_idx}.1"
            gateway_ip = f"222.222.{gateway_idx}.2"

            ip_to_node[server_ip] = server_id
            ip_to_node[gateway_ip] = gateway_idx

            if server_id not in node_to_ip:
                node_to_ip[server_id] = []
            node_to_ip[server_id].append(server_ip)

            if gateway_idx not in node_to_ip:
                node_to_ip[gateway_idx] = []
            node_to_ip[gateway_idx].append(gateway_ip)

    return ip_to_node, node_to_ip
