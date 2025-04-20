def fc_graph(n):
    adj_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0 if i == j else 1)
        adj_matrix.append(row)
    return adj_matrix


def changeAdj(actions, original_adj_matrix, edge_index=None):
    new_adj = [row.copy() for row in original_adj_matrix]

    for i, (action, edge) in enumerate(zip(actions, zip(*edge_index))):
        src, dst = edge
        if src < dst:
            if action == 1:
                new_adj[src][dst] = 0
                new_adj[dst][src] = 0
    return new_adj
