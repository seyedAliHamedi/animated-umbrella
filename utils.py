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
