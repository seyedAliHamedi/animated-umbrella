def fc_graph(n):
    adj_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0 if i == j else 1)
        adj_matrix.append(row)
    return adj_matrix


def changeAdj(actions, adj_matrix):
    for index, action in enumerate(actions):
        if action:
            adj_matrix[index] = [0]*len(adj_matrix[index])
            for row in adj_matrix:
                row[index] = 0
    return adj_matrix
