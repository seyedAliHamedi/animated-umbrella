from ns import ns
import numpy as np
import pandas as pd
import traceback
import networkx as nx

from sim.topology import Topology
from sim.app import App
from sim.monitor import Monitor


class NetworkEnv:

    def __init__(self, adj_matrix, n_clients, original_adj_matrix, n_servers, simulation_duration=100, min_throughput=1.0, max_latency=100.0, max_packet_loss=0.1, router_energy_cost=10, link_energy_cost=2):

        self.adj_matrix = adj_matrix
        self.simulation_duration = simulation_duration
        self.current_step = 0

        self.min_throughput = min_throughput
        self.max_latency = max_latency
        self.max_packet_loss = max_packet_loss

        self.n_clients = n_clients
        self.n_servers = n_servers
        self.original_adj_matrix = original_adj_matrix

        self.router_energy_cost = router_energy_cost
        self.link_energy_cost = link_energy_cost

        self.inter_info = {}

        self.setup_environment()

    def setup_environment(self):

        self.topology = Topology(adj_matrix=self.adj_matrix)

        self.active_routers = []
        for i in range(self.topology.N_routers):
            flag = False
            for j in range(self.topology.N_routers):
                if self.adj_matrix[i][j] == 1:
                    flag = True
                    break

            self.active_routers.append(1 if flag else 0)

        self.active_links = {
            i: int(sum(self.adj_matrix[i]))
            for i in range(self.topology.N_routers)
        }

        # for i in range(self.topology.N_routers):
        #     for j in range(i+1, self.topology.N_routers):
        #         if self.adj_matrix[i][j] == 1:
        #             self.active_links.append(1)
        #         else:
        #             self.active_links.append(0)

        self.app = App(self.topology, app_interval=1, n_clients=self.n_clients, n_servers=self.n_servers,
                       app_duration=self.simulation_duration)

        self.app.monitor = Monitor(
            self.topology.nodes,
            self.app
        )

        self.app.monitor = Monitor(self.app.topology, self.app)

        mobility = ns.MobilityHelper()
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.app.topology.nodes)
        mobility.Install(self.app.clients)
        mobility.Install(self.app.servers)

        anim = self.app.monitor.setup_animation(self.app.animFile)
        # self.app.monitor.setup_pcap_capture()
        self.app.monitor.setup_packet_log()
        self.app.monitor.setup_flow_monitor()
        # self.app.monitor.position_nodes(anim)

    def reset(self):
        print("reset environment")
        ns.Simulator.Destroy()
        self.setup_environment()

    def step(self):
        self.run_simulation(self.simulation_duration)

        metrics = self.collect_metrics()
        self.collect_edge_features()
        # print(self.inter_info)
        print("-" * 30)
        self.app.monitor.collect_flow_stats()
        print("-" * 30)
        reward = self.calculate_reward(metrics)

        return metrics, reward

    def run_simulation(self, duration):
        # print(f"Running simulation for {duration} seconds...")
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()

        self.app.monitor.get_node_ips_by_id()
        self.app.monitor.trace_routes()

        self.app.monitor.collect_flow_stats(
            app_port=self.app.app_port, filter_noise=True)
        self.app.monitor.get_packet_logs()
        # print("Simulation completed")

    # def set_interface_state(self, r, interface_index, state):
    #     """
    #     Set the state of router interfaces.

    #     Parameters:
    #     r - The router node
    #     interface_index - The interface to modify (-1 for all interfaces)
    #     state - Boolean (True = UP, False = DOWN)
    #     """
    #     try:
    #         # Get IPv4 stack safely
    #         ipv4 = None
    #         try:
    #             ipv4 = r.GetObject[ns.Ipv4]()
    #         except Exception as e:
    #             print(f"Failed to get IPv4 object: {e}")
    #             return

    #         if ipv4 is None:
    #             print(f"Warning: No IPv4 stack on router {r.GetId()}")
    #             return

    #         num_interfaces = ipv4.GetNInterfaces()  # Get total interfaces

    #         if interface_index == -1:
    #             # Apply to all interfaces (except loopback, usually index 0)
    #             for i in range(1, num_interfaces):
    #                 try:
    #                     if state:
    #                         ipv4.SetUp(i)
    #                     else:
    #                         ipv4.SetDown(i)

    #                     # Get routing protocol safely
    #                     routing_protocol = None
    #                     try:
    #                         routing_protocol = ipv4.GetRoutingProtocol()
    #                     except Exception as e:
    #                         print(f"Failed to get routing protocol: {e}")
    #                         continue

    #                     if routing_protocol:
    #                         if not state:
    #                             routing_protocol.NotifyInterfaceDown(i)
    #                         else:
    #                             routing_protocol.NotifyInterfaceUp(i)
    #                 except Exception as e:
    #                     print(f"Error setting interface {i} state: {e}")

    #             print(
    #                 f"Router {r.GetId()} {'enabled' if state else 'disabled'} (all {num_interfaces-1} interfaces)")
    #         else:
    #             # Apply only to the specified interface
    #             if 0 < interface_index < num_interfaces:
    #                 try:
    #                     if state:
    #                         ipv4.SetUp(interface_index)
    #                     else:
    #                         ipv4.SetDown(interface_index)

    #                     # Get routing protocol safely
    #                     routing_protocol = None
    #                     try:
    #                         routing_protocol = ipv4.GetRoutingProtocol()
    #                     except Exception as e:
    #                         print(f"Failed to get routing protocol: {e}")
    #                         return

    #                     if routing_protocol:
    #                         if not state:
    #                             routing_protocol.NotifyInterfaceDown(
    #                                 interface_index)
    #                         else:
    #                             routing_protocol.NotifyInterfaceUp(
    #                                 interface_index)
    #                 except Exception as e:
    #                     print(
    #                         f"Error setting interface {interface_index} state: {e}")

    #                 print(
    #                     f"Router {r.GetId()} {'enabled' if state else 'disabled'} (interface {interface_index})")
    #             else:
    #                 print(
    #                     f"Invalid interface index {interface_index} for router {r.GetId()}")
    #     except Exception as e:
    #         print(f"Error in set_interface_state: {e}")
    #         traceback.print_exc()

    def calculate_reward(self, metrics):
        r = 0
        n_total = sum(info["max_packets"]
                      for info in self.app.client_info.values())
        n_failed = sum(info["failed"]
                       for info in self.app.client_info.values())

        if n_failed > 0:
            r = -1 * (n_failed / n_total)
        else:
            r = 1 + len(self.active_routers) / sum(self.active_routers)
            # r =  (np.exp(-len(self.active_routers)+sum(self.active_routers)))
        return r

    def calculate_energy(self, edge_features):
        return 5

    def collect_graph_metrics(self):
        current_graph = nx.from_numpy_array(
            np.array(self.topology.adj_matrix))

        original_graph = nx.from_numpy_array(
            np.array(self.original_adj_matrix))

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
                    'diameter': nx.diameter(original_graph) if nx.is_connected(original_graph) else float('inf'),
                    'radius': nx.radius(original_graph) if nx.is_connected(original_graph) else float('inf'),
                    'is_connected': nx.is_connected(original_graph),
                    'number_of_components': nx.number_connected_components(original_graph)
                },
                'current': {
                    'diameter': nx.diameter(current_graph) if nx.is_connected(current_graph) else float('inf'),
                    'radius': nx.radius(current_graph) if nx.is_connected(current_graph) else float('inf'),
                    'is_connected': nx.is_connected(current_graph),
                    'number_of_components': nx.number_connected_components(current_graph)
                }
            },

        }
        return metrics

    def collect_metrics(self):
        """Collect metrics for all nodes efficiently."""
        # Get graph metrics once
        graph_metrics = self.collect_graph_metrics()

        # Process packet data once instead of per node
        packet_data = self.process_packet_data_once()

        # Create metrics for each node
        all_node_metrics = {}
        for index in range(self.topology.N_routers):
            all_node_metrics[index] = self.collect_node_metrics(
                index, graph_metrics, packet_data)
        return all_node_metrics

    def process_packet_data_once(self):
        df = pd.read_csv('./sim/monitor/logs/packets_log.csv')
        df = df[(df["Port"] == 9) | (df["Port"] == 49153)]

        df['Time'] = df['Time'].astype(float)

        packet_summary = df.groupby(
            ["Packet", "Port", "total_hops"]).size().reset_index(name="count")

        packet_summary["status"] = packet_summary.apply(
            lambda row: "LOST" if row["total_hops"] == 0 else (
                "OK" if row["count"] % (2 * row["total_hops"]) == 0 else "LOST"
            ),
            axis=1
        )

        df = df.merge(
            packet_summary[['Packet', 'Port', 'total_hops', 'status']],
            on=['Packet', 'Port', 'total_hops'],
            how='left'
        )

        node_data = {}
        lost_packets_df = df[df['status'] == 'LOST']

        for node_idx in range(self.topology.N_routers):
            node_tx = df[(df['Node'] == node_idx) & (df['Direction'] == 'TX')]
            node_rx = df[(df['Node'] == node_idx) & (df['Direction'] == 'RX')]

            total_transmit_time = 0
            total_receive_time = 0
            transmit_count = 0
            receive_count = 0

            for _, tx_row in node_tx.iterrows():
                packet_id = tx_row['Packet']
                next_hop = tx_row['next_hop']

                if next_hop != "Null" and next_hop != "Server" and next_hop != "Client":
                    next_hop = int(next_hop)
                    rx_at_next = df[(df['Node'] == next_hop) &
                                    (df['Packet'] == packet_id) &
                                    (df['Direction'] == 'RX')]

                    if not rx_at_next.empty:
                        tx_time = tx_row['Time']
                        rx_time = rx_at_next.iloc[0]['Time']
                        transmit_time = rx_time - tx_time

                        if transmit_time > 0:  # Sanity check
                            total_transmit_time += transmit_time
                            transmit_count += 1

            for _, rx_row in node_rx.iterrows():
                packet_id = rx_row['Packet']
                prev_hop = rx_row['prev_hop']

                if prev_hop != "Null" and prev_hop != "Server" and prev_hop != "Client":
                    prev_hop = int(prev_hop)
                    tx_at_prev = df[(df['Node'] == prev_hop) &
                                    (df['Packet'] == packet_id) &
                                    (df['Direction'] == 'TX')]

                    if not tx_at_prev.empty:
                        rx_time = rx_row['Time']
                        tx_time = tx_at_prev.iloc[0]['Time']
                        receive_time = rx_time - tx_time

                        if receive_time > 0:  # Sanity check
                            total_receive_time += receive_time
                            receive_count += 1

            avg_transmit_time = total_transmit_time / \
                transmit_count if transmit_count > 0 else 0
            avg_receive_time = total_receive_time / \
                receive_count if receive_count > 0 else 0

            node_data[node_idx] = {
                'tx_packets': len(node_tx),
                'tx_bytes': int(node_tx['Size'].sum()) if not node_tx.empty else 0,
                'rx_packets': len(node_rx),
                'rx_bytes': int(node_rx['Size'].sum()) if not node_rx.empty else 0,
                'lost_on_send': len(lost_packets_df[lost_packets_df['Node'] == node_idx]),
                'lost_on_receive': len(lost_packets_df[lost_packets_df['next_hop'] == node_idx]),
                'total_transmit_time': float(total_transmit_time),
                'avg_transmit_time': float(avg_transmit_time),
                'total_receive_time': float(total_receive_time),
                'avg_receive_time': float(avg_receive_time),
            }

        return node_data

    def collect_node_metrics(self, node_idx, graph_metrics, packet_data):
        node_metrics = {
            'is_active': self.active_routers[node_idx],
            'is_client_server': 1 if node_idx in self.app.client_gateways or node_idx in self.app.server_gateways else 0,
            'avg_power_per_operation': np.random.uniform(0.5, 2.0),
            'avg_energy_consumption': np.random.uniform(5.0, 20.0),
            'idle_interface_energy': np.random.uniform(0.1, 0.5),


            # Use pre-calculated packet data
            'tx_packets': packet_data[node_idx]['tx_packets'],
            'tx_bytes': packet_data[node_idx]['tx_bytes'],
            'rx_packets': packet_data[node_idx]['rx_packets'],
            'rx_bytes': packet_data[node_idx]['rx_bytes'],
            'lost_on_send': packet_data[node_idx]['lost_on_send'],
            'lost_on_receive': packet_data[node_idx]['lost_on_receive'],
            'total_transmit_time': packet_data[node_idx]['total_transmit_time'],
            'avg_transmit_time': packet_data[node_idx]['avg_transmit_time'],
            'total_receive_time': packet_data[node_idx]['total_receive_time'],
            'avg_receive_time': packet_data[node_idx]['avg_receive_time'],

            'expected_packets': {
                'high_priority': np.random.randint(10, 50),
                'medium_priority': np.random.randint(5, 30),
                'low_priority': np.random.randint(1, 20)
            },
            'active_interfaces': self.active_links[node_idx],

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

        node = self.topology.nodes.Get(node_idx)
        ipv4 = node.GetObject[ns.Ipv4]()

        if ipv4:
            active_interfaces = sum(1 for i in range(
                ipv4.GetNInterfaces()) if ipv4.IsUp(i))
            node_metrics['active_interfaces'] = active_interfaces

        return node_metrics

    def collect_edge_features(self):
        df = pd.read_csv('./sim/monitor/logs/packets_log.csv')
        df = df[(df["Port"] == 9) | (df["Port"] == 49153)]
        df['Time'] = df['Time'].astype(float)

        for i in range(self.topology.N_routers):
            for j in range(self.topology.N_routers):
                if i == j:
                    continue
                edge_id = (i, j)

                is_active = self.adj_matrix[i][j] == 1

                # Ensure these variables are always defined
                latencies_tx = []
                latencies_rx = []
                tx_bytes = 0
                rx_bytes = 0
                packets_i_to_j = pd.DataFrame()
                packets_j_to_i = pd.DataFrame()

                if is_active:
                    packets_i_to_j = df[(df['Node'] == i) & (
                        df['next_hop'] == str(j))]
                    packets_j_to_i = df[(df['Node'] == j) & (
                        df['next_hop'] == str(i))]

                    tx_bytes = int(packets_i_to_j['Size'].sum())
                    rx_bytes = int(packets_j_to_i['Size'].sum())

                    for _, tx_row in packets_i_to_j.iterrows():
                        packet_id = tx_row['Packet']
                        rx_packets = df[(df['Node'] == j) & (df['Packet'] == packet_id) &
                                        (df['Direction'] == 'RX') & (df['Time'] > tx_row['Time'])]
                        if not rx_packets.empty:
                            tx_time = tx_row['Time']
                            rx_time = rx_packets.iloc[0]['Time']
                            latency = rx_time - tx_time
                            if latency > 0:
                                latencies_tx.append(latency)

                    for _, tx_row in packets_j_to_i.iterrows():
                        packet_id = tx_row['Packet']
                        rx_packets = df[(df['Node'] == i) & (df['Packet'] == packet_id) &
                                        (df['Direction'] == 'RX') & (df['Time'] > tx_row['Time'])]
                        if not rx_packets.empty:
                            tx_time = tx_row['Time']
                            rx_time = rx_packets.iloc[0]['Time']
                            latency = rx_time - tx_time
                            if latency > 0:
                                latencies_rx.append(latency)

                total_time_tx = sum(latencies_tx)
                avg_time_tx = total_time_tx / \
                    len(latencies_tx) if len(latencies_tx) > 0 else 0

                total_time_rx = sum(latencies_rx)
                avg_time_rx = total_time_rx / \
                    len(latencies_rx) if len(latencies_rx) > 0 else 0

                # print(f"Edge {edge_id}: tx_bytes={tx_bytes}, rx_bytes={rx_bytes}, total_time_tx={total_time_tx}, avg_time_tx={avg_time_tx}, total_time_rx={total_time_rx}, avg_time_rx={avg_time_rx}")
                print(
                    f"Edge {edge_id}: tx_ps={len(packets_i_to_j)}, rx_ps={len(packets_j_to_i)}, total_time_tx={total_time_tx}, total_time_rx={total_time_rx}")

                if is_active:
                    self.inter_info[edge_id] = {
                        'node': i,
                        'interface': edge_id,
                        'is_active': is_active,
                        'tx_bytes': tx_bytes,
                        'rx_bytes': rx_bytes,
                        'total_time_tx': total_time_tx,
                        'avg_time_tx': avg_time_tx,
                        'total_time_rx': total_time_rx,
                        'avg_time_rx': avg_time_rx,
                        'tx_packets': len(packets_i_to_j),
                        'rx_packets': len(packets_j_to_i),
                        # 'lost_tx': len(packets_i_to_j[packets_i_to_j['status'] == 'LOST']),
                        # 'lost_rx': len(packets_j_to_i[packets_j_to_i['status'] == 'LOST'])
                    }
                else:
                    self.inter_info[edge_id] = {
                        'node': i,
                        'interface': edge_id,
                        'is_active': 0,
                        'tx_bytes': 0,
                        'rx_bytes': 0,
                        'total_time_tx': 0,
                        'avg_time_tx': 0,
                        'total_time_rx': 0,
                        'avg_time_rx': 0,
                        'tx_packets': 0,
                        'rx_packets': 0,
                        'lost_tx': 0,
                        'lost_rx': 0
                    }
