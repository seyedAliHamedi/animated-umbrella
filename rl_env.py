from ns import ns
import numpy as np
import pandas as pd
import traceback
import networkx as nx

from sim.topology import Topology
from sim.app import App
from sim.monitor import Monitor


class NetworkEnv:

    def __init__(self, adj_matrix, simulation_duration=100, min_throughput=1.0, max_latency=100.0, max_packet_loss=0.1, router_energy_cost=10, link_energy_cost=2):

        self.adj_matrix = adj_matrix
        self.simulation_duration = simulation_duration
        self.current_step = 0

        self.min_throughput = min_throughput
        self.max_latency = max_latency
        self.max_packet_loss = max_packet_loss

        self.router_energy_cost = router_energy_cost
        self.link_energy_cost = link_energy_cost

        self.setup_environment()

    def setup_environment(self):

        self.topology = Topology(
            adj_matrix=self.adj_matrix, base_network="192.168.1.0/24")

        self.active_links = [1] * self.topology.N_links
        self.active_routers = [1] * self.topology.N_routers

        self.app = App(self.topology)
        self.app.monitor = Monitor(
            self.topology.nodes,
            self.app
        )

        self.app.monitor = Monitor(self.app.topology, self.app)

        anim = self.app.monitor.setup_animation(self.app.animFile)
        self.app.monitor.setup_pcap_capture()
        self.app.monitor.setup_packet_log()
        self.app.monitor.setup_flow_monitor()
        self.app.monitor.position_nodes(anim)

    def reset(self):
        print("reset environment")
        ns.Simulator.Destroy()
        self.setup_environment()

    def step(self):
        self.run_simulation(self.simulation_duration)

        metrics = self.collect_metrics()
        reward = self.calculate_reward(metrics)

        return metrics, reward

    def run_simulation(self, duration):
        print(f"Running simulation for {duration} seconds...")
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()

        self.app.monitor.get_node_ips_by_id()
        self.app.monitor.trace_routes()

        self.app.monitor.collect_flow_stats(
            app_port=self.app.app_port, filter_noise=True)
        self.app.monitor.get_packet_logs()
        print("Simulation completed")

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
        return -5

    def calculate_energy(self):
        return 5

    def collect_graph_metrics(self):
        original_graph = nx.from_numpy_array(
            np.array(self.topology.adj_matrix))

        current_graph = nx.Graph()

        active_nodes = [i for i in range(
            self.topology.N_routers) if self.active_routers[i]]
        current_graph.add_nodes_from(active_nodes)

        for i in range(self.topology.N_routers):
            for j in range(i+1, self.topology.N_routers):
                if (self.topology.adj_matrix[i][j] == 1 and
                        self.active_routers[i] and self.active_routers[j]):
                    current_graph.add_edge(i, j)

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

    def collect_node_metrics(self, node_idx):
        node_metrics = {
            'is_active': self.active_routers[node_idx],

            'avg_power_per_operation': np.random.uniform(0.5, 2.0),
            'avg_energy_consumption': np.random.uniform(5.0, 20.0),
            'idle_interface_energy': np.random.uniform(0.1, 0.5),

            'tx_packets': 0,
            'tx_bytes': 0,
            'rx_packets': 0,
            'rx_bytes': 0,

            'lost_on_send': 0,
            'lost_on_receive': 0,

            'expected_packets': {
                'high_priority': np.random.randint(10, 50),
                'medium_priority': np.random.randint(5, 30),
                'low_priority': np.random.randint(1, 20)
            },

            "active_interfaces": 0,

            'graph_metrics': {
                'betweenness_centrality': {
                    'original': 0,
                    'current': 0
                },
                'degree_centrality': {
                    'original': 0,
                    'current': 0
                },
                'clustering_coefficient': {
                    'original': 0,
                    'current': 0
                },
                'eigenvector_centrality': {
                    'original': 0,
                    'current': 0
                },
                'is_articulation_point': {
                    'original': False,
                    'current': False
                }
            },


        }

        df = pd.read_csv('./sim/monitor/logs/packets_log.csv')
        df = df[(df["Port"] == 9) | (df["Port"] == 49153)]

        packet_summary = df.groupby(
            ["Packet", "Port", "total_hops"]).size().reset_index(name="count")

        packet_summary["status"] = packet_summary.apply(
            lambda row: "OK" if row["count"] % (2 * row['total_hops']) == 0 else "LOST", axis=1
        )
        df = df.merge(
            packet_summary[['Packet', 'Port', 'total_hops', 'status']],
            on=['Packet', 'Port', 'total_hops'],
            how='left'
        )

        node_tx = df[(df['Node'] == node_idx) & (df['Direction'] == 'TX')]
        node_rx = df[(df['Node'] == node_idx) & (df['Direction'] == 'RX')]

        node_metrics['tx_packets'] = len(node_tx)
        node_metrics['tx_bytes'] = int(node_tx['Size'].sum())
        node_metrics['rx_packets'] = len(node_rx)
        node_metrics['rx_bytes'] = int(node_rx['Size'].sum())

        lost_packets_df = df[df['status'] == 'LOST']

        node_metrics['lost_on_send'] = len(
            lost_packets_df[lost_packets_df['Node'] == node_idx])
        node_metrics['lost_on_receive'] = len(
            lost_packets_df[lost_packets_df['next_hop'] == node_idx])

        node = self.topology.nodes.Get(node_idx)
        ipv4 = node.GetObject[ns.Ipv4]()

        if ipv4:
            active_interfaces = []
            for i in range(ipv4.GetNInterfaces()):
                if ipv4.IsUp(i):
                    active_interfaces.append(i)

        node_metrics['active_interfaces'] = len(active_interfaces)
        graph_metrics = self.collect_graph_metrics()

        node_metrics['graph_metrics']['betweenness_centrality']['original'] = (
            graph_metrics['betweenness_centrality']['original'].get(
                node_idx, 0)
        )
        node_metrics['graph_metrics']['betweenness_centrality']['current'] = (
            graph_metrics['betweenness_centrality']['current'].get(
                node_idx, 0)
        )

        node_metrics['graph_metrics']['degree_centrality']['original'] = (
            graph_metrics['degree_centrality']['original'].get(node_idx, 0)
        )
        node_metrics['graph_metrics']['degree_centrality']['current'] = (
            graph_metrics['degree_centrality']['current'].get(node_idx, 0)
        )

        node_metrics['graph_metrics']['clustering_coefficient']['original'] = (
            graph_metrics['clustering_coefficient']['original'].get(
                node_idx, 0)
        )
        node_metrics['graph_metrics']['clustering_coefficient']['current'] = (
            graph_metrics['clustering_coefficient']['current'].get(
                node_idx, 0)
        )

        node_metrics['graph_metrics']['eigenvector_centrality']['original'] = (
            graph_metrics.get('eigenvector_centrality', {}).get(
                'original', {}).get(node_idx, 0)
        )
        node_metrics['graph_metrics']['eigenvector_centrality']['current'] = (
            graph_metrics.get('eigenvector_centrality', {}).get(
                'current', {}).get(node_idx, 0)
        )

        # Check if node is an articulation point
        node_metrics['graph_metrics']['is_articulation_point']['original'] = (
            node_idx in graph_metrics['articulation_points']['original']
        )
        node_metrics['graph_metrics']['is_articulation_point']['current'] = (
            node_idx in graph_metrics['articulation_points']['current']
        )

        return node_metrics

    def collect_metrics(self):
        all_node_metrics = {}
        for index in range(self.topology.N_routers):
            all_node_metrics[index] = self.collect_node_metrics(index)

        return all_node_metrics
