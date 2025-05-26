import time
from ns import ns
import numpy as np
import pandas as pd
import networkx as nx
from sim.utils import *
from sim.app import App
from sim.monitor import Monitor
from sim.topology import Topology


class NetworkEnv:

    def __init__(self,
                 adj_matrix,
                 n_clients,
                 original_adj_matrix,
                 n_servers,
                 client_gateways,
                 server_gateways,
                 ip_to_node,
                 node_to_ip,
                 simulation_duration=100,
                 ):

        self.adj_matrix = adj_matrix
        self.simulation_duration = simulation_duration

        self.n_clients = n_clients
        self.n_servers = n_servers
        self.original_adj_matrix = original_adj_matrix

        self.client_gateways = client_gateways
        self.server_gateways = server_gateways
        self.inter_info = {}
        self.ip_to_node = ip_to_node
        self.node_to_ip = node_to_ip
        self.setup_environment()

        self.router_type = {
            i: sample_data["routers"][i % len(sample_data["routers"])]
            for i in range(self.topology.N_routers)
        }

    def setup_environment(self):
        self.topology = Topology(adj_matrix=self.adj_matrix)

        self.active_routers = []
        self.active_links = {}

        for i in range(self.topology.N_routers):
            row_sum = sum(self.adj_matrix[i])
            self.active_routers.append(1 if row_sum > 0 else 0)
            self.active_links[i] = int(row_sum)

        self.app = App(self.topology, client_gateways=self.client_gateways, server_gateways=self.server_gateways, app_interval=1, n_clients=self.n_clients, n_servers=self.n_servers, app_start_time=40,
                       app_duration=self.simulation_duration)

        self.app.monitor = Monitor(self.app.topology, self.app)
        self.app.monitor.ip_to_node = self.ip_to_node
        self.app.monitor.node_to_ip = self.node_to_ip

        mobility = ns.MobilityHelper()
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.app.topology.nodes)
        mobility.Install(self.app.clients)
        mobility.Install(self.app.servers)

        anim = self.app.monitor.setup_animation(self.app.animFile)
        self.app.monitor.setup_packet_log()
        self.app.monitor.setup_flow_monitor()

    def step(self):
        self.run_simulation(self.simulation_duration)
        e = self.calculate_energy()
        q = self.calculate_qos()
        reward = self.calculate_reward(e, q)
        return None, reward, e, q

    def run_simulation(self, duration):
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()
        self.app.monitor.trace_routes()
        self.app.monitor.get_packet_logs()

    def calculate_reward(self, e, q):
        # Normalize energy
        e_norm = e / 415000

        # Calculate success rate
        n_total = sum(info["max_packets"]
                      for info in self.app.client_info.values())
        n_failed = sum(info["failed"]
                       for info in self.app.client_info.values())

        if n_failed > 0:
            r = 1 - (n_failed / n_total) + 0.5
        else:
            r = 100 * (1 - e_norm)
        return r

    def calculate_energy(self):
        total_e = 0
        sim_duration = self.simulation_duration
        for i in range(self.topology.N_routers):
            if self.active_routers[i] == 0:
                continue
            e_base = sample_data["routers"][i]["P_base"] * sim_duration
            total_e += e_base

            for edge_id, interface in self.inter_info.items():
                if interface['is_active'] == 1 and interface['node'] == i:
                    t_tx = interface['total_time_tx']
                    t_rx = interface['total_time_rx']
                    t_idle = sim_duration - (t_rx + t_tx)
                    e_rx = t_rx * sample_data["routers"][i]["P_rx"]
                    e_tx = t_tx * sample_data["routers"][i]["P_tx"]
                    e_idle = t_idle * sample_data["routers"][i]["P_idle"]
                    self.inter_info[edge_id]['energy'] = e_rx + e_tx + e_idle
                    total_e += e_rx + e_tx + e_idle

        return total_e

    def calculate_qos(self):
        W = []
        Q = []
        for flow_id, flow in self.app.monitor.flow_info.items():
            q_type = flow["q_type"]
            w_b = sample_data["q_list"][q_type]["w_b"]
            w_j = sample_data["q_list"][q_type]["w_j"]
            w_d = sample_data["q_list"][q_type]["w_d"]
            w_l = sample_data["q_list"][q_type]["w_l"]

            n = flow["rx_packets"]
            p = sample_data["q_list"][q_type]["p"]
            w = n * p
            W.append(w)

            l = flow["lost_packets"] / n if n > 0 else 0
            d = flow["total_delay"]
            j = flow["total_jitter"]

            q = 1 - (w_j * j + w_d * d + w_l * l)
            Q.append(q)

        total_weight = sum(W)
        if total_weight == 0:
            return 0

        weighted_qos = sum(w * q for w, q in zip(W, Q)) / total_weight
        return weighted_qos

    def collect_edge_features(self):
        # Optimized packet processing without pandas in hot loop
        csv_path = './sim/monitor/logs/packets_log.csv'

        # Read CSV once
        df = pd.read_csv(csv_path)
        df = df[(df["Port"] == 9) | (df["Port"] == 49153)]
        df['Time'] = df['Time'].astype(float)

        # Group by source and destination for batch processing
        grouped = df.groupby(['Node', 'next_hop', 'Direction'])

        for i in range(self.topology.N_routers):
            for j in range(self.topology.N_routers):
                if i == j:
                    continue
                edge_id = (i, j)
                is_active = self.adj_matrix[i][j] == 1

                if is_active:
                    # Get relevant packets efficiently
                    try:
                        packets_i_to_j = grouped.get_group((i, str(j), 'TX'))
                        tx_bytes = int(packets_i_to_j['Size'].sum())

                        # Calculate latencies efficiently
                        latencies_tx = []
                        for _, tx_row in packets_i_to_j.iterrows():
                            packet_id = tx_row['Packet']
                            rx_packets = df[(df['Node'] == j) &
                                            (df['Packet'] == packet_id) &
                                            (df['Direction'] == 'RX') &
                                            (df['Time'] > tx_row['Time'])]
                            if not rx_packets.empty:
                                latency = rx_packets.iloc[0]['Time'] - \
                                    tx_row['Time']
                                latencies_tx.append(latency)

                        total_time_tx = sum(latencies_tx)
                        avg_time_tx = total_time_tx / \
                            len(latencies_tx) if latencies_tx else 0

                        self.inter_info[edge_id] = {
                            'node': i,
                            'interface': edge_id,
                            'is_active': 1,
                            'tx_bytes': tx_bytes,
                            'rx_bytes': 0,
                            'total_time_tx': total_time_tx,
                            'avg_time_tx': avg_time_tx,
                            'total_time_rx': 0,
                            'avg_time_rx': 0,
                            'tx_packets': len(packets_i_to_j),
                            'rx_packets': len(latencies_tx),
                            'energy': 0
                        }
                    except KeyError:
                        # No packets for this edge
                        self.inter_info[edge_id] = {
                            'node': i,
                            'interface': edge_id,
                            'is_active': 1,
                            'tx_bytes': 0,
                            'rx_bytes': 0,
                            'total_time_tx': 0,
                            'avg_time_tx': 0,
                            'total_time_rx': 0,
                            'avg_time_rx': 0,
                            'tx_packets': 0,
                            'rx_packets': 0,
                            'energy': 0
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
                        'energy': 0
                    }
