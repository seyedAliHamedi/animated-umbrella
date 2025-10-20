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
                           conf,
                           n_apps,
                 original_adj_matrix,
                 n_servers,
                 client_gateways,
                 server_gateways,
                 ip_to_node,
                 node_to_ip,
                 simulation_duration=10,
       
                 ):

        self.adj_matrix = adj_matrix
        self.simulation_duration = simulation_duration
        self.conf = conf
        self.n_apps=n_apps
        self.n_clients = n_clients
        self.n_servers = n_servers
        self.original_adj_matrix = original_adj_matrix

        self.client_gateways = client_gateways
        self.server_gateways = server_gateways
        self.inter_info = {}
        self.ip_to_node = ip_to_node
        self.node_to_ip = node_to_ip
        self.apps=[]
        self.setup_environment()


        # self.router_type = {
        #     i: sample_data["routers"][i % len(sample_data["routers"])]
        #     for i in range(self.topology.N_routers)
        # }
        # self.router_type = {0: sample_data["routers"][4],
        #                     1: sample_data["routers"][9],
        #                     2: sample_data["routers"][1],
        #                     3: sample_data["routers"][8],
        #                     4: sample_data["routers"][12],
        #                     5: sample_data["routers"][12], }

        # self.router_type = {0: sample_data["routers"][4],
        #                     1: sample_data["routers"][7],
        #                     2: sample_data["routers"][13],
        #                     3: sample_data["routers"][8],
        #                     4: sample_data["routers"][4],
        #                     5: sample_data["routers"][6],
        #                     6: sample_data["routers"][1],
        #                     7: sample_data["routers"][3], }

        self.router_type = {
            i: sample_data["mawi_routers_one_per_city"][i]
            for i in range(len(adj_matrix))
        }

    def setup_environment(self):
        self.topology = Topology(adj_matrix=self.adj_matrix)

        self.active_routers = []
        self.active_links = {}

        for i in range(self.topology.N_routers):
            row_sum = sum(self.adj_matrix[i])
            self.active_routers.append(1 if row_sum > 0 else 0)
            self.active_links[i] = int(row_sum)
        for i in range(self.n_apps):
            a=App(self.topology, client_gateways=[self.client_gateways[i]], server_gateways=[self.server_gateways[i]], n_clients=1, n_servers=1, app_start_time=40,
                        app_duration=self.simulation_duration,configurations=self.conf,app_index=i)
            self.apps.append(a)

        self.monitor = Monitor(self.topology, self.apps)
        self.monitor.ip_to_node = self.ip_to_node
        self.monitor.node_to_ip = self.node_to_ip

        mobility = ns.MobilityHelper()
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.topology.nodes)
        [mobility.Install(app.clients.Get(0)) for app in self.apps ]
        [mobility.Install(app.servers.Get(0)) for app in self.apps ]


        
        self.monitor.setup_packet_log()
        self.monitor.setup_flow_monitor()
        self.monitor.setup_animation(self.apps[0].animFile)

    def step(self):
        overall_start = time.time()

        self.run_simulation(self.simulation_duration)
        sim_elapsed = time.time() - overall_start
        print(f"[timing] run_simulation: {sim_elapsed:.3f}s")

        energy_start = time.time()
        e = self.calculate_energy()
        print(f"[timing] calculate_energy: {time.time()-energy_start:.3f}s")

        qos_start = time.time()
        q = self.calculate_qos()
        print(f"[timing] calculate_qos: {time.time()-qos_start:.3f}s")

        reward_start = time.time()
        reward, f, r, e_eff = self.calculate_reward(e, q)
        print(f"[timing] calculate_reward: {time.time()-reward_start:.3f}s")
        print(f"[timing] step_total: {time.time()-overall_start:.3f}s")
        return None, reward, f, r, e_eff, q

    def run_simulation(self, duration):
        i=0
        sim_start = time.time()
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()
        print(f"[timing] Simulator.Run app {i}: {time.time()-sim_start:.3f}s")

        monitor = self.monitor
        trace_start = time.time()
        monitor.trace_routes()
        print(f"[timing] trace_routes app {i}: {time.time()-trace_start:.3f}s")

        packets_start = time.time()
        monitor.get_packet_logs()
        print(f"[timing] get_packet_logs app {i}: {time.time()-packets_start:.3f}s")

        flow_start = time.time()
        monitor.collect_flow_stats(
            app_port=self.apps[i].app_port, filter_noise=True, q=True)
        print(f"[timing] collect_flow_stats app {i}: {time.time()-flow_start:.3f}s")
      

    def calculate_reward(self, e, q, m=0.2, alpha=3):
        num_active_routers = sum(self.active_routers) * len(self.apps)
        num_path_routers = sum(self.monitor.path_routers)
        print(num_active_routers, num_path_routers)
        # Normalize energy
        # e_norm = e / 560750
        # e_norm = e / 583500  # latest
        # e_norm = e / 415000
        # e_norm = e / 1401250

        e_norm = e / 1431000  # mawi

        if num_path_routers != 0:
            r = num_active_routers / num_path_routers
        else:
            r = 0


        # e_eff = e * (m + alpha * (r - 1))
        # e_eff /= 820000

        # Calculate success rate
        for app in self.apps:
            n_total = sum(info["max_packets"]
                        for info in app.client_info.values())
            n_failed = sum(info["failed"]
                        for info in app.client_info.values())

        if n_failed > 0:
            f = 1
            # reward = 1 - (n_failed / n_total) + 1e-6
            reward = -1 * (n_failed / n_total)
        else:
            f = 0
            # reward = 100 * ((1 - e_norm))
            # reward = 1 * ((1 - e_norm) + q)
            if num_active_routers == len(self.active_routers) and r != 1:
                reward = -1
            else:
                # reward = 100 * (q / (e_norm + 1e-6))
                reward = np.exp(-e_norm)*np.exp(q)
                # reward = (1 / (e_norm + 1e-6))
                # reward *= (1/r)
                reward *= np.exp(1-r)
        return reward, f, r, e_norm

    def calculate_energy(self):
        total_e = 0
        sim_duration = self.simulation_duration
        for i in range(self.topology.N_routers):
            if self.active_routers[i] == 0:
                continue
            # e_base = sample_data["routers"][i % len(
            #     sample_data["routers"])]["P_base"] * sim_duration

            e_base = self.router_type[i]["P_base"] * sim_duration
            total_e += e_base

            for edge_id, interface in self.inter_info.items():
                if interface['is_active'] == 1 and interface['node'] == i:
                    t_tx = interface['total_time_tx']
                    t_rx = interface['total_time_rx']
                    t_idle = sim_duration - (t_rx + t_tx)
                    # e_rx = t_rx * sample_data["routers"][i]["P_rx"]
                    # e_tx = t_tx * sample_data["routers"][i]["P_tx"]
                    # e_idle = t_idle * sample_data["routers"][i]["P_idle"]
                    e_rx = t_rx * self.router_type[i]["P_rx"]
                    e_tx = t_tx * self.router_type[i]["P_tx"]
                    e_idle = t_idle * self.router_type[i]["P_idle"]
                    self.inter_info[edge_id]['energy'] = e_rx + e_tx + e_idle
                    total_e += e_rx + e_tx + e_idle

        return total_e

    def calculate_qos(self):

        W = []
        Q = []
        for flow_id, flow in self.monitor.flow_info.items():
            q_type = flow["q_type"]
            cfg = sample_data["mawi_q_list"][q_type]

            n_tx, n_rx = flow["tx_packets"], flow["rx_packets"]
            if n_tx == 0:                               # noise / empty flow
                continue

            w_b = cfg["w_b"]
            w_j = cfg["w_j"]
            w_d = cfg["w_d"]
            w_l = cfg["w_l"]

            p = cfg["p"]
            w = n_rx * p
            W.append(w)

            l = flow["lost_packets"] / n_tx if n_tx > 0 else 0
            l = min(1.0, l / cfg["sla_loss"])
            # d = flow["total_delay"]
            # j = flow["total_jitter"]

            d = min(1.0, flow["mean_delay"] / cfg["sla_delay"])
            j = min(1.0, flow["mean_jitter"] / cfg["sla_jitter"])

            # # Throughput term is optional: only if goodput is recorded & weight > 0
            # if cfg["w_b"] > 0.0 and "goodput" in flow:
            #     b_norm = min(1.0, flow["goodput"] / cfg["sla_bw_mbps"])
            # else:
            #     b_norm = 1.0

            q = 1 - (w_j * j + w_d * d + w_l * l)
            if q < 0.5:
                print(f"l: {l}, d: {d}, j: {j}")
        #     q = 1.0 - (
        #     cfg["w_d"] * d +
        #     cfg["w_j"] * j +
        #     cfg["w_l"] * l +
        #     cfg["w_b"] * (1.0 - b)
        # )
            # q = max(0.0, min(1.0, q))
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
