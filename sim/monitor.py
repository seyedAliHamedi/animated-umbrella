from ns import ns

import math
import cppyy
import csv
from sim.utils import *


import xml.etree.ElementTree as ET
from pathlib import Path

cpp_code_loaded = False


class Monitor:

    def __init__(self, topology=None, app=None):
        self.topology = topology
        self.app = app

        self.flow_monitor = None
        self.flow_helper = None
        self.anim = None
        self.ip_to_node = {}
        self.node_to_ip = {}
        self.routing_tables = None
        self.trace_modules = []
        self.flow_info = {}

    def setup_animation(self, anim_file=sample_data['xml_animation_file'], enable_packet_metadata=True):
        self.anim = ns.AnimationInterface('/dev/null')
        self.anim.EnableIpv4RouteTracking(
            sample_data['routing_table_file'], ns.Seconds(10), ns.Seconds(20))
        return self.anim

    def setup_flow_monitor(self):
        self.flow_helper = ns.FlowMonitorHelper()
        self.flow_monitor = self.flow_helper.InstallAll()
        return self.flow_monitor

    def setup_packet_log(self):
        global cpp_code_loaded
        if not cpp_code_loaded:
            cppyy.cppdef(sample_data['cpp_code_f'])
            cpp_code_loaded = True
        module = cppyy.gbl

        # Set up callbacks for all routers
        rx_callback = module.CreateRxCallback()
        tx_callback = module.CreateTxCallback()

        for i in range(self.topology.nodes.GetN()):
            router = self.topology.nodes.Get(i)
            ipv4 = router.GetObject[ns.Ipv4]()
            if ipv4:
                ipv4.TraceConnectWithoutContext("Rx", rx_callback)
                ipv4.TraceConnectWithoutContext("Tx", tx_callback)

        self.packet_module = module

    def get_packet_logs(self):
        """Optimized packet log generation"""
        # Calculate routing paths more efficiently
        routing_paths = []
        for i in range(self.app.n_clients):
            client_node = self.app.clients.Get(i)
            client_id = client_node.GetId()

            server_idx = i % self.app.n_servers
            server_node = self.app.servers.Get(server_idx)
            server_id = server_node.GetId()

            client_ip = self.node_to_ip[client_id][0]
            server_ip = self.node_to_ip[server_id][0]

            path = find_path(client_id, server_ip,
                             self.routing_tables, self.ip_to_node)
            if path:
                routing_paths.append({
                    "src_ip": client_ip,
                    "dest_ip": server_ip,
                    "path": path
                })

                reverse_path = path[::-1]  # More efficient reversal
                routing_paths.append({
                    "src_ip": server_ip,
                    "dest_ip": client_ip,
                    "path": reverse_path
                })

        paths_map = {(p["src_ip"], p["dest_ip"]): p["path"]
                     for p in routing_paths}

        module = self.packet_module
        packet_count = module.GetPacketCount()

        chunk_size = 10000
        with open("./sim/monitor/logs/packets_log.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "Node", "Packet", "Direction", "Protocol", "Port",
                "Time", "Size", "Offset", "src IP", "dest IP",
                "prev_hop", "next_hop", "total_hops"
            ])

            for start_idx in range(0, packet_count, chunk_size):
                end_idx = min(start_idx + chunk_size, packet_count)
                csv_data = []

                for i in range(start_idx, end_idx):
                    node_id = module.GetPacketNodeId(i)
                    packet_id = module.GetPacketUid(i)
                    direction = module.GetPacketDirection(i)
                    protocol = module.GetPacketType(i)
                    port = module.GetPacketPort(i)
                    time = module.GetPacketTime(i)
                    size = module.GetPacketSize(i)
                    offset = module.GetPacketOffset(i)
                    src_ip = module.GetPacketSrcIp(i)
                    dest_ip = module.GetPacketDestIp(i)

                    prev_hop = "Null"
                    next_hop = "Null"
                    total_hops = 0

                    path = paths_map.get((src_ip, dest_ip), [])
                    if path and node_id in path:
                        try:
                            node_index = path.index(node_id)
                            if node_index > 0:
                                prev_hop = path[node_index - 1]
                            if node_index < len(path) - 1:
                                next_hop = path[node_index + 1]

                            if node_index == 1 and port == 9:
                                prev_hop = "Client"
                            elif node_index == 1 and port == 49153:
                                prev_hop = "Server"
                            elif node_index == len(path) - 2 and port == 9:
                                next_hop = "Server"
                            elif node_index == len(path) - 2 and port == 49153:
                                next_hop = "Client"

                            total_hops = len(path) - 2
                        except ValueError:
                            pass

                    csv_data.append([
                        node_id, packet_id, direction, protocol, port,
                        time, size, offset, src_ip, dest_ip,
                        prev_hop, next_hop, total_hops
                    ])

                writer.writerows(csv_data)

        module.ClearPacketData()

    def get_node_ips_by_id(self):
        node_ips = {}
        all_nodes = []

        if self.app and hasattr(self.app, 'clients'):
            for i in range(self.app.clients.GetN()):
                all_nodes.append(self.app.clients.Get(i))

        if self.topology and hasattr(self.topology, 'nodes'):
            for i in range(self.topology.nodes.GetN()):
                all_nodes.append(self.topology.nodes.Get(i))

        if self.app and hasattr(self.app, 'servers'):
            for i in range(self.app.servers.GetN()):
                all_nodes.append(self.app.servers.Get(i))

        for node in all_nodes:
            node_id = node.GetId()
            ipv4 = node.GetObject[ns.Ipv4]()

            if ipv4:
                ip_list = []
                for j in range(ipv4.GetNInterfaces()):
                    ip_addr = str(ipv4.GetAddress(j, 0).GetLocal())

                    if ip_addr != "127.0.0.1":
                        ip_list.append(ip_addr)

                if ip_list:
                    node_ips[node_id] = ip_list

        self.ip_to_node = get_ip_to_node(node_ips)
        self.node_to_ip = node_ips
        return node_ips

    def trace_routes(self):

        routing_tables = parse_routes_manually(
            sample_data['routing_table_file'])
        self.routing_tables = routing_tables
        for i in range(self.app.n_clients):
            client_node = self.app.clients.Get(i)
            client_id = client_node.GetId()

            server_idx = i % self.app.n_servers
            server_node = self.app.servers.Get(server_idx)
            server_id = server_node.GetId()

            client_ip = self.node_to_ip[client_id][self.app.client_gateways[i]]
            server_ip = self.node_to_ip[server_id][self.app.server_gateways[i]]

            # print(
            # f"\nRoute from Client {i} (Node {client_id}, IP {client_ip}) to Server {server_idx} (Node {server_id}, IP {server_ip}):")

            path = find_path(client_id, server_ip,
                             routing_tables, self.ip_to_node)

            if not path:
                self.app.client_info[client_id]["failed"] = self.app.client_info[client_id]["max_packets"]

    def collect_flow_stats(self, stats_file=sample_data['flow_stats_file'], app_port=None, filter_noise=True, q=False):
        self.flow_monitor.CheckForLostPackets()
        classifier = self.flow_helper.GetClassifier()

        for flow_id, flowStats in self.flow_monitor.GetFlowStats():
            flowClass = classifier.FindFlow(flow_id)

            if filter_noise and flowStats.rxPackets < 3:
                continue

            # Extract values
            src_ip = str(flowClass.sourceAddress).strip()
            dst_ip = str(flowClass.destinationAddress).strip()
            tx_packets = flowStats.txPackets
            rx_packets = flowStats.rxPackets
            lost_packets = tx_packets - rx_packets
            total_delay = flowStats.delaySum.GetSeconds()
            mean_delay = total_delay / rx_packets if rx_packets > 0 else 0
            total_jitter = flowStats.jitterSum.GetSeconds()
            mean_jitter = total_jitter / rx_packets if rx_packets > 0 else 0

            # Match q_type more efficiently
            q_type = "NA"
            for info in self.app.client_info.values():
                if ((info["src_ip"] == src_ip and info["dest_ip"] == dst_ip) or
                        (info["src_ip"] == dst_ip and info["dest_ip"] == src_ip)):
                    q_type = info["q_type"]
                    break

            if q:
                self.flow_info[flow_id] = {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "tx_packets": tx_packets,
                    "rx_packets": rx_packets,
                    "lost_packets": lost_packets,
                    "mean_delay": mean_delay,
                    "total_delay": total_delay,
                    "mean_jitter": mean_jitter,
                    "total_jitter": total_jitter,
                    "q_type": q_type
                }
