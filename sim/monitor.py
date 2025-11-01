


from ns import ns

import math
import time
import cppyy
import csv
from sim.utils import *


import xml.etree.ElementTree as ET
from pathlib import Path

cpp_code_loaded = False


class Monitor:

    def __init__(self, topology=None, apps=None):
        self.topology = topology
        self.apps = apps

        self.flow_monitor = None
        self.flow_helper = None
        self.anim = None
        self.ip_to_node = {}
        self.node_to_ip = {}
        self.routing_tables = None
        self.trace_modules = []
        self.flow_info = {}

        self.all_paths = []
        self.path_routers = [0] * self.topology.N_routers

    def setup_animation(self, anim_file=sample_data['xml_animation_file'], enable_packet_metadata=True):
        start_time = time.time()
        try:
            with suppress_cpp_output():
                self.anim = ns.AnimationInterface('/dev/null')
                if hasattr(self.anim, "SetMaxPktsPerTraceFile"):
                    self.anim.SetMaxPktsPerTraceFile(0x7fffffff)
                if enable_packet_metadata and hasattr(self.anim, "EnablePacketMetadata"):
                    self.anim.EnablePacketMetadata(True)
                self.anim.EnableIpv4RouteTracking(
                    sample_data['routing_table_file'], ns.Seconds(30), ns.Seconds(30)
                )
        except Exception as exc:
           
            print(f"[warn] monitor.setup_animation skipped: {exc}")
            self.anim = None
        print(f"[timing] monitor.setup_animation: {time.time()-start_time:.3f}s")
        return self.anim

    def setup_flow_monitor(self):
        start_time = time.time()
        self.flow_helper = ns.FlowMonitorHelper()
        self.flow_monitor = self.flow_helper.InstallAll()
        print(f"[timing] monitor.setup_flow_monitor: {time.time()-start_time:.3f}s")
        return self.flow_monitor

    def setup_packet_log(self):
        start_time = time.time()
        global cpp_code_loaded
        if not cpp_code_loaded:
            with suppress_cpp_output():
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
        print(f"[timing] monitor.setup_packet_log: {time.time()-start_time:.3f}s")

    def get_packet_logs(self):
        import time
        start_time = time.time()
        """Optimized packet log generation"""
        routing_paths = []
        for i in range(len(self.apps)):
            client_node = self.apps[i].clients.Get(0)
            client_id = client_node.GetId()

            server_node = self.apps[i].servers.Get(0)
            server_id = server_node.GetId()
            # print(self.all_paths)
            if len(self.all_paths)== 0:
                continue
            client_ip = self.node_to_ip[client_id][self.all_paths[0][0]]
            server_ip = self.node_to_ip[server_id][self.all_paths[0][-1]]

            path = find_path(client_id, server_ip,
                             self.routing_tables, self.ip_to_node)

            for path in self.all_paths:
                if path:
                    routing_paths.append({
                        "src_ip": client_ip,
                        "dest_ip": server_ip,
                        "path": path
                    })

                    reverse_path = path[::-1]  
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
        import time
        print(f"[timing] monitor.get_packet_logs: {time.time()-start_time:.3f}s")

    def get_node_ips_by_id(self):
        node_ips = {}
        all_nodes = []

        for i in range(len(self.apps)):
            all_nodes.append(self.apps[i].clients.Get(0))

        for i in range(self.topology.nodes.GetN()):
            all_nodes.append(self.topology.nodes.Get(i))

        for i in range(len(self.apps)):
            all_nodes.append(self.apps[i].servers.Get(0))

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
        start_time = time.time()
        routing_tables = parse_routes_manually(
            sample_data['routing_table_file'])
        self.routing_tables = routing_tables
        used_routers = set()
        for i in range(len(self.apps)):
            client_node = self.apps[i].clients.Get(0)
            client_id = client_node.GetId()

            
         
            client = self.apps[i].client_gateways[0]
            server = self.apps[i].server_gateways[0]
            path = find_path(client, server,
                             routing_tables, self.ip_to_node,)
            if path is None:
                path = find_path(server, client,
                                 routing_tables, self.ip_to_node,)
         
            print("PATH ___",path)
            if path:
                self.all_paths.append(path)

              
                for node_id in path:
                  
                    if node_id < self.topology.N_routers:
                        used_routers.add(node_id)

            if not path:
                self.apps[i].client_info[client_id]["failed"] = self.apps[i].client_info[client_id]["max_packets"]

        self.path_routers = [1 if i in used_routers else 0 for i in range(
            self.topology.N_routers)]
        print(f"[timing] monitor.trace_routes: {time.time()-start_time:.3f}s")

    def _resolve_flow_q_type(self, src_ip, dst_ip, src_port, dst_port):

        for app in self.apps:
            for client_info in app.client_info.values():
                client_src = client_info.get("src_ip")
                client_dst = client_info.get("dest_ip")
                if client_src == src_ip and client_dst == dst_ip:
                    return client_info.get("q_type")
                if client_src == dst_ip and client_dst == src_ip:
                    return client_info.get("q_type")

            if app.app_port in (src_port, dst_port) and app.client_info:
                first_info = next(iter(app.client_info.values()))
                return first_info.get("q_type")

        return None

    def collect_flow_stats(self, stats_file=sample_data['flow_stats_file'], app_port=None, filter_noise=True, q=False):
        start_time = time.time()
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

            print("AAAAAA ",rx_packets,tx_packets,tx_packets-rx_packets)          
            q_type = self._resolve_flow_q_type(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=int(flowClass.sourcePort),
                dst_port=int(flowClass.destinationPort),
            )

            if q_type is None:
                continue

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
        print(f"[timing] monitor.collect_flow_stats: {time.time()-start_time:.3f}s")
