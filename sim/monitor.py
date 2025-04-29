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
        # if enable_packet_metadata:
        # self.anim.EnablePacketMetadata(True)

        self.anim.EnableIpv4RouteTracking(
            sample_data['routing_table_file'], ns.Seconds(10), ns.Seconds(20))

        # self.anim.EnableIpv4L3ProtocolCounters(
        #     ns.Seconds(0), ns.Seconds(10), ns.Seconds(10))

        # self.anim.EnableQueueCounters(
        #     ns.Seconds(0), ns.Seconds(10), ns.Seconds(10))

        # self.anim.SetMaxPktsPerTraceFile(1000000)

        return self.anim

    def setup_pcap_capture(self, prefix=sample_data['pcap_files_prefix'], per_node=True, per_device=False):

        created_files = []

        p2p_helper = ns.PointToPointHelper()
        csma_helper = ns.CsmaHelper()

        all_devices = []
        device_names = []

        if self.topology and hasattr(self.topology, 'devices'):
            for i, device_container in enumerate(self.topology.devices):
                for j in range(device_container.GetN()):
                    all_devices.append(device_container.Get(j))
                    device_names.append(f"topology_dev_{i}_{j}")

        if self.app and hasattr(self.app, 'clients_ip'):
            for i, ipv4_container in enumerate(self.app.clients_ip):
                for j in range(ipv4_container.GetN()):
                    device = ipv4_container.Get(j).first
                    all_devices.append(device)
                    device_names.append(f"client_{i}_dev_{j}")

        if self.app and hasattr(self.app, 'servers_ip'):
            for i, ipv4_container in enumerate(self.app.servers_ip):
                for j in range(ipv4_container.GetN()):
                    device = ipv4_container.Get(j).first
                    all_devices.append(device)
                    device_names.append(f"server_{i}_dev_{j}")

        if per_device:
            for i, device in enumerate(all_devices):
                name = device_names[i]
                device_type = device.GetInstanceTypeId().GetName()

                filename = f"{prefix}_{name}.pcap"
                created_files.append(filename)

                if "PointToPointNetDevice" in device_type:
                    p2p_helper.EnablePcap(filename, device, True, True)
                elif "CsmaNetDevice" in device_type:
                    csma_helper.EnablePcap(filename, device, True, True)

        if per_node:
            all_nodes = []
            node_names = []

            if self.topology and hasattr(self.topology, 'nodes'):
                for i in range(self.topology.nodes.GetN()):
                    all_nodes.append(self.topology.nodes.Get(i))
                    node_names.append(f"router_{i}")

            if self.app and hasattr(self.app, 'clients'):
                for i in range(self.app.clients.GetN()):
                    all_nodes.append(self.app.clients.Get(i))
                    node_names.append(f"client_{i}")

            if self.app and hasattr(self.app, 'servers'):
                for i in range(self.app.servers.GetN()):
                    all_nodes.append(self.app.servers.Get(i))
                    node_names.append(f"server_{i}")

            for i, node in enumerate(all_nodes):
                name = node_names[i]
                filename = f"{prefix}_{name}.pcap"
                created_files.append(filename)

                for j in range(node.GetNDevices()):
                    device = node.GetDevice(j)
                    device_type = device.GetInstanceTypeId().GetName()

                    if "PointToPointNetDevice" in device_type:
                        p2p_helper.EnablePcap(filename, device, True, True)
                    elif "CsmaNetDevice" in device_type:
                        csma_helper.EnablePcap(filename, device, True, True)

        return created_files

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
        # print("Packet log setup completed with in-memory data structures")

    def get_packet_logs(self):
        """Generate packet logs directly from memory to CSV."""
        # Calculate routing paths
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

                reverse_path = path.copy()
                reverse_path.reverse()
                routing_paths.append({
                    "src_ip": server_ip,
                    "dest_ip": client_ip,
                    "path": reverse_path
                })

        # Build path lookup for fast access
        paths_map = {}
        for path_info in routing_paths:
            src_ip = path_info["src_ip"]
            dest_ip = path_info["dest_ip"]
            path = path_info["path"]
            paths_map[(src_ip, dest_ip)] = path

        # Get packet data directly from memory
        module = self.packet_module
        packet_count = module.GetPacketCount()

        # Prepare data for CSV
        csv_data = []
        for i in range(packet_count):
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

            # Calculate path information (same logic as in create_csv)
            prev_hop = "Null"
            next_hop = "Null"
            total_hops = 0

            try:
                path = paths_map.get((src_ip, dest_ip), [])
                if path:
                    node_index = path.index(node_id)
                    if node_index > 0:
                        prev_hop = path[node_index - 1]
                    if node_index < len(path) - 1:
                        next_hop = path[node_index + 1]

                    # Special cases for client/server endpoints
                    if node_index - 1 == 0 and port == 9:
                        prev_hop = "Client"
                    if node_index - 1 == 0 and port == 49153:
                        prev_hop = "Server"
                    if node_index + 1 == len(path) - 1 and port == 9:
                        next_hop = "Server"
                    if node_index + 1 == len(path) - 1 and port == 49153:
                        next_hop = "Client"

                    total_hops = len(path) - 2
            except:
                pass  # Keep default values if path info can't be calculated

            csv_data.append([
                node_id, packet_id, direction, protocol, port,
                time, size, offset, src_ip, dest_ip,
                prev_hop, next_hop, total_hops
            ])

        # Write directly to CSV
        with open("./sim/monitor/logs/packets_log.csv", "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "Node", "Packet", "Direction", "Protocol", "Port",
                "Time", "Size", "Offset", "src IP", "dest IP",
                "prev_hop", "next_hop", "total_hops"
            ])

            for chunk_start in range(0, len(csv_data), 5000):
                chunk_end = min(chunk_start + 5000, len(csv_data))
                writer.writerows(csv_data[chunk_start:chunk_end])

        csv_data = None
        module.ClearPacketData()

    def position_nodes(self, anim=None):
        if anim is None:
            anim = self.anim

        if self.topology and hasattr(self.topology, 'nodes'):
            angle_step = 360 / self.topology.N_routers
            angle = 0
            radius = 30

            for i in range(self.topology.N_routers):
                x = 100 + radius * math.cos(math.radians(angle))
                y = 50 + radius * math.sin(math.radians(angle))
                anim.SetConstantPosition(self.topology.nodes.Get(i), x, y, 0)
                angle += angle_step

        if self.app:
            if hasattr(self.app, 'clients') and hasattr(self.app, 'n_clients'):
                for i in range(self.app.n_clients):
                    anim.SetConstantPosition(
                        self.app.clients.Get(i), 0, 0+i*20, 0)

            if hasattr(self.app, 'servers') and hasattr(self.app, 'n_servers'):
                for i in range(self.app.n_servers):
                    anim.SetConstantPosition(
                        self.app.servers.Get(i), 200, 0+i*20, 0)

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

            client_ip = self.node_to_ip[client_id][0]
            server_ip = self.node_to_ip[server_id][0]

            # print(
            # f"\nRoute from Client {i} (Node {client_id}, IP {client_ip}) to Server {server_idx} (Node {server_id}, IP {server_ip}):")

            path = find_path(client_id, server_ip,
                             routing_tables, self.ip_to_node)

            if path:
                # print(f"  {' → '.join(str(node) for node in path)}")
                pass
            else:
                self.app.client_info[client_id]["failed"] = self.app.client_info[client_id]["max_packets"]
                # print(self.app.client_info[client_id]["failed"])
        #
            # print(
            # f"  No path found: {self.app.client_info[client_id]['failed']}")

    def collect_flow_stats(self, stats_file=sample_data['flow_stats_file'], app_port=None,  filter_noise=True, q=False):

        self.flow_monitor.CheckForLostPackets()
        # self.flow_monitor.SerializeToXmlFile(stats_file, True, True)

        classifier = self.flow_helper.GetClassifier()
        for flow_id, flowStats in self.flow_monitor.GetFlowStats():
            flowClass = classifier.FindFlow(flow_id)

            if filter_noise and flowStats.rxPackets < 3:
                continue

            # Extract values
            src_ip = str(flowClass.sourceAddress).strip()
            dst_ip = str(flowClass.destinationAddress).strip()
            # print(f"src_ip: {src_ip}, dst_ip: {dst_ip}")
            tx_packets = flowStats.txPackets
            rx_packets = flowStats.rxPackets
            lost_packets = tx_packets - rx_packets
            total_delay = flowStats.delaySum.GetSeconds()
            mean_delay = total_delay / rx_packets if rx_packets > 0 else 0
            total_jitter = flowStats.jitterSum.GetSeconds()
            mean_jitter = total_jitter / rx_packets if rx_packets > 0 else 0
            # Match q_type by looking up src/dest IP in self.app.client_info
            q_type = "NA"
            for info in self.app.client_info.values():
                if (info["src_ip"] == src_ip and info["dest_ip"] == dst_ip) or info["src_ip"] == dst_ip and info["dest_ip"] == src_ip:
                    q_type = info["q_type"]
                    break

            if q == True:
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
