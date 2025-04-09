from ns import ns

import math
import json
from utils import *


import xml.etree.ElementTree as ET
from pathlib import Path


class Monitor:

    def __init__(self, topology=None, app=None):
        self.topology = topology
        self.app = app
        self.all_nodes = ns.NodeContainer()
        self.all_nodes.Add(app.clients)
        self.all_nodes.Add(topology.nodes)
        self.all_nodes.Add(app.servers)

        self.flow_monitor = None
        self.flow_helper = None
        self.anim = None
        self.ip_to_node = {}
        self.node_to_ip = {}
        self.routing_tables = None
        self.trace_modules = []

    def setup_animation(self, anim_file="./sim/monitor/xml/animation.xml", enable_packet_metadata=True):
        print(
            "\n--------------------------- Setting up animation ---------------------------")
        self.anim = ns.AnimationInterface(anim_file)
        if enable_packet_metadata:
            self.anim.EnablePacketMetadata(True)

        self.anim.EnableIpv4RouteTracking(
            "./sim/monitor/xml/routes.xml", ns.Seconds(10), ns.Seconds(20))

        self.anim.EnableIpv4L3ProtocolCounters(
            ns.Seconds(0), ns.Seconds(10), ns.Seconds(10))

        self.anim.EnableQueueCounters(
            ns.Seconds(0), ns.Seconds(10), ns.Seconds(10))

        self.anim.SetMaxPktsPerTraceFile(1000000)

        print(f"Enhanced animation setup complete. Output file")
        print("- Routing tables, IP counters, and queue information enabled for NetAnim")
        return self.anim

    def setup_pcap_capture(self, prefix="./sim/monitor/pcap/capture", per_node=True, per_device=False):
        print("\n--------------------------- Setting up PCAP capture ---------------------------")

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
                else:
                    print(
                        f"Warning: Unknown device type {device_type} for device {name}")

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

        print(f"PCAP capture enabled.")
        return created_files

    def setup_flow_monitor(self):
        print("\n--------------------------- Setting up flow monitor ---------------------------")

        self.flow_helper = ns.FlowMonitorHelper()

        self.flow_monitor = self.flow_helper.InstallAll()

        print("FlowMonitor setup completed.")
        return self.flow_monitor

    def setup_packet_log(self):
        print("\n--------------------------- Setting up packet logs ---------------------------")
        generate_node_files(self.topology.nodes.GetN())

        trace_modules = []

        for i in range(self.topology.nodes.GetN()):
            trace_modules.append(run_cpp_file(
                f"./sim/monitor/cpps/node{i}.cpp"))

        for i in range(self.topology.nodes.GetN()):
            setup_packet_tracing_for_router(
                self.topology.nodes.Get(i), trace_modules)

        self.trace_modules = trace_modules
        print("Packet log setup Completed")

    def get_packet_logs(self):
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

        for i in range(self.topology.nodes.GetN()):
            close_func = getattr(
                self.trace_modules[i], f"node{i}_ClosePacketLog")
            close_func()

        # Create the CSV with path information
        create_csv("./sim/monitor/logs/packets_log.txt", routing_paths)

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
        print("\n--------------------------- Getting node IPs ---------------------------")

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

        for node_id, ips in node_ips.items():
            print(f"Node {node_id}: {', '.join(ips)}")
        self.ip_to_node = get_ip_to_node(node_ips)
        self.node_to_ip = node_ips

    def setup_tracert(self):

        for i in range(self.app.n_clients):
            server_ip = self.app.servers_ip[i % self.app.n_servers]
            client = self.app.clients.Get(i)

            tracer = ns.V4TraceRouteHelper(server_ip.GetAddress(0, 0))

            app = tracer.Install(client)

            app.Start(ns.Seconds(self.app.app_start_time + 1.0 + i))

            app.Stop(ns.Seconds(self.app.app_start_time + 4.0 + i))
            print(
                f"Set up trace route from Client {i} to Server {i % self.app.n_servers}")

        print("Tracert setup completed.")

    def trace_routes(self):
        print("\n--------------------------- Tracing routes from clients to servers ---------------------------")

        routing_tables = parse_routes_manually("./sim/monitor/xml/routes.xml")
        self.routing_tables = routing_tables
        for i in range(self.app.n_clients):
            client_node = self.app.clients.Get(i)
            client_id = client_node.GetId()

            server_idx = i % self.app.n_servers
            server_node = self.app.servers.Get(server_idx)
            server_id = server_node.GetId()

            client_ip = self.node_to_ip[client_id][0]
            server_ip = self.node_to_ip[server_id][0]

            print(
                f"\nRoute from Client {i} (Node {client_id}, IP {client_ip}) to Server {server_idx} (Node {server_id}, IP {server_ip}):")

            path = find_path(client_id, server_ip,
                             routing_tables, self.ip_to_node)

            if path:
                print(f"  {' → '.join(str(node) for node in path)}")
            else:
                print(f"  No path found")

        print("Routing trace completed.")

    def create_csv_summary(self, flow_stats, output_file="./animated-umbrella/sim/monitor/logs/flow_summary.csv"):
        if not flow_stats:
            print("No flow statistics to write to CSV.")
            return
        try:

            with open(output_file, "w") as file:
                headers = list(flow_stats[0].keys())
                file.write(",".join(headers) + "\n")

                for flow in flow_stats:
                    values = []
                    for key in headers:
                        value = flow.get(key, "")
                        if value is None:
                            value = ""
                        elif key == 'loss_ratio':
                            value = f"{value:.6f}"
                        elif key in ['delay_ms', 'jitter_ms']:
                            value = f"{value:.2f}"
                        values.append(str(value))

                    file.write(",".join(values) + "\n")
            print(f"CSV summary created at {output_file}")

        except Exception as e:
            print(f"Error creating CSV summary: {e}")
            import traceback
            traceback.print_exc()

    def collect_flow_stats(self, stats_file="./sim/monitor/xml/flow-stats.xml", app_port=None,  filter_noise=True):
        print("\n--------------------------- Collecting flow statistics ---------------------------")

        self.flow_monitor.CheckForLostPackets()
        self.flow_monitor.SerializeToXmlFile(stats_file, True, True)

        classifier = self.flow_helper.GetClassifier()
        for flow_id, flowStats in self.flow_monitor.GetFlowStats():
            flowClass = classifier.FindFlow(flow_id)

            if filter_noise and flowStats.rxPackets < 3:
                continue

            print(f"📊 Flow {flow_id}: ")
            print(
                f"   Source IP: {flowClass.sourceAddress}, Dest IP: {flowClass.destinationAddress}")
            print(
                f"   Tx Packets: {flowStats.txPackets}, Rx Packets: {flowStats.rxPackets}")
            print(
                f"   Lost Packets: {flowStats.txPackets - flowStats.rxPackets}")
            print(
                f"   Throughput: {(flowStats.rxBytes/(flowStats.rxPackets*self.app.app_interval))} Bps")
            print(f"   Mean Delay: {flowStats.delaySum.GetSeconds()} sec")
            print(f"   Mean Jitter: {flowStats.jitterSum.GetSeconds()} sec")
