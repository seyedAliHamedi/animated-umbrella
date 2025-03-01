from ns import ns

import os
import sys
import math
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topology.topology import Topology



sample_links_type = ['csma', 'p2p', 'p2p', 'p2p', 'csma']
sample_links_rate = ['5Mbps', '5Mbps',  '1Mbps', '1Mbps',  '5Mbps']
sample_links_delay = ['5ms', '10ms', '10ms','1ms', '5ms']

class App:
    def __init__(self, topology, n_servers=2, n_clients=3, 
                 links_type=sample_links_type, links_rate=sample_links_rate, 
                 links_delay=sample_links_delay, app_type="udp_echo"):
        self.topology = topology
        self.n_servers = n_servers
        self.n_clients = n_clients
        self.app_type = app_type
        self.links_type = links_type
        self.links_rate = links_rate
        self.links_delays = links_delay

        self.clients, self.servers = self.initialize_client_server()

        self.setup_routing()

        if self.app_type == "udp_echo":
            self.enable_udp_echo()
        elif self.app_type == "tcp_echo":
            self.enable_tcp_echo()
        elif self.app_type == "icmp_ping":
            self.enable_icmp_ping()

    def initialize_client_server(self):
        clients = ns.NodeContainer()
        servers = ns.NodeContainer()
        clients.Create(self.n_clients)
        servers.Create(self.n_servers)

        available_gateways = list(range(self.topology.N_routers))
        client_gateways = random.sample(available_gateways, self.n_clients)
        remaining_gateways = [gw for gw in available_gateways if gw not in client_gateways]
        server_gateways = random.sample(remaining_gateways, self.n_servers)

        print(f"✅ Clients assigned to gateways: {client_gateways}")
        print(f"✅ Servers assigned to gateways: {server_gateways}")

        address = ns.Ipv4AddressHelper()
        base_network = 1

        links_types = self._distribute_values(self.links_type, self.n_clients + self.n_servers)
        links_rate = self._distribute_values(self.links_rate, self.n_clients + self.n_servers)
        link_delays = self._distribute_values(self.links_delays, self.n_clients + self.n_servers)

        for i, gateway_idx in enumerate(client_gateways):
            gateway = self.topology.nodes.Get(gateway_idx)
            client = clients.Get(i)

            if links_types[i] == "p2p":
                link = ns.PointToPointHelper()
                link.SetDeviceAttribute("DataRate", ns.StringValue(links_rate[i]))
                link.SetChannelAttribute("Delay", ns.StringValue(link_delays[i]))
            elif links_types[i] == "csma":
                link = ns.CsmaHelper()
                link.SetChannelAttribute("DataRate", ns.DataRateValue(ns.DataRate(links_rate[i])))
                link.SetChannelAttribute("Delay", ns.StringValue(link_delays[i]))
            
            node_pair = ns.NodeContainer(gateway, client)
            devices = link.Install(node_pair)

            address.SetBase(ns.Ipv4Address(f"10.1.{base_network}.0"), ns.Ipv4Mask("255.255.255.0"))
            address.Assign(devices)
            base_network += 1

        for i, gateway_idx in enumerate(server_gateways):
            gateway = self.topology.nodes.Get(gateway_idx)
            server = servers.Get(i)

            if links_types[i+self.n_clients] == "p2p":
                link = ns.PointToPointHelper()
                link.SetDeviceAttribute("DataRate", ns.StringValue(links_rate[i+self.n_clients]))
                link.SetChannelAttribute("Delay", ns.StringValue(link_delays[i+self.n_clients]))
            elif links_types[i+self.n_clients] == "csma":
                link = ns.CsmaHelper()
                link.SetChannelAttribute("DataRate", ns.DataRateValue(ns.DataRate(links_rate[i+self.n_clients])))
                link.SetChannelAttribute("Delay", ns.StringValue(link_delays[i+self.n_clients]))
            
            node_pair = ns.NodeContainer(gateway, server)
            devices = link.Install(node_pair)

            address.SetBase(ns.Ipv4Address(f"192.168.{base_network}.0"), ns.Ipv4Mask("255.255.255.0"))
            address.Assign(devices)
            base_network += 1

        return clients, servers

      

    def setup_routing(self):
        all_nodes = ns.NodeContainer()
        for client in self.clients:
            all_nodes.Add(client)
        for server in self.servers:
            all_nodes.Add(server)
        for router in self.topology.nodes:
            all_nodes.Add(router)
        
        internet = ns.InternetStackHelper()
        ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()

        rip = ns.RipHelper()
        ipv4RoutingHelper.Add(rip, 10)

        internet.SetRoutingHelper(ipv4RoutingHelper)
        internet.Install(all_nodes)


    def enable_udp_echo(self,duration,client,server,max_packets=5,interval=1.0,packet_size=1024,port=9):
        udp_echo_server = ns.UdpEchoServerHelper(9)
        server_app = udp_echo_server.Install(server)
        server_app.Start(ns.Seconds(3.0))
        server_app.Stop(ns.Seconds(3.0+duration))

        udp_echo_client = ns.UdpEchoClientHelper(client, port)
        udp_echo_client.SetAttribute("MaxPackets", ns.UintegerValue(max_packets))
        udp_echo_client.SetAttribute("Interval", ns.TimeValue(ns.Seconds(interval)))
        udp_echo_client.SetAttribute("PacketSize", ns.UintegerValue(packet_size))

        client_app = udp_echo_client.Install(self.clients.Get(0))
        client_app.Start(ns.Seconds(3.0))
        client_app.Stop(ns.Seconds(3.0+duration))



    def _distribute_values(self, values, count):
        chunk_size = math.ceil(count / len(values))
        return [val for val in values for _ in range(chunk_size)][:count]

    def run(self, duration=15):
        ns.Simulator.Run()
        ns.Simulator.stop(duration)
        ns.Simulator.Destroy()
        print("Simulation Finished!")


# ========================== EXAMPLE USAGE ==========================
t = Topology()

app = App(t, n_clients=3, n_servers=2, app_type="udp_echo")

app.run()