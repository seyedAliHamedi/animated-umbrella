from ns import ns

import os
import sys
import math
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topology.topology import Topology

sample_links_type = ['csma', 'p2p', 'p2p', 'p2p', 'csma']
sample_links_rate = ['5Mbps', '5Mbps', '1Mbps', '1Mbps', '5Mbps']
sample_links_delay = ['5ms', '10ms', '10ms', '1ms', '5ms']


ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

class App:
    def __init__(self, topology, n_servers=2, n_clients=3, 
                 links_type=sample_links_type, links_rate=sample_links_rate, 
                 links_delay=sample_links_delay, app_type="udp_echo",
                 app_max_packets=100, app_interval=1, app_packet_size=1024, 
                 app_duration=150, app_port=9):
        self.topology = topology
        self.n_servers = n_servers
        self.n_clients = n_clients
        self.app_type = app_type
        self.links_type = links_type
        self.links_rate = links_rate
        self.links_delays = links_delay
        self.app_max_packets = app_max_packets
        self.app_interval = app_interval
        self.app_packet_size = app_packet_size
        self.app_duration = app_duration
        self.app_port = app_port

        self.clients, self.servers,self.clients_ip,self.servers_ip = self.initialize_client_server()
        self.routing()
        self.install_app()

    def initialize_client_server(self):
        clients = ns.NodeContainer()
        servers = ns.NodeContainer()
        clients_ip=[]
        servers_ip=[]
        clients.Create(self.n_clients)
        servers.Create(self.n_servers)

        available_gateways = list(range(self.topology.N_routers))
        client_gateways = random.sample(available_gateways, self.n_clients)
        remaining_gateways = [gw for gw in available_gateways if gw not in client_gateways]
        server_gateways = random.sample(remaining_gateways, self.n_servers)

        print("clients gateways ",client_gateways)
        print("wervers gateways ",server_gateways)


        links_types = self._distribute_values(self.links_type, self.n_clients + self.n_servers)
        links_rate = self._distribute_values(self.links_rate, self.n_clients + self.n_servers)
        link_delays = self._distribute_values(self.links_delays, self.n_clients + self.n_servers)

        address = ns.Ipv4AddressHelper()
        stack = ns.InternetStackHelper()
        

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
            
            node_pair = ns.NodeContainer()
            node_pair.Add(gateway)
            node_pair.Add(client)
            
            device_pair = link.Install(node_pair)
            
            stack.Install(client)
            
            address.SetBase(ns.Ipv4Address(f"192.167.{1+i}.0"), ns.Ipv4Mask("255.255.255.0"))
            ip_interface = address.Assign(device_pair)
            clients_ip.append(ip_interface)

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
            
            node_pair = ns.NodeContainer()
            node_pair.Add(gateway)
            node_pair.Add(server)
            
            device_pair = link.Install(node_pair)

            stack.Install(server)
            
            address.SetBase(ns.Ipv4Address(f"192.169.{1+i}.0"), ns.Ipv4Mask("255.255.255.0"))
            ip_interface = address.Assign(device_pair)
            servers_ip.append(ip_interface)
            
        

        return clients, servers,clients_ip,servers_ip

        

    def routing(self):
        all_nodes = ns.NodeContainer()
        for i in range(self.n_clients):
                all_nodes.Add(self.clients.Get(i))

        for i in range(self.n_servers):
            all_nodes.Add(self.servers.Get(i))
    
        for i in range(self.topology.N_routers):
            all_nodes.Add(self.topology.nodes.Get(i))
            
        internet = ns.InternetStackHelper()
        ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()
        print("8")

        rip = ns.RipHelper()
        ipv4RoutingHelper.Add(rip, 10)
        print("9")

        internet.SetRoutingHelper(ipv4RoutingHelper)
        internet.Install(all_nodes)
        print("10")


    def install_app(self):
        for i in range(self.n_servers):
            self.setup_server(self.servers.Get(i))
        for i in range(self.n_clients):
            client = self.clients.Get(i)
            server = self.servers_ip[i % self.n_servers]
            self.setup_client(client,server) 
            
    

    def setup_server(self,server):
        if self.app_type == "udp_echo":
            udp_echo_server = ns.UdpEchoServerHelper(self.app_port)
            server_app = udp_echo_server.Install(server)
            server_app.Start(ns.Seconds(3.0))
            server_app.Stop(ns.Seconds(3.0 + self.app_duration))
            print("udp echo installed on server")
    def setup_client(self, client, server):
        server_ip = server.GetAddress(1, 0).ConvertTo()
        udp_echo_client = ns.UdpEchoClientHelper(server_ip, self.app_port)
        udp_echo_client.SetAttribute("MaxPackets", ns.UintegerValue(self.app_max_packets))
        udp_echo_client.SetAttribute("Interval", ns.TimeValue(ns.Seconds(self.app_interval)))
        udp_echo_client.SetAttribute("PacketSize", ns.UintegerValue(self.app_packet_size))

        client_app = udp_echo_client.Install(client)
        client_app.Start(ns.Seconds(3.0))
        client_app.Stop(ns.Seconds(3.0 + self.app_duration))

        print("udp echo installed on client")

    def _distribute_values(self, values, count):
        chunk_size = math.ceil(count / len(values))
        return [val for val in values for _ in range(chunk_size)][:count]

    def run(self,duration):
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()
        ns.Simulator.Destroy()
        print("Simulation Finished!")


# ========================== EXAMPLE USAGE ==========================
t = Topology()
t.generate_animation_xml()
app = App(t, n_clients=3, n_servers=2, app_type="udp_echo")
app.run(100)