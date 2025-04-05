from ns import ns
import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import *



class App:
    def __init__(self, topology, n_servers=sample_data['app_n_servers'], n_clients=sample_data['app_n_clients'], 
                 links_type=sample_data['app_links_type'], links_rate=sample_data['app_links_rate'], 
                 links_delay=sample_data['app_links_delay'], app_type=sample_data['app_type'],
                 app_max_packets=sample_data['app_max_packets'], app_interval=sample_data['app_interval'], 
                 app_packet_size=sample_data['app_packet_size'], app_start_time=sample_data['app_start_time'],
                 app_duration=sample_data['app_duration'], tcp_app_data_rate=sample_data['tcp_app_data_rate'], 
                 app_port=sample_data['app_port'], animFile=sample_data['app_animation_file']):
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
        self.app_start_time = app_start_time
        self.app_duration = app_duration
        self.tcp_app_data_rate = tcp_app_data_rate
        self.app_port = app_port
        self.animFile = animFile
        self.monitor = None

        self.clients, self.servers, self.clients_ip, self.servers_ip = self.initialize_client_server()
        self.install_app()

    def initialize_client_server(self):
        clients = ns.NodeContainer()
        servers = ns.NodeContainer()
        clients_ip = []
        servers_ip = []
        clients.Create(self.n_clients)
        servers.Create(self.n_servers)
        
        stack = ns.InternetStackHelper()
        ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()
        rip = ns.RipHelper()
        ipv4RoutingHelper.Add(rip, 10)
        stack.SetRoutingHelper(ipv4RoutingHelper)
        stack.Install(clients)
        stack.Install(servers)

        available_gateways = list(range(self.topology.N_routers))
        client_gateways = random.sample(available_gateways, self.n_clients)
        remaining_gateways = [gw for gw in available_gateways if gw not in client_gateways]
        server_gateways = random.sample(remaining_gateways, self.n_servers) if len(remaining_gateways) >= self.n_servers else random.sample(available_gateways, self.n_servers)

        print("Clients gateways:", client_gateways)
        print("Servers gateways:", server_gateways)

        links_types = distribute_values(self.links_type, self.n_clients + self.n_servers)
        links_rate = distribute_values(self.links_rate, self.n_clients + self.n_servers)
        link_delays = distribute_values(self.links_delays, self.n_clients + self.n_servers)

        address = ns.Ipv4AddressHelper()

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
            node_pair.Add(client)
            node_pair.Add(gateway)
            device_pair = link.Install(node_pair)
            
            address.SetBase(ns.Ipv4Address(f"1.1.{10+i}.0"), ns.Ipv4Mask("255.255.255.0"))
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
            node_pair.Add(server)
            node_pair.Add(gateway)
            device_pair = link.Install(node_pair)
            
            address.SetBase(ns.Ipv4Address(f"2.2.{10+i}.0"), ns.Ipv4Mask("255.255.255.0"))
            ip_interface = address.Assign(device_pair)
            servers_ip.append(ip_interface)

        return clients, servers, clients_ip, servers_ip

    def install_app(self):
        for i in range(self.n_servers):
            self.setup_server(self.servers.Get(i))
        
        for i in range(self.n_clients):
            client = self.clients.Get(i)
            server = self.servers_ip[i % self.n_servers]
            self.setup_client(client, server)

    def setup_server(self, server):
        if self.app_type == "udp_echo":
            udp_echo_server = ns.UdpEchoServerHelper(self.app_port)
            server_app = udp_echo_server.Install(server)
        elif self.app_type == "tcp_echo":
            localAddress = ns.InetSocketAddress(ns.Ipv4Address.GetAny(), self.app_port).ConvertTo()
            tcp_server = ns.PacketSinkHelper("ns3::TcpSocketFactory", localAddress)
            server_app = tcp_server.Install(server)
            
        server_app.Start(ns.Seconds(self.app_start_time))
        server_app.Stop(ns.Seconds(self.app_start_time + self.app_duration))

    def setup_client(self, client, server):
        server_ip = server.GetAddress(0, 0).ConvertTo()
        
        if self.app_type == "udp_echo":
            echo_client = ns.UdpEchoClientHelper(server_ip, self.app_port)
            echo_client.SetAttribute("MaxPackets", ns.UintegerValue(self.app_max_packets))
            echo_client.SetAttribute("Interval", ns.TimeValue(ns.Seconds(self.app_interval)))
            echo_client.SetAttribute("PacketSize", ns.UintegerValue(self.app_packet_size))
        elif self.app_type == "tcp_echo":
            echo_client = ns.OnOffHelper("ns3::TcpSocketFactory", 
                                          ns.Address(ns.InetSocketAddress(server.GetAddress(0, 0), self.app_port).ConvertTo()))
            echo_client.SetAttribute("OnTime", ns.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
            echo_client.SetAttribute("OffTime", ns.StringValue("ns3::ConstantRandomVariable[Constant=0]"))
            echo_client.SetAttribute("DataRate", ns.DataRateValue(ns.DataRate(self.tcp_app_data_rate)))
            echo_client.SetAttribute("PacketSize", ns.UintegerValue(self.app_packet_size))

        client_app = echo_client.Install(client)
        client_app.Start(ns.Seconds(self.app_start_time))
        client_app.Stop(ns.Seconds(self.app_start_time + self.app_duration))
