from ns import ns
import os
import sys
import pandas as pd
import random


from sim.utils import *


class App:
    def __init__(self, topology,
                 client_gateways,
                 server_gateways,
                 configurations,
                 app_index,
                  app_duration, 

                 n_servers=sample_data['app_n_servers'], n_clients=sample_data['app_n_clients'],
                 links_type=sample_data['app_links_type'], links_rate=sample_data['app_links_rate'],
                 links_delay=sample_data['app_links_delay'], app_type=sample_data['app_type'],
                 app_max_packets=sample_data['app_max_packets'], app_interval=sample_data['app_interval'],
                 app_packet_size=sample_data['app_packet_size'], app_start_time=sample_data['app_start_time'],
                tcp_app_data_rate=sample_data['tcp_app_data_rate'],
                 app_port=sample_data['app_port'], animFile=sample_data['app_animation_file'],):
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
        self.app_start_time = 40
        self.app_duration = app_duration
        self.tcp_app_data_rate = tcp_app_data_rate
        self.app_port = app_port
        self.animFile = animFile
        self.monitor = None
        self.client_info = {}
        self.client_gateways = client_gateways
        self.server_gateways = server_gateways
        self.clients_ip = []
        self.clients, self.servers, self.servers_ip = self.initialize_client_server()
  
        self.configurations = configurations
        self.app_index=app_index

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

        print("Clients gateways:", self.client_gateways)
        print("Servers gateways:", self.server_gateways)

        links_types = distribute_values(
            self.links_type, self.n_clients + self.n_servers)
        links_rate = distribute_values(
            self.links_rate, self.n_clients + self.n_servers)
        link_delays = distribute_values(
            self.links_delays, self.n_clients + self.n_servers)

        address = ns.Ipv4AddressHelper()

        for i, gateway_idx in enumerate(self.client_gateways):
            gateway = self.topology.nodes.Get(gateway_idx)
            client = clients.Get(i)

            if links_types[i] == "p2p":
                link = ns.PointToPointHelper()
                link.SetDeviceAttribute(
                    "DataRate", ns.StringValue(links_rate[i]))
                link.SetChannelAttribute(
                    "Delay", ns.StringValue(link_delays[i]))
                link.SetQueue("ns3::DropTailQueue",
                                  "MaxSize",
                                  ns.QueueSizeValue(ns.QueueSize(f"{10000}p")))
            elif links_types[i] == "csma":
                link = ns.CsmaHelper()
                link.SetChannelAttribute(
                    "DataRate", ns.DataRateValue(ns.DataRate(links_rate[i])))
                link.SetChannelAttribute(
                    "Delay", ns.StringValue(link_delays[i]))

            node_pair = ns.NodeContainer()
            node_pair.Add(client)
            node_pair.Add(gateway)
            device_pair = link.Install(node_pair)

            address.SetBase(ns.Ipv4Address(
                f"111.111.{gateway_idx}.0"), ns.Ipv4Mask("255.255.255.0"))
            ip_interface = address.Assign(device_pair)
            self.clients_ip.append(ip_interface)

        for i, gateway_idx in enumerate(self.server_gateways):
            gateway = self.topology.nodes.Get(gateway_idx)
            server = servers.Get(i)

            if links_types[i+self.n_clients] == "p2p":
                link = ns.PointToPointHelper()
                link.SetDeviceAttribute(
                    "DataRate", ns.StringValue(links_rate[i+self.n_clients]))
                link.SetChannelAttribute(
                    "Delay", ns.StringValue(link_delays[i+self.n_clients]))
            elif links_types[i+self.n_clients] == "csma":
                link = ns.CsmaHelper()
                link.SetChannelAttribute("DataRate", ns.DataRateValue(
                    ns.DataRate(links_rate[i+self.n_clients])))
                link.SetChannelAttribute(
                    "Delay", ns.StringValue(link_delays[i+self.n_clients]))

            node_pair = ns.NodeContainer()
            node_pair.Add(server)
            node_pair.Add(gateway)
            device_pair = link.Install(node_pair)

            address.SetBase(ns.Ipv4Address(
                f"222.222.{gateway_idx}.0"), ns.Ipv4Mask("255.255.255.0"))
            ip_interface = address.Assign(device_pair)
            servers_ip.append(ip_interface)

        return clients, servers, servers_ip

    def install_app(self):
        self.setup_server(self.servers.Get(0))

        client = self.clients.Get(0)
        server = self.servers_ip[0]

        self.setup_client(self.app_index, client, server)

    def _seconds_to_ns3_time(self, seconds):
        # Pick the coarsest ns-3 time unit that keeps the count >= 1
        units = (
            ("Seconds", ns.Seconds, 1.0),
            ("MilliSeconds", ns.MilliSeconds, 1e-3),
            ("MicroSeconds", ns.MicroSeconds, 1e-6),
            ("NanoSeconds", ns.NanoSeconds, 1e-9),
        )

        for unit_name, unit_ctor, scale in units:
            scaled_value = seconds / scale
            if scaled_value >= 1 or unit_name == "NanoSeconds":
                rounded_value = max(1, int(round(scaled_value)))
                return unit_ctor(rounded_value), rounded_value, unit_name

        return ns.NanoSeconds(1), 1, "NanoSeconds"
    def setup_server(self, server):
        if self.app_type == "udp_echo":
            udp_echo_server = ns.UdpEchoServerHelper(self.app_port)
            server_app = udp_echo_server.Install(server)

        server_app.Start(ns.Seconds(self.app_start_time))
        server_app.Stop(ns.Seconds(self.app_start_time) + ns.Minutes(self.app_duration))

    def setup_client(self, client_idx, client, server):
        t=self.configurations[f'F{client_idx+1}/T'] / 500
        p=self.configurations[f'F{client_idx+1}/P'] / 500

        avg_packet_size = ((t*1e6)/(p*1e3))/8   
        n_packets=  self.app_duration * 60*p*1e3
        interval = 1/(p*1e3)
        q_type = self.configurations[f'F{client_idx+1}/q_type']
        q_config = sample_data["mawi_q_list"][q_type]
        max_packets = int(n_packets)
        interval = int(interval*1e6)
        packet_size = int(avg_packet_size)
        client_ip = str(self.clients_ip[0].GetAddress(0)).strip()
        server_ip = str(server.GetAddress(0, 0)).strip()

        print(f"Client {client_idx} src_ip: {client_ip}, dst ip: {server_ip} ,qtype {q_type} \n")

        self.client_info[client.GetId()] = {
            "q_type": q_type,
            "max_packets": max_packets,
            "packet_size": packet_size,
            "interval":interval,
            "q_config":q_config,
            "failed": 0,
            "is_clientServer": 1,
            "src_ip": client_ip,
            "dest_ip": server_ip
        }
        server_ip = server.GetAddress(0, 0).ConvertTo()
     

        if self.app_type == "udp_echo":
            echo_client = ns.UdpEchoClientHelper(server_ip, self.app_port)
            echo_client.SetAttribute(
                "MaxPackets", ns.UintegerValue(max_packets))
            echo_client.SetAttribute(
                "Interval", ns.TimeValue(ns.MicroSeconds(interval)))
            echo_client.SetAttribute(
                "PacketSize", ns.UintegerValue(packet_size))
 

        client_app = echo_client.Install(client)
        client_app.Start(ns.Seconds(self.app_start_time))
        client_app.Stop(ns.Seconds(self.app_start_time) + ns.Minutes( self.app_duration))
