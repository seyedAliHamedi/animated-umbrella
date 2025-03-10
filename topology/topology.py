from ns import ns

import sys
import os
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import fix_xml,create_xml

sample_adj_matrix = [
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
]
sample_links_type=['csma', 'p2p']
sample_links_rate=['5Mbps', '10Mbps', '1Mbps']
sample_links_delay = ['5ms', '10ms', '10ms',]
sample_links_queue = ['5000','10000']
sample_links_errors = [0,0.1,0]
class Topology:
    def __init__(self,adj_matrix=sample_adj_matrix,links_type=sample_links_type,links_rate=sample_links_rate,links_delay=sample_links_delay,links_queue=sample_links_queue,links_errors=sample_links_errors,base_network="192.166.1.0/24",animation_file ="./visual/topology/top1.xml"):
        self.adj_matrix=adj_matrix
        self.N_routers=len(self.adj_matrix)
        self.N_links = sum(sum(row) for row in self.adj_matrix) // 2 
        self.links_type=links_type
        self.links_rate=links_rate
        self.links_delays=links_delay
        self.links_queue = links_queue
        self.links_errors = links_errors
        self.base_networks = base_network
        self.animation_file = animation_file
  
        self.nodes, self.devices, self.interfaces, self.ip_interfaces = self.initialize()

    def initialize(self):
        routers = ns.NodeContainer()
        routers.Create(self.N_routers)


        links = []
        devices = []

        links_types = self._distribute_values(self.links_type, self.N_links)
        links_rate = self._distribute_values(self.links_rate, self.N_links)
        link_delays = self._distribute_values(self.links_delays, self.N_links)
        links_queue = self._distribute_values(self.links_queue, self.N_links)
        links_errors = self._distribute_values(self.links_errors, self.N_links)


        x=0
        for i in range(self.N_routers):
            for j in range(i, self.N_routers):
                if self.adj_matrix[i][j] == 1:
                    if links_types[x] == "p2p":
                        link = ns.PointToPointHelper()
                        link.SetDeviceAttribute("DataRate", ns.StringValue(links_rate[x]))
                        link.SetChannelAttribute("Delay", ns.StringValue(link_delays[x]))
                        link.SetQueue("ns3::DropTailQueue", "MaxSize",ns.QueueSizeValue(ns.QueueSize(f"{links_queue[i]}p")))
                    elif links_types[x] == "csma":
                        link = ns.CsmaHelper()
                        link.SetChannelAttribute("DataRate",ns.DataRateValue(ns.DataRate(links_rate[x])))
                        link.SetChannelAttribute("Delay", ns.StringValue(link_delays[x]))
                        link.SetQueue("ns3::DropTailQueue", "MaxSize",ns.QueueSizeValue(ns.QueueSize(f"{links_queue[i]}p")))
                
                       

                        

                    node_pair = ns.NodeContainer()
                    node_pair.Add(routers.Get(i))
                    node_pair.Add(routers.Get(j))
                    dev_pair = link.Install(node_pair)  
                    devices.append(dev_pair)
                    
                    error_model = ns.CreateObject[ns.RateErrorModel]()
                    error_model.SetRate(links_errors[i])
                    error_model.SetUnit(ns.RateErrorModel.ERROR_UNIT_PACKET)
                    dev_pair.Get(1).SetAttribute("ReceiveErrorModel",
                                            ns.PointerValue(error_model))
                    dev_pair.Get(0).SetAttribute("ReceiveErrorModel",
                                            ns.PointerValue(error_model))
                    
                    x = x+1

        internet = ns.InternetStackHelper()
        ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()

        rip = ns.RipHelper()

        ipv4RoutingHelper.Add(rip, 10)

        internet.SetRoutingHelper(ipv4RoutingHelper)
        internet.Install(routers)


        address = ns.Ipv4AddressHelper()
        
        base_ip, base_mask = self.base_networks.split("/")
        mask_bits = int(base_mask)
        subnet_mask = self._calculate_subnet_mask(mask_bits)
        
        address.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask(subnet_mask))
        
        ip_interfaces = []
        for i,_ in enumerate(devices):
            ip_interfaces.append(address.Assign(devices[i]))
            

        return routers, devices, internet, ip_interfaces



    # ========== HELPERS ==========
    def _distribute_values(self, values, count):
        return [random.choice(values) for _ in range(count)]

    def _calculate_subnet_mask(self, mask_bits):
        """Generates a subnet mask based on CIDR notation"""
        subnet_mask = [0, 0, 0, 0]
        for j in range(4):
            if mask_bits > 8:
                subnet_mask[j] = 255
                mask_bits -= 8
            else:
                subnet_mask[j] = 256 - 2 ** (8 - mask_bits)
                mask_bits = 0
        return ".".join(map(str, subnet_mask))
    
    def generate_animation_xml(self):
        positions = []
        angle_step = 360 / self.N_routers  
        angle = 0
        radius = 50  

        for i in range(self.N_routers):
            x = 50 + radius * math.cos(math.radians(angle))
            y = 50 + radius * math.sin(math.radians(angle))
            positions.append([x, y])  
            angle += angle_step  

        print("\n✅ Node Positions (x, y):")
        print(positions)

        self.nodes, anim = create_xml(self.nodes, positions, self.animation_file)


        # ns.Simulator.Stop(ns.Seconds(10))
        # ns.Simulator.Run()
        # ns.Simulator.Destroy()
        print(f"📂 Animation XML file generated: {self.animation_file}")
        
    def summary(self):
        """Prints a summary of the network topology."""
        print("\n" + "=" * 50)
        print("🔥 NETWORK TOPOLOGY SUMMARY 🔥")
        print("=" * 50)
        print(f"🔹 Number of Routers: {self.N_routers}")
        print(f"🔹 Number of Links: {self.N_links}")
        print(f"🔹 Base Network: {self.base_networks}")
        print(f"🔹 Animation File: {self.animation_file}")
        print("-" * 50)

        print("\n🔹 Routers:")
        for i in range(self.N_routers):
            print(f"   - Router {i}")

        print("\n🔹 IP Interfaces:")
        for i, ip in enumerate(self.ip_interfaces):
            print(f"   - Interface {i}: {ip.GetAddress(0)}") 

        print("=" * 50 + "\n")
        
        

# ========================== EXAMPLE USAGE ==========================
def main():
    t = Topology()
    t.summary()
    t.generate_animation_xml()


# main()