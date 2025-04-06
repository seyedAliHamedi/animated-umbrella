from ns import ns
import os
import sys


from sim.utils import *

class Topology:
    def __init__(self,adj_matrix=sample_data['topology_adj_matrix'],links_type=sample_data['topology_links_type'],links_rate=sample_data['topology_links_rate'],links_delay=sample_data['topology_links_delay'],links_queue=sample_data['topology_links_queue'],links_errors=sample_data['topology_links_errors'],base_network=sample_data['topology_base_network'],xml_file=sample_data['topology_xml_file']):
        
        
        self.adj_matrix=adj_matrix
        self.N_routers=len(self.adj_matrix)
        self.N_links = sum(sum(row) for row in self.adj_matrix) // 2 
        self.links_type=links_type
        self.links_rate=links_rate
        self.links_delays=links_delay
        self.links_queue = links_queue
        self.links_errors = links_errors
        self.base_networks = base_network
        self.xml_file = xml_file
  
        self.nodes, self.devices, self.interfaces, self.ip_interfaces = self.initialize()

    def initialize(self):
        routers = ns.NodeContainer()
        routers.Create(self.N_routers)

        devices = []

        links_types = distribute_values(self.links_type, self.N_links)
        links_rate = distribute_values(self.links_rate, self.N_links)
        link_delays = distribute_values(self.links_delays, self.N_links)
        links_queue = distribute_values(self.links_queue, self.N_links)
        links_errors = distribute_values(self.links_errors, self.N_links)


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
        subnet_mask = calculate_subnet_mask(mask_bits)
        
        address.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask(subnet_mask))
        
        ip_interfaces = []
        for i,_ in enumerate(devices):
            ip_interfaces.append(address.Assign(devices[i]))
            

        return routers, devices, internet, ip_interfaces



  