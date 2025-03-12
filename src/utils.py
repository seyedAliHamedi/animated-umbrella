from ns import ns
import random




sample_data = {
    "topology_adj_matrix" : [
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
    ],
    "topology_links_type":['csma', 'p2p'],
    "topology_links_rate":['5Mbps', '10Mbps', '1Mbps'],
    "topology_links_delay" :['5ms', '10ms', '10ms',],
    "topology_links_queue" :['5000','10000'],
    "topology_links_errors" : [0,0.1,0],
    "topology_base_network" : "192.166.1.0/24",
    "topology_xml_file" : "./src/monitor/xml/topology.xml",
    
    
    "app_n_servers": 4,
    "app_n_clients": 5,
    "app_links_type": ['csma', 'p2p'],
    "app_links_rate": ['5Mbps', '10Mbps', '1Mbps'],
    "app_links_delay": ['5ms', '10ms', '10ms',],
    "app_type": "udp_echo",
    "app_max_packets": 10000,
    "app_interval": 1,
    "app_packet_size": 1024,
    "app_port": 9,
    "tcp_app_data_rate":500000,
    
    "app_animation_file": "./src/monitor/xml/app.xml",
    
    "app_duration": 100,
    "app_start_time":10,
    
    
}






def distribute_values(values, count):
    return [random.choice(values) for _ in range(count)]

def calculate_subnet_mask(mask_bits):
    subnet_mask = [0, 0, 0, 0]
    for j in range(4):
        if mask_bits > 8:
            subnet_mask[j] = 255
            mask_bits -= 8
        else:
            subnet_mask[j] = 256 - 2 ** (8 - mask_bits)
            mask_bits = 0
    return ".".join(map(str, subnet_mask))









def fix_xml(animFile="./animated-umbrella/rip_udp.xml"):
    print("\n ------------------------- fixing XML --------------------------")
    with open(animFile,"r") as file:
        data = file.read()
    data = data.replace("&amp;#10","&#10")
    with open(animFile,"w") as file:
        file.write(data)
    print("XML Fixed")

def create_xml(all_nodes, positions, animFile):
    print("\n ------------------------- creating XML --------------------------")
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(all_nodes)

    positions_vec = [ns.Vector(pos[0], pos[1], 0) for pos in positions]

    for i in range(all_nodes.GetN()):
        mobility_model = all_nodes.Get(i).GetObject[ns.MobilityModel]()
        if mobility_model:
            mobility_model.SetPosition(positions_vec[i])

    anim = ns.AnimationInterface(animFile)

    for i in range(all_nodes.GetN()):  
        node = all_nodes.Get(i)   
        anim.SetConstantPosition(node, positions[i][0], positions[i][1],0)

    print("XML Created")
    return all_nodes, anim


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

def get_all_ipv6_addresses(node):
        """ Retrieves all IPv6 addresses assigned to a node's interfaces. """
        ipv6 = node.GetObject[ns.Ipv6]()
        if ipv6 is None:
            return ["No IPv6"]

        num_interfaces = ipv6.GetNInterfaces()
        addresses = []

        for i in range(1,num_interfaces):
            for j in range(ipv6.GetNAddresses(i)):
                addr = ipv6.GetAddress(i, j).GetAddress()    
                if not addr.IsLinkLocal():  # Ignore link-local addresses
                    addresses.append(f"{addr.ConvertTo()} (iface {i})\n")

        return addresses
