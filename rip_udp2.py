from ns import ns
import cppyy
from topology.topology import Topology
import os
from utils import run_cpp_file, create_csv, generate_node_files


sikim = {}


def get_node_ips_by_id(node_container):
    """ Retrieves all IP addresses for each node and maps them to their Node ID """
    node_ips = {}

    for i in range(node_container.GetN()):  # Iterate over nodes
        node = node_container.Get(i)
        node_id = node.GetId()  # Get Node ID
        ipv4 = node.GetObject[ns.Ipv4]()  # Get IPv4 stack

        if ipv4:
            ip_list = []
            for j in range(ipv4.GetNInterfaces()):  # Iterate over interfaces
                ip_addr = str(ipv4.GetAddress(j, 0).GetLocal())

                if ip_addr != "127.0.0.1":  # Ignore loopback
                    ip_list.append(ip_addr)  # Store IP for the node

            if ip_list:  # Ensure we only add nodes with valid IPs
                node_ips[node_id] = ip_list

    return node_ips  # Dictionary mapping Node ID to list of IPs


def setup_packet_tracing_for_router(router, trace_modules):
    node_id = router.GetId()
    print(f"Setting up packet tracing for router {node_id}")
    
    for i in range(router.GetNDevices()):
        device = router.GetDevice(i)
        

        module = trace_modules[i]
        
        rx_callback_func = getattr(module, f"node{node_id}_CreateRxCallback")
        tx_callback_func = getattr(module, f"node{node_id}_CreateTxCallback")
        
        rx_callback = rx_callback_func()
        tx_callback = tx_callback_func()
                
        device.TraceConnectWithoutContext("PhyRxEnd", rx_callback)
        device.TraceConnectWithoutContext("PhyTxEnd", tx_callback)


print("Create nodes")
a = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0],
]
t = Topology(adj_matrix=a, links_type=['p2p'])
end_device = ns.NodeContainer()
end_device.Create(2)

n0 = end_device.Get(0)
n1 = end_device.Get(1)

net1 = ns.NodeContainer()
net1.Add(n0)
net1.Add(t.nodes.Get(0))

net2 = ns.NodeContainer()
net2.Add(n1)
net2.Add(t.nodes.Get(3))

all_nodes = ns.NodeContainer()
all_nodes.Add(n0)
for i in range(t.N_routers):
    all_nodes.Add(t.nodes.Get(i))
all_nodes.Add(n1)

print("Created nodes")

print("Attaching C++ Tracing to NetDevices")
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))

d1 = p2p.Install(net1)
d2 = p2p.Install(net2)

devices = ns.NetDeviceContainer()
devices.Add(d1)
devices.Add(d2)

print("Installing Internet Stack")
internet = ns.InternetStackHelper()
internet.Install(all_nodes)

ipv4 = ns.Ipv4AddressHelper()
ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
i1 = ipv4.Assign(d1)

ipv4.SetBase(ns.Ipv4Address("192.168.2.0"), ns.Ipv4Mask("255.255.255.0"))
i2 = ipv4.Assign(d2)

print("Setting up UDP Server")
udpServer = ns.UdpEchoServerHelper(9)
serverApps = udpServer.Install(n1)
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(30.0))

print("Setting up UDP Client")
udpClient = ns.UdpEchoClientHelper(i2.GetAddress(0, 0).ConvertTo(), 9)
udpClient.SetAttribute("MaxPackets", ns.UintegerValue(500))
udpClient.SetAttribute("Interval", ns.TimeValue(ns.Seconds(0.1)))
udpClient.SetAttribute("PacketSize", ns.UintegerValue(1024))

clientApps = udpClient.Install(n0)
clientApps.Start(ns.Seconds(2.0))
clientApps.Stop(ns.Seconds(30.0))


ip_list = get_node_ips_by_id(all_nodes)
# sikim[all_nodes.Get(1)]={
#     "ips":ip_list[1],
#     "packets":[
#         {id:5,size:200,port:0,time:103},
#     ]
# }

generate_node_files(all_nodes.GetN())

trace_modules = []

for i in range(all_nodes.GetN()):
    trace_modules.append(run_cpp_file(f"animated-umbrella/cpps/node{i}.cpp"))


for i in range(all_nodes.GetN()):
    setup_packet_tracing_for_router(all_nodes.Get(i), trace_modules)

# print("mmd2")


print("Starting Simulation")
ns.Simulator.Stop(ns.Seconds(30.0))
ns.Simulator.Run()


print("-" * 60)

# **Destroy Simulation**
ns.Simulator.Destroy()
for i in range(all_nodes.GetN()):
        close_func = getattr(trace_modules[i], f"node{i}_ClosePacketLog")
        close_func()
create_csv("animated-umbrella/cpps/packets_log.txt")

print("Simulation Completed!")
print("*"*60)

# GetTransmittedPackets = cppyy.gbl.GetTransmittedPackets
# GetReceivedPackets = cppyy.gbl.GetReceivedPackets

# transmitted_packets = GetTransmittedPackets()
# received_packets = GetReceivedPackets()

# # Print transmitted packets information
# print("Transmitted Packets:")
# for pkt in transmitted_packets:
#     print(f"UID: {pkt.uid}, Type: {pkt.type}, Dest Port: {pkt.destPort}, Time: {pkt.time}, Size: {pkt.size}")

    
# print("*"*60)
# # Print received packets information
# print("Received Packets:")
# for pkt in received_packets:
#     print(f"UID: {pkt.uid}, Type: {pkt.type}, Dest Port: {pkt.destPort}, Time: {pkt.time}, Size: {pkt.size}")
    
# print("*"*60)