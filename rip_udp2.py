from ns import ns
import cppyy
from src.topology import Topology
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
    print(f"Setting up IP-layer packet tracing for router {node_id}")

    # Get the Ipv4 object from this node
    ipv4 = router.GetObject[ns.Ipv4]()
    if ipv4 is None:
        print(f"No Ipv4 object found on node {node_id}, skipping IP-layer hookup.")
        return

    # Assuming your module is at index 0 in trace_modules,
    # or adapt if you have per-interface modules.
    module = trace_modules[node_id]

    # Build the C++ callbacks for this node
    rx_callback_func = getattr(module, f"node{node_id}_CreateRxCallback")
    tx_callback_func = getattr(module, f"node{node_id}_CreateTxCallback")

    rx_callback = rx_callback_func()
    tx_callback = tx_callback_func()

    # Connect the IP-layer "Rx" and "Tx" trace sources:
    ipv4.TraceConnectWithoutContext("Rx", rx_callback)
    ipv4.TraceConnectWithoutContext("Tx", tx_callback)


print("Create nodes")
all_nodes = ns.NodeContainer()
all_nodes.Create(6) 

n0 = all_nodes.Get(0)
r0 = all_nodes.Get(1)
r1 = all_nodes.Get(2)
r2 = all_nodes.Get(3)
r3 = all_nodes.Get(4)
n1 = all_nodes.Get(5)

net1 = ns.NodeContainer()
net1.Add(n0)
net1.Add(r0)

net2 = ns.NodeContainer()
net2.Add(n0)
net2.Add(r1)

net3 = ns.NodeContainer()
net3.Add(r0)
net3.Add(r3)

net4 = ns.NodeContainer()
net4.Add(r1)
net4.Add(r2)

net5 = ns.NodeContainer()
net5.Add(r2)
net5.Add(r3)

net6 = ns.NodeContainer()
net6.Add(r3)
net6.Add(n1)

# Install IPv4 Internet Stack with RIP routing protocol
internet = ns.InternetStackHelper()
ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()

rip = ns.RipHelper()

ipv4RoutingHelper.Add(rip, 10)

internet.SetRoutingHelper(ipv4RoutingHelper)
internet.Install(all_nodes)

# Create channels
csma2 = ns.CsmaHelper()
csma2.SetChannelAttribute(
    "DataRate", ns.DataRateValue(ns.DataRate(5000000)))
csma2.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(2)))

csma = ns.CsmaHelper()
csma.SetChannelAttribute(
    "DataRate", ns.DataRateValue(ns.DataRate(5000000)))
csma.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(1)))
csma.SetQueue("ns3::DropTailQueue", "MaxSize",
                ns.QueueSizeValue(ns.QueueSize("1000p")))

# error_model = ns.CreateObject[ns.RateErrorModel]()
# error_model.SetRate(0.1)
# # Apply per packet
# error_model.SetUnit(ns.RateErrorModel.ERROR_UNIT_PACKET)
# error_model.SetAttribute("ErrorRate", ns.DoubleValue(0.1))

d1 = csma.Install(net1)  # n0 - r0
# d1.Get(1).SetAttribute("ReceiveErrorModel",
#                        ns.PointerValue(error_model))
# d1.Get(0).SetAttribute("ReceiveErrorModel",
#                        ns.PointerValue(error_model))

d2 = csma.Install(net2)  # n0 - r1
d3 = csma.Install(net3)  # r0 - r3
d4 = csma.Install(net4)  # r1 - r2
d5 = csma.Install(net5)  # r2 - r3
d6 = csma.Install(net6)  # r3 - n1
# trafficControl = ns.TrafficControlHelper()

# trafficControl.SetRootQueueDisc("ns3::PfifoFastQueueDisc",
#                                 "MaxSize", ns.QueueSizeValue(
#                                     ns.QueueSize("1000p"))
# )

# r0_devices = ns.NetDeviceContainer()
# r0_devices.Add(d1.Get(1))
# r0_devices.Add(d3.Get(0))
# trafficControl.Install(d1)
# trafficControl.Install(d2)
# trafficControl.Install(d3)
# trafficControl.Install(d4)
# trafficControl.Install(d5)
# trafficControl.Install(d6)

print("Addressing")
ipv4 = ns.Ipv4AddressHelper()

ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
i1 = ipv4.Assign(d1)

ipv4.SetBase(ns.Ipv4Address("192.168.2.0"), ns.Ipv4Mask("255.255.255.0"))
i2 = ipv4.Assign(d2)

ipv4.SetBase(ns.Ipv4Address("192.168.3.0"), ns.Ipv4Mask("255.255.255.0"))
i3 = ipv4.Assign(d3)

ipv4.SetBase(ns.Ipv4Address("192.168.4.0"), ns.Ipv4Mask("255.255.255.0"))
i4 = ipv4.Assign(d4)

ipv4.SetBase(ns.Ipv4Address("192.168.5.0"), ns.Ipv4Mask("255.255.255.0"))
i5 = ipv4.Assign(d5)

ipv4.SetBase(ns.Ipv4Address("192.168.6.0"), ns.Ipv4Mask("255.255.255.0"))
i6 = ipv4.Assign(d6)

print(30*"--")
ip_addresses = get_node_ips_by_id(all_nodes)
for node_id, ips in ip_addresses.items():
    print(f"Node {node_id}: {', '.join(ips)}")
print(30*"--")
print("Setting up energy models for routers...")
energyHelper = ns.BasicEnergySourceHelper()
energyHelper.Set("BasicEnergySourceInitialEnergyJ", ns.DoubleValue(1.0))
energySources = energyHelper.Install(all_nodes)

# set_interface_state(r1, -1, False)
# set_interface_state(r0, -1, False)

# # Create UDP Server on n1
# # ---------------------------
# # TCP Server on n1
# # ---------------------------
# print("Setting up TCP Server")
# serverPort = 9
# packetSinkHelper = ns.PacketSinkHelper(
#     "ns3::TcpSocketFactory",
#     ns.InetSocketAddress(ns.Ipv4Address.GetAny(), serverPort).ConvertTo()
# )

# serverApps = packetSinkHelper.Install(n1)
# serverApps.Start(ns.Seconds(1.0))
# serverApps.Stop(ns.Seconds(30.0))

# # ---------------------------
# # TCP Client on n0
# # ---------------------------
# print("Setting up TCP Client")

# bulkSendHelper = ns.BulkSendHelper(
#     "ns3::TcpSocketFactory",
#     ns.InetSocketAddress(i6.GetAddress(1, 0), serverPort).ConvertTo()
# )
# bulkSendHelper.SetAttribute("MaxBytes", ns.UintegerValue(0))
# # Optional: control the send size (default 512 bytes)
# bulkSendHelper.SetAttribute("SendSize", ns.UintegerValue(1024))

# clientApps = bulkSendHelper.Install(n0)
# clientApps.Start(ns.Seconds(2.0))
# clientApps.Stop(ns.Seconds(30.0))

print("Setting up UDP Server")
udpServer = ns.UdpEchoServerHelper(9)
serverApps = udpServer.Install(n1)
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(30.0))

print("Setting up UDP Client")
udpClient = ns.UdpEchoClientHelper(i6.GetAddress(1, 0).ConvertTo(), 9)
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


# print("Tracing")
# ascii = ns.AsciiTraceHelper()
# csma.EnableAsciiAll(ascii.CreateFileStream("udp_rip2.tr"))
# csma.EnablePcapAll("udp_rip2", True)

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