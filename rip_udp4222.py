from ns import ns
import cppyy


def set_interface_state(r, interface_index, state):
    """
    Parameters:
    r - The router node
    interface_index - The interface to modify (-1 for all interfaces)
    state - Boolean (True = UP, False = DOWN)
    """
    ipv4 = r.GetObject[ns.Ipv4]()  # Get the IPv6 stack
    num_interfaces = ipv4.GetNInterfaces()  # Get total interfaces

    if interface_index == -1:
        # Apply to all interfaces (except loopback, usually index 0)
        for i in range(1, num_interfaces):
            ipv4.SetUp(i) if state else ipv4.SetDown(i)
            ipv4.GetRoutingProtocol().NotifyInterfaceDown(
                i) if not state else ipv4.GetRoutingProtocol().NotifyInterfaceUp(i)
        print(
            f"Router {r.GetId()} {'enabled' if state else 'disabled'} (all {num_interfaces-1} interfaces)")
    else:
        # Apply only to the specified interface
        if 0 < interface_index < num_interfaces:
            ipv4.SetUp(interface_index) if state else ipv4.SetDown(
                interface_index)
            ipv4.GetRoutingProtocol().NotifyInterfaceDown(
                i) if not state else ipv4.GetRoutingProtocol().NotifyInterfaceUp(i)
            print(
                f"Router {r.GetId()} {'enabled' if state else 'disabled'} (interface {interface_index})")
        else:
            print(
                f"Invalid interface index {interface_index} for router {r.GetId()}")


def calculate_interval(desired_rate_mbps, packet_size_bytes=1024):
    """
    Calculates the UDP packet interval (in seconds) needed to match the desired sending rate.

    Parameters:
    - desired_rate_mbps: Desired UDP sending rate in Mbps.
    - packet_size_bytes: Size of each UDP packet in bytes (default: 1024 bytes).

    Returns:
    - Interval in seconds (time between packet transmissions).
    """
    # Convert desired rate from Mbps to bps
    desired_rate_bps = desired_rate_mbps * 1e6  # Convert Mbps to bps

    # Calculate packets per second needed
    packets_per_second = desired_rate_bps / \
        (packet_size_bytes * 8)  # Convert bytes to bits

    # Calculate interval (time between packets)
    udp_interval_seconds = 1 / packets_per_second

    return udp_interval_seconds


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


def analyze_node_statistics(monitor, classifier, node_ips):
    """ Extracts packet statistics per node and returns a dictionary """
    monitor.CheckForLostPackets()

    node_stats = {}

    for flow_id, flowStats in monitor.GetFlowStats():
        flowClass = classifier.FindFlow(flow_id)

        # Normalize IPs
        src_ip = str(flowClass.sourceAddress)
        dest_ip = str(flowClass.destinationAddress)

        # Convert IP to Node ID by searching the mapping
        src_node = next(
            (node_id for node_id, ips in node_ips.items() if src_ip in ips), "Unknown")
        dest_node = next(
            (node_id for node_id, ips in node_ips.items() if dest_ip in ips), "Unknown")

        if src_node == "Unknown" or dest_node == "Unknown":
            print(
                f"⚠️ Warning: Could not map {src_ip} or {dest_ip} to a node!")
            continue  # Skip if mapping fails

        # Initialize node statistics if not present
        if src_node not in node_stats:
            node_stats[src_node] = {"tx_packets": 0,
                                    "rx_packets": 0, "delay_sum": 0.0}
        if dest_node not in node_stats:
            node_stats[dest_node] = {"tx_packets": 0,
                                     "rx_packets": 0, "delay_sum": 0.0}

        # Update statistics
        node_stats[src_node]["tx_packets"] += flowStats.txPackets
        node_stats[dest_node]["rx_packets"] += flowStats.rxPackets
        if flowStats.rxPackets > 0:
            node_stats[dest_node]["delay_sum"] += flowStats.delaySum.GetSeconds()

    return node_stats  # Return statistics dictionary

cppyy.cppdef("""
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include <iostream>

using namespace ns3;

class PacketTracer {
public:
    void RxCallback(Ptr<const Packet> packet) {
        UdpHeader udpHeader;
        TcpHeader tcpHeader;
        std::string packetType = "Unknown";
        uint16_t destPort = 0;

        Ptr<Packet> copy = packet->Copy();
        if (copy->PeekHeader(udpHeader)) {
            destPort = udpHeader.GetDestinationPort();
            packetType = "UDP";
        } else if (copy->PeekHeader(tcpHeader)) {
            destPort = tcpHeader.GetDestinationPort();
            packetType = "TCP";
        }

        std::cout << "Recived Packet: " << packet << ", Type: " << packetType << ", Dest Port: " << destPort << ", At time: " << Simulator::Now().GetSeconds()<<  " Size: " << packet->GetSize() << std::endl;
    }
        void TxCallback(Ptr<const Packet> packet) {
        UdpHeader udpHeader;
        TcpHeader tcpHeader;
        std::string packetType = "Unknown";
        uint16_t destPort = 0;

        Ptr<Packet> copy = packet->Copy();
        if (copy->PeekHeader(udpHeader)) {
            destPort = udpHeader.GetDestinationPort();
            packetType = "UDP";
        } else if (copy->PeekHeader(tcpHeader)) {
            destPort = tcpHeader.GetDestinationPort();
            packetType = "TCP";
        }

        std::cout << "Transmitted Packet: " << packet << ", Type: " << packetType << ", Dest Port: " << destPort  <<", At time: " << Simulator::Now().GetSeconds()<<  " Size: " << packet->GetSize() << std::endl;
    }
    

    
    

    static Callback<void, Ptr<const Packet>> CreateRxCallback() {
        PacketTracer* tracer = new PacketTracer();
        return MakeCallback(&PacketTracer::RxCallback, tracer);
    }

    static Callback<void, Ptr<const Packet>> CreateTxCallback() {
        PacketTracer* tracer = new PacketTracer();
        return MakeCallback(&PacketTracer::TxCallback, tracer);
    }
};
""")

PacketTracer = cppyy.gbl.PacketTracer

def setup_packet_tracing_for_router(router):
    for i in range(router.GetNDevices()):
        device = router.GetDevice(i)
        rx_callback = PacketTracer.CreateRxCallback()
        tx_callback = PacketTracer.CreateTxCallback()
        device.TraceConnectWithoutContext("PhyRxEnd", rx_callback)
        device.TraceConnectWithoutContext("PhyTxEnd", tx_callback)

# Assuming 'all_nodes' is a NodeContainer and router 4 is at index 4
def calculate_energy_consumption(node_stats, tx_power=0.0174, rx_power=0.0130, idle_power=0.0008, sleep_power=0.00001, voltage=3.7, total_sim_time=300):
    """ Computes energy consumption based on packet activity """

    print("\n🔋 📊 Final Per-Node Energy Consumption Statistics:")

    for node_id, stats in node_stats.items():
        sent = stats["tx_packets"]
        received = stats["rx_packets"]
        avg_delay = (stats["delay_sum"] / received) if received > 0 else 0

        # Compute estimated times
        tx_time = sent * avg_delay
        rx_time = received * avg_delay
        idle_time = total_sim_time - (tx_time + rx_time)
        sleep_time = 0

        idle_time -= sleep_time  # Remaining idle time

        # Compute energy consumption
        energy_tx = tx_power * tx_time
        energy_rx = rx_power * rx_time
        energy_idle = idle_power * idle_time
        energy_sleep = sleep_power * sleep_time
        total_energy = energy_tx + energy_rx + energy_idle + energy_sleep

        # Print results
        print(f"Node {node_id}:")
        print(f"   📤 Sent: {sent} packets")
        print(f"   📥 Received: {received} packets")
        print(f"   ⏳ Avg Packet Delay: {avg_delay:.6f} sec")
        print(
            f"   🕒 Time in Tx: {tx_time:.2f} sec, Rx: {rx_time:.2f} sec, Idle: {idle_time:.2f} sec, Sleep: {sleep_time:.2f} sec")
        print(f"   ⚡ Energy Consumption (Joules):")
        print(f"       🔴 Transmission: {energy_tx:.6f} J")
        print(f"       🟢 Reception: {energy_rx:.6f} J")
        print(f"       🔵 Idle: {energy_idle:.6f} J")
        print(f"       💤 Sleep: {energy_sleep:.6f} J")
        print(f"   🔋 Total Energy Used: {total_energy:.6f} J\n")


ip_to_node = {}
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

# Create UDP Server on n1
print("Setting up UDP Server")
udpServer = ns.UdpEchoServerHelper(9)  # Port 9
serverApps = udpServer.Install(n1)
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(200.0))

# Create UDP Client on n0
print("Setting up UDP Client")
udpClient = ns.UdpEchoClientHelper(i6.GetAddress(1, 0).ConvertTo(), 9)
udpClient.SetAttribute(
    "MaxPackets", ns.UintegerValue(20000000))

udp_app_interval = 0.1
udpClient.SetAttribute("Interval", ns.TimeValue(
    ns.Seconds(udp_app_interval))) 
udpClient.SetAttribute(
    "PacketSize", ns.UintegerValue(1024))  

clientApps = udpClient.Install(n0)
clientApps.Start(ns.Seconds(3.0))
clientApps.Stop(ns.Seconds(200.0))


flowmonHelper = ns.FlowMonitorHelper()
# monitor = flowmonHelper.Install(all_nodes.Get(4))
monitor = flowmonHelper.InstallAll()

setup_packet_tracing_for_router(all_nodes.Get(3))

# ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

animFile = "./udp_rip4.xml"  
anim = ns.AnimationInterface(animFile)
mobility = ns.MobilityHelper()
mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")

mobility.Install(all_nodes)

anim.SetConstantPosition(all_nodes.Get(0), 0.0, 20.0, 0.0)  
anim.SetConstantPosition(all_nodes.Get(1), 20.0, 20.0, 0.0) 
anim.SetConstantPosition(all_nodes.Get(2), 40.0, 20.0, 0.0) 
anim.SetConstantPosition(all_nodes.Get(3), 12.0, 3.0, 0.0)  
anim.SetConstantPosition(all_nodes.Get(4), 18.0, 3.0, 0.0)  
anim.SetConstantPosition(all_nodes.Get(5), 24.0, 3.0, 0.0)  

print_time = ns.Seconds(10)
end_time = ns.Seconds(300)  

class EventImpl(ns.EventImpl):
    def __init__(self, message, interval, end_time):
        super().__init__()
        self.message = message
        self.interval = interval
        self.end_time = end_time
        self.current_time = ns.Simulator.Now()

    def Notify(self):
        print(
            f"---------------- {self.message} at {self.current_time.GetSeconds()}s ----------------")
        set_interface_state(r0, -1, False)


# event = EventImpl("r0 turned off", print_time, end_time)

ns.Simulator.Stop(ns.Seconds(200.0))
# ns.Simulator.Schedule(print_time, event)
ns.Simulator.Run()

print("\nFinal Energy Report:")
for i in range(all_nodes.GetN()):  # Iterate over all nodes
    energySource = energySources.Get(i)

    # Get energy values
    initial_energy = energySource.GetInitialEnergy()
    remaining_energy = energySource.GetRemainingEnergy()
    total_energy_consumed = initial_energy - remaining_energy  # Manually calculate

    print(f"Node {i}: Initial Energy = {initial_energy} J, Remaining Energy = {remaining_energy} J, Total Consumed = {total_energy_consumed} J")

monitor.CheckForLostPackets()
classifier = flowmonHelper.GetClassifier()
node_stats = analyze_node_statistics(monitor, classifier, ip_addresses)
# calculate_energy_consumption(node_stats)

for flow_id, flowStats in monitor.GetFlowStats():
    flowClass = classifier.FindFlow(flow_id)
    proto = "TCP" if flowClass.protocol == 6 else "UDP"  # Extract protocol
# 
    print(f"📊 Flow {flow_id}: {proto}")
    print(
        f"   Source IP: {flowClass.sourceAddress}, Dest IP: {flowClass.destinationAddress}")
    print(
        f"   Tx Packets: {flowStats.txPackets}, Rx Packets: {flowStats.rxPackets}")
    print(f"   Lost Packets: {flowStats.txPackets - flowStats.rxPackets}")
# 
    # print(
    #     f"   Throughput: {(flowStats.rxBytes/(flowStats.rxPackets*udp_app_interval)) } Bps")
    print(
        f"   Mean Delay: {flowStats.delaySum.GetSeconds()} sec")
    print(
        f"   Mean Jitter: {flowStats.jitterSum.GetSeconds()} sec")
# 
flowmonHelper.SerializeToXmlFile("flow-monitor-results.xml", True, True)
ns.Simulator.Destroy()


