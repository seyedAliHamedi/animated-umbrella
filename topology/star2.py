from ns import ns
import cppyy

################# STAR TOPOLOGY #################
"""
            E1
            |
    E2-- [Router] --E3
            |
            E4
"""
# ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
print("\n\n\n")
# define end devices
endDevices = ns.NodeContainer()
endDevices.Create(4)

# define router
routers = ns.NodeContainer()
routers.Create(1)
router = routers.Get(0)

# define p2p links
link = ns.PointToPointHelper()
link.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
link.SetChannelAttribute("Delay", ns.StringValue("2ms"))

# Install devices
devices = ns.NetDeviceContainer()
for i in range(endDevices.GetN()):
    linkDevices = link.Install(endDevices.Get(i), router)
    devices.Add(linkDevices.Get(0))
    devices.Add(linkDevices.Get(1))

# install internet stack
stack = ns.InternetStackHelper()
stack.Install(router)
stack.Install(endDevices)

# assign ip
address = ns.Ipv4AddressHelper()
address.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
interfaces = address.Assign(devices)


# ns3::Address& ip, uint16_t port
port = 9
server = endDevices.Get(3)
server_ip = interfaces.GetAddress(3).ConvertTo()


# create udp server
udpServer = ns.UdpEchoServerHelper(port)
serverApp = udpServer.Install(server)
serverApp.Start(ns.Seconds(1.0))
serverApp.Stop(ns.Seconds(10.0))

# create udp client
udpClient = ns.UdpEchoClientHelper(server_ip, port)
udpClient.SetAttribute("MaxPackets", ns.UintegerValue(5))
udpClient.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
udpClient.SetAttribute("PacketSize", ns.UintegerValue(512))

# install udp client
clientApp = udpClient.Install(endDevices.Get(0))
clientApp.Start(ns.Seconds(2.0))
clientApp.Stop(ns.Seconds(10.0))

clientApp2 = udpClient.Install(endDevices.Get(2))
clientApp2.Start(ns.Seconds(2.0))
clientApp2.Stop(ns.Seconds(10.0))


# initalize flow monitor
flowmonHelper = ns.FlowMonitorHelper()
monitor = flowmonHelper.InstallAll()


# ? wireshark
# link.EnablePcapAll("star_topology")


# ? stop simulator
ns.Simulator.Stop(ns.Seconds(12.0))

# run the simulation
ns.Simulator.Run()
print("-------- simulation started --------- \n")

monitor.CheckForLostPackets()
classifier = flowmonHelper.GetClassifier()

for flow_id, flowStats in monitor.GetFlowStats():
    flowClass = classifier.FindFlow(flow_id)
    proto = "TCP" if flowClass.protocol == 6 else "UDP"  # Extract protocol

    print(f"📊 Flow {flow_id}: {proto}")
    print(
        f"   Source IP: {flowClass.sourceAddress}, Dest IP: {flowClass.destinationAddress}")
    print(
        f"   Tx Packets: {flowStats.txPackets}, Rx Packets: {flowStats.rxPackets}")
    print(f"   Lost Packets: {flowStats.lostPackets}")
    # Convert bytes to Mbps
    print(f"   Throughput: {flowStats.rxBytes * 8 / 10**6} Mbps")
    print(
        f"   Mean Delay: {flowStats.delaySum.GetSeconds() / flowStats.rxPackets} sec")
    print(
        f"   Mean Jitter: {flowStats.jitterSum.GetSeconds() / flowStats.rxPackets} sec")

flowmonHelper.SerializeToXmlFile("flow-monitor-results.xml", True, True)
ns.Simulator.Destroy()
