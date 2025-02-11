from ns import ns
import cppyy

# Enable logging for debugging purposes
ns.LogComponentEnable("Ipv4GlobalRouting", ns.LOG_LEVEL_ALL)

# Create the nodes
endpoints = ns.NodeContainer()
endpoints.Create(2)  # n0 (source), n1 (destination)

routers = ns.NodeContainer()
routers.Create(4)  # r0, r1, r2, r3

allNodes = ns.NodeContainer()
allNodes.Add(endpoints)
allNodes.Add(routers)

# Point-to-Point helpers
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("5ms"))

# Connect nodes
devices = []
devices.append(p2p.Install(endpoints.Get(0), routers.Get(0)))  # n0 <-> r0
devices.append(p2p.Install(routers.Get(0), routers.Get(1)))    # r0 <-> r1
devices.append(p2p.Install(routers.Get(1), endpoints.Get(1)))  # r1 <-> n1
devices.append(p2p.Install(routers.Get(0), routers.Get(2)))    # r0 <-> r2
devices.append(p2p.Install(routers.Get(2), routers.Get(3)))    # r2 <-> r3
devices.append(p2p.Install(routers.Get(3), endpoints.Get(1)))  # r3 <-> n1
devices.append(p2p.Install(routers.Get(1), routers.Get(2)))    # r1 <-> r2

# Install Internet stack
internet = ns.InternetStackHelper()
internet.Install(allNodes)

# Assign IP addresses
address = ns.Ipv4AddressHelper()
ip_interfaces = []

for i, dev in enumerate(devices):
    subnet = f"10.1.{i + 1}.0"
    address.SetBase(ns.Ipv4Address(subnet), ns.Ipv4Mask("255.255.255.0"))
    ip_interfaces.append(address.Assign(dev))

# Populate routing tables
# ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

# UDP Echo Server on Node n1 (Receiver)
echoServer = ns.UdpEchoServerHelper(9)  # Port 9
serverApps = echoServer.Install(endpoints.Get(1))  # Install on n1 (Receiver)
serverApps.Start(ns.Seconds(1))  # Start at 1s
serverApps.Stop(ns.Seconds(10))  # Stop at 10s

# Get the IP address of the server (n1)
serverAddress = ip_interfaces[2].GetAddress(1).ConvertTo()

# UDP Echo Client on Node n0 (Sender)
echoClient = ns.UdpEchoClientHelper(serverAddress, 9)  # Server's address and port
echoClient.SetAttribute("MaxPackets", ns.UintegerValue(1))  # Send 1 packet
echoClient.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1)))  # Interval between packets
echoClient.SetAttribute("PacketSize", ns.UintegerValue(1024))  # Packet size in bytes

# Install the client application on Node n0 (Sender)
clientApps = echoClient.Install(endpoints.Get(0))  # Install on n0 (Sender)
clientApps.Start(ns.Seconds(2))  # Start at 2s
clientApps.Stop(ns.Seconds(10))  # Stop at 10s

# Flow monitor setup
flowmonHelper = ns.FlowMonitorHelper()
# monitor = flowmonHelper.InstallAll()

# Run the simulation
ns.Simulator.Run()
ns.Simulator.Destroy()
print("end")