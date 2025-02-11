from ns import ns
import cppyy

# Step 1: Create the two endpoints (n0 and n1)
endpoints = ns.NodeContainer()
endpoints.Create(2)  # n0 (source), n1 (destination)

# Step 2: Create the four routers (r0, r1, r2, r3)
routers = ns.NodeContainer()
routers.Create(4)  # r0, r1, r2, r3

# Add all nodes to a container
allNodes = ns.NodeContainer()
allNodes.Add(endpoints)
allNodes.Add(routers)

# Step 3: Install the Internet Stack
internet = ns.InternetStackHelper()
internet.Install(allNodes)

# Step 4: Create network devices and assign them to the nodes
devices = ns.NetDeviceContainer()
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("5ms"))

# Create the links between nodes (P2P connections) and get devices
devices.Add(p2p.Install(endpoints.Get(0), routers.Get(0)))  # n0 <-> r0
devices.Add(p2p.Install(routers.Get(0), routers.Get(1)))    # r0 <-> r1
devices.Add(p2p.Install(routers.Get(1), endpoints.Get(1)))  # r1 <-> n1
devices.Add(p2p.Install(routers.Get(0), routers.Get(2)))    # r0 <-> r2
devices.Add(p2p.Install(routers.Get(2), routers.Get(3)))    # r2 <-> r3
devices.Add(p2p.Install(routers.Get(3), endpoints.Get(1)))  # r3 <-> n1
devices.Add(p2p.Install(routers.Get(1), routers.Get(2)))    # r1 <-> r2

# Step 5: Create bridge for each node (except the endpoints, which only have one link)
for i in range(routers.GetN()):
    node =q
    
    bridge = ns.BridgeHelper()
    bridge.Install(node, devices)

# Step 6: Assign IP addresses to each device (node) using a single address per node
address = ns.Ipv4AddressHelper()
address.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))

# Assign IPs to devices and capture the interfaces
ipv4_interfaces = address.Assign(devices)

# Print the IP addresses
for j in range(ipv4_interfaces.GetN()):
    ipv4 = ipv4_interfaces.GetAddress(j, 0)  # This will give you the local IP address directly
    print(ipv4)

# Run the simulation
ns.Simulator.Run()
ns.Simulator.Destroy()
