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
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("10Mbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("5ms"))

# Create the links between nodes (P2P connections)
devices = ns.NetDeviceContainer()
links = []  # Store device containers for IP assignment

links.append(p2p.Install(endpoints.Get(0), routers.Get(0)))  # n0 <-> r0
links.append(p2p.Install(routers.Get(0), routers.Get(1)))    # r0 <-> r1
links.append(p2p.Install(routers.Get(1), endpoints.Get(1)))  # r1 <-> n1
links.append(p2p.Install(routers.Get(0), routers.Get(2)))    # r0 <-> r2
links.append(p2p.Install(routers.Get(2), routers.Get(3)))    # r2 <-> r3
links.append(p2p.Install(routers.Get(3), endpoints.Get(1)))  # r3 <-> n1
links.append(p2p.Install(routers.Get(1), routers.Get(2)))    # r1 <-> r2

# Step 5: Assign Unique IP Addresses to Each Link
ipv4 = ns.Ipv4AddressHelper()
subnet_prefix = "10.1.{}.0"

interfaces = []  # Store interfaces for later use

for i, link in enumerate(links):
    ipv4.SetBase(ns.Ipv4Address(subnet_prefix.format(i + 1)), ns.Ipv4Mask("255.255.255.0"))
    interfaces.append(ipv4.Assign(link))

# Step 6: Print IP Addresses
for i, iface in enumerate(interfaces):
    print(f"Link {i + 1} - Network: {subnet_prefix.format(i + 1)}")
    print(f"  Node 1 IP: {iface.GetAddress(0)}")
    print(f"  Node 2 IP: {iface.GetAddress(1)}")

# Run the simulation
ns.Simulator.Run()
ns.Simulator.Destroy()
