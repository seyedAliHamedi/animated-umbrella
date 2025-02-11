from ns import ns
import cppyy

# Step 1: Create HUB nodes (end devices)
hub1_devices = ns.NodeContainer()
hub1_devices.Create(5)  # 5 devices in HUB1

hub2_devices = ns.NodeContainer()
hub2_devices.Create(5)  # 5 devices in HUB2

hub3_devices = ns.NodeContainer()
hub3_devices.Create(5)  # 5 devices in HUB3

# Step 2: Create 6 routers (R0 to R5)
routers = ns.NodeContainer()
routers.Create(6)

# Add all nodes to a container
allNodes = ns.NodeContainer()
allNodes.Add(hub1_devices)
allNodes.Add(hub2_devices)
allNodes.Add(hub3_devices)
allNodes.Add(routers)


# Step 3: Install Internet Stack
internet = ns.InternetStackHelper()
internet.Install(allNodes)

# Step 4: Create PointToPoint helper
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("1Gbps"))  # Set bandwidth
p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))      # Set delay

# Containers for network devices
hub1_net_devices = ns.NetDeviceContainer()
hub2_net_devices = ns.NetDeviceContainer()
hub3_net_devices = ns.NetDeviceContainer()
router_net_devices = ns.NetDeviceContainer()

# Connect HUB1 to R0
for i in range(hub1_devices.GetN()):
    hub1_net_devices.Add(p2p.Install(hub1_devices.Get(i), routers.Get(0)))

# Connect HUB2 to R2
for i in range(hub2_devices.GetN()):
    hub2_net_devices.Add(p2p.Install(hub2_devices.Get(i), routers.Get(2)))

# Connect HUB3 to R4
for i in range(hub3_devices.GetN()):
    hub3_net_devices.Add(p2p.Install(hub3_devices.Get(i), routers.Get(4)))

# Fully connect the 6 routers (R0 to R5)
for i in range(routers.GetN()):
    for j in range(i + 1, routers.GetN()):
        router_net_devices.Add(p2p.Install(routers.Get(i), routers.Get(j)))


# Step 5: Assign IP Addresses
address = ns.Ipv4AddressHelper()

# Assign IPs for HUB1
address.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
hub1_interfaces = address.Assign(hub1_net_devices)

# Assign IPs for HUB2
address.SetBase(ns.Ipv4Address("192.169.1.0"), ns.Ipv4Mask("255.255.255.0"))
hub2_interfaces = address.Assign(hub2_net_devices)

# Assign IPs for HUB3
address.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.0.0")) # /16 subnet (32 IPs)
hub3_interfaces = address.Assign(hub3_net_devices)

router_address = ns.Ipv4AddressHelper()
router_address.SetBase(ns.Ipv4Address("192.168.5.0"), ns.Ipv4Mask("255.255.255.224"))  # /27 subnet (32 IPs)
router_interfaces = router_address.Assign(router_net_devices)




# hub1 <->hub2 : r0<->r1<->r2
# hub2 <->hub3 : r2<->r3<->r4
# hub1 <->hub3 : r0<->r5<->r4

# Step 6: Set Up Basic Static Routing
static_routing_helper = ns.Ipv4StaticRoutingHelper()

# Function to add a static route
def add_static_route(router, destination, mask, next_hop):
    ipv4 = router.GetObject[ns.Ipv4]() 
    static_router = static_routing_helper.GetStaticRouting(ipv4)  

    static_router.AddNetworkRouteTo(ns.Ipv4Address(destination), ns.Ipv4Mask(mask), ns.Ipv4Address(next_hop), 1)
    print(f"✅ Route added on Router {router.GetId()} → {destination} via {next_hop}")


# Routes for R0 (HUB1 <-> HUB2, HUB1 <-> HUB3)
add_static_route(routers.Get(0), "192.169.1.0", "255.255.255.0", "192.168.5.2")  # Route to HUB2 via R1
add_static_route(routers.Get(0), "10.1.0.0", "255.255.0.0", "192.168.6.2")  # Route to HUB3 via R5

# Routes for R1 (HUB1 <-> HUB2)
add_static_route(routers.Get(1), "192.168.1.0", "255.255.255.0", "192.168.5.1")  # Route to HUB1 via R0
add_static_route(routers.Get(1), "192.169.1.0", "255.255.255.0", "192.168.7.2")  # Route to HUB2 via R2

# Routes for R2 (HUB1 <-> HUB2 and HUB2 <-> HUB3)
add_static_route(routers.Get(2), "192.168.1.0", "255.255.255.0", "192.168.7.1")  # Route to HUB1 via R1
add_static_route(routers.Get(2), "10.1.0.0", "255.255.0.0", "192.168.8.2")  # Route to HUB3 via R3

# Routes for R3 (HUB2 <-> HUB3)
add_static_route(routers.Get(3), "192.169.1.0", "255.255.255.0", "192.168.8.1")  # Route to HUB2 via R2
add_static_route(routers.Get(3), "10.1.0.0", "255.255.0.0", "192.168.9.2")  # Route to HUB3 via R4

# Routes for R4 (HUB2 <-> HUB3, HUB1 <-> HUB3)
add_static_route(routers.Get(4), "192.169.1.0", "255.255.255.0", "192.168.9.1")  # Route to HUB2 via R3
add_static_route(routers.Get(4), "192.168.1.0", "255.255.255.0", "192.168.10.2")  # Route to HUB1 via R5

# Routes for R5 (HUB1 <-> HUB3)
add_static_route(routers.Get(5), "192.168.1.0", "255.255.255.0", "192.168.10.1")  # Route to HUB1 via R0
add_static_route(routers.Get(5), "10.1.0.0", "255.255.0.0", "192.168.6.1")  # Route to HUB3 via R4


for i in range(allNodes.GetN()):
    node = allNodes.Get(i)
    ipv4 = node.GetObject[ns.Ipv4]()  # Get the IPv4 object
    print(f"Node {i}:")
    for j in range(ipv4.GetNInterfaces()):
        ip = ipv4.GetAddress(j, 0)  # Get the assigned IP
        print(f"  Interface {j}: {ip}")

