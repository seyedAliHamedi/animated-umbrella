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

# Step 3: Install Internet Stack
internet = ns.InternetStackHelper()
internet.Install(hub1_devices)
internet.Install(hub2_devices)
internet.Install(hub3_devices)
internet.Install(routers)

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
address.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.0.0"))
hub3_interfaces = address.Assign(hub3_net_devices)

# Assign IPs for routers
router_address = ns.Ipv4AddressHelper()
router_address.SetBase(ns.Ipv4Address("192.168.5.0"), ns.Ipv4Mask("255.255.255.224"))  # /27 subnet (32 IPs)
router_interfaces = router_address.Assign(router_net_devices)



print("Done")