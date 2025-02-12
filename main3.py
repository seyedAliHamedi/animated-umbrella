from ns import ns


ns.cppyy.cppdef("""
            using namespace ns3;

            Callback<void,Ptr<const Packet>,const Address&,const Address&>
            make_sinktrace_callback(void(*func)(Ptr<const Packet>, const Address&,const Address&))
            {
                return MakeCallback(func);
            }
        """)

def SinkTracer(packet: ns.Packet, src_address: ns.Address, dst_address: ns.Address) -> None:
    print(f"At {ns.Simulator.Now().GetSeconds():.0f}s, '{dst_address}' received packet"
          f" with {packet.__deref__().GetSerializedSize()} bytes from '{src_address}'")

# Create network topology
# ======================

# Create HUB nodes (end devices)
hub1 = ns.NodeContainer()
hub1.Create(5)
hub2 = ns.NodeContainer()
hub2.Create(5)
hub3 = ns.NodeContainer()
hub3.Create(5)

# Create routers
routers = ns.NodeContainer()
routers.Create(6)

# Combine all nodes
all_nodes = ns.NodeContainer()
all_nodes.Add(hub1)
all_nodes.Add(hub2)
all_nodes.Add(hub3)
all_nodes.Add(routers)

# Install internet stack
internet = ns.InternetStackHelper()
internet.Install(all_nodes)

# Create connections
p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("1Gbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))

# Connect HUBs to routers
hub1_devices = ns.NetDeviceContainer()
for i in range(hub1.GetN()):
    hub1_devices.Add(p2p.Install(hub1.Get(i), routers.Get(0)))

hub2_devices = ns.NetDeviceContainer()
for i in range(hub2.GetN()):
    hub2_devices.Add(p2p.Install(hub2.Get(i), routers.Get(2)))

hub3_devices = ns.NetDeviceContainer()
for i in range(hub3.GetN()):
    hub3_devices.Add(p2p.Install(hub3.Get(i), routers.Get(4)))

# Connect routers in full mesh
router_devices = ns.NetDeviceContainer()
for i in range(routers.GetN()):
    for j in range(i+1, routers.GetN()):
        router_devices.Add(p2p.Install(routers.Get(i), routers.Get(j)))

# Assign IP addresses
ipv4 = ns.Ipv4AddressHelper()

# Hub1: 192.168.1.0/24
ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
hub1_ips = ipv4.Assign(hub1_devices)

# Hub2: 192.169.1.0/24
ipv4.SetBase(ns.Ipv4Address("192.169.1.0"), ns.Ipv4Mask("255.255.255.0"))
hub2_ips = ipv4.Assign(hub2_devices)

# Hub3: 10.1.0.0/16
ipv4.SetBase(ns.Ipv4Address("10.1.0.0"), ns.Ipv4Mask("255.255.0.0"))
hub3_ips = ipv4.Assign(hub3_devices)

# Routers: 192.168.5.0/27
ipv4.SetBase(ns.Ipv4Address("192.168.5.0"), ns.Ipv4Mask("255.255.255.224"))
router_ips = ipv4.Assign(router_devices)

# Configure static routing
def add_route(router, dest, mask, next_hop):
    ipv4 = router.GetObject[ns.Ipv4]()
    static = ns.Ipv4StaticRoutingHelper().GetStaticRouting(ipv4)
    static.AddNetworkRouteTo(ns.Ipv4Address(dest), ns.Ipv4Mask(mask), 
                           ns.Ipv4Address(next_hop), 1)

# R0 routes
add_route(routers.Get(0), "192.169.1.0", "255.255.255.0", "192.168.5.2")  # To R1
add_route(routers.Get(0), "10.1.0.0", "255.255.0.0", "192.168.6.2")     # To R5

# R1 routes
add_route(routers.Get(1), "192.168.1.0", "255.255.255.0", "192.168.5.1")  # To R0
add_route(routers.Get(1), "192.169.1.0", "255.255.255.0", "192.168.7.2")  # To R2

# R2 routes
add_route(routers.Get(2), "192.168.1.0", "255.255.255.0", "192.168.7.1")  # To R1
add_route(routers.Get(2), "10.1.0.0", "255.255.0.0", "192.168.8.2")      # To R3

# R3 routes
add_route(routers.Get(3), "192.169.1.0", "255.255.255.0", "192.168.8.1")  # To R2
add_route(routers.Get(3), "10.1.0.0", "255.255.0.0", "192.168.9.2")      # To R4

# R4 routes
add_route(routers.Get(4), "192.169.1.0", "255.255.255.0", "192.168.9.1")  # To R3
add_route(routers.Get(4), "192.168.1.0", "255.255.255.0", "192.168.10.2") # To R5

# R5 routes
add_route(routers.Get(5), "192.168.1.0", "255.255.255.0", "192.168.10.1") # To R0
add_route(routers.Get(5), "10.1.0.0", "255.255.0.0", "192.168.6.1")      # To R4


print("\n=== Network Configuration ===")
for i in range(all_nodes.GetN()):
    node = all_nodes.Get(i)
    ipv4 = node.GetObject[ns.Ipv4]()
    print(f"Node {i} ({'Router' if i >= 15 else 'End Device'}):")
    for j in range(ipv4.GetNInterfaces()):
        addr = ipv4.GetAddress(j, 0)
        print(f"  Interface {j}: {addr.GetLocal()}")


# Create applications
# ==================
server_port = 9
server_ip = hub2_ips.GetAddress(0).ConvertTo()

# Server application
server = ns.UdpEchoServerHelper(server_port)
server_app = server.Install(hub2.Get(0))
server_app.Start(ns.Seconds(1.0))
server_app.Stop(ns.Seconds(10.0))

# Client application
client = ns.UdpEchoClientHelper(server_ip, server_port)
client.SetAttribute("MaxPackets", ns.UintegerValue(10))
client.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
client.SetAttribute("PacketSize", ns.UintegerValue(1024))

client_app = client.Install(hub1.Get(0))
client_app.Start(ns.Seconds(2.0))
client_app.Stop(ns.Seconds(10.0))

# Enable tracing
sinkTraceCallback = ns.cppyy.gbl.make_sinktrace_callback(SinkTracer)
server_app.Get(0).__deref__().TraceConnectWithoutContext("RxWithAddresses", sinkTraceCallback);

# Simulation Control ----------------------------------------------
ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("Ipv4StaticRouting", ns.LOG_LEVEL_DEBUG)

ns.Simulator.Stop(ns.Seconds(11.0))
print("\n=== Starting Simulation ===")
ns.Simulator.Run()
ns.Simulator.Destroy()
print("=== Simulation Complete ===")