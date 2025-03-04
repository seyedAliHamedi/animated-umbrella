from ns import ns
import cppyy
from topology.topology import Topology

print("Create nodes")
a =[
    [0,1,1,0],
    [1,0,0,1],
    [1,0,0,1],
    [0,1,1,0],
]
t=Topology(adj_matrix=a,links_type=['p2p'])
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
print("Createdddddd nodes")
internet = ns.InternetStackHelper()
ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()

rip = ns.RipHelper()

ipv4RoutingHelper.Add(rip, 10)

internet.SetRoutingHelper(ipv4RoutingHelper)
internet.Install(end_device)

print("installlllllllled nodes")

p2p = ns.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))

d1 = p2p.Install(net1) 
d2 = p2p.Install(net2)


print("p222222p2p2p2p2p2p2p2p2p")

ipv4 = ns.Ipv4AddressHelper()

ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
i1 = ipv4.Assign(d1)

ipv4.SetBase(ns.Ipv4Address("192.168.2.0"), ns.Ipv4Mask("255.255.255.0"))
i2 = ipv4.Assign(d2)


print("Setting up UDP Server")
udpServer = ns.UdpEchoServerHelper(9)  
serverApps = udpServer.Install(n1)
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(300.0))

print("Setting up UDP Client")
udpClient = ns.UdpEchoClientHelper(i2.GetAddress(0, 0).ConvertTo(), 9)
udpClient.SetAttribute(
    "MaxPackets", ns.UintegerValue(20000000))
udpClient.SetAttribute("Interval", ns.TimeValue(
    ns.Seconds(1.0))) 
udpClient.SetAttribute(
    "PacketSize", ns.UintegerValue(1024)) 

clientApps = udpClient.Install(n0)
clientApps.Start(ns.Seconds(3.0))
clientApps.Stop(ns.Seconds(300.0))


ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

animFile = "./udp_rip4.xml"  
anim = ns.AnimationInterface(animFile)

anim.SetConstantPosition(all_nodes.Get(0), 10.0, 10.0,0) 
anim.SetConstantPosition(all_nodes.Get(1), 16.0, 10.0,0) 
anim.SetConstantPosition(all_nodes.Get(2), 20.0, 15.0,0) 
anim.SetConstantPosition(all_nodes.Get(3), 20.0, 5.0,0)
anim.SetConstantPosition(all_nodes.Get(4), 25.0, 10.0,0)
anim.SetConstantPosition(all_nodes.Get(5), 30.0, 10.0,0)

print_time = ns.Seconds(10)
end_time = ns.Seconds(40)  


ns.Simulator.Stop(ns.Seconds(50.0))
ns.Simulator.Run()
ns.Simulator.Destroy()


