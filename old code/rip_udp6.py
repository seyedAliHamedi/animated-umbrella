try:
    from ns import ns
    import cppyy
    cppyy.include("ns3/event-impl.h")
except ModuleNotFoundError:
    raise SystemExit(
        "Error: ns3 Python module not found;"
        " Python bindings may not be enabled"
        " or your PYTHONPATH might not be properly configured"
    )


def set_interface_state(r, interface_index, state):
    """
    Parameters:
    r - The router node
    interface_index - The interface to modify (-1 for all interfaces)
    state - Boolean (True = UP, False = DOWN)
    """
    ipv6 = r.GetObject[ns.Ipv6]()  # Get the IPv6 stack
    num_interfaces = ipv6.GetNInterfaces()  # Get total interfaces

    if interface_index == -1:
        # Apply to all interfaces (except loopback, usually index 0)
        for i in range(1, num_interfaces):
            ipv6.SetUp(i) if state else ipv6.SetDown(i)
            ipv6.GetRoutingProtocol().NotifyInterfaceDown(
                i) if not state else ipv6.GetRoutingProtocol().NotifyInterfaceUp(i)
        print(
            f"Router {r.GetId()} {'enabled' if state else 'disabled'} (all {num_interfaces-1} interfaces)")
    else:
        # Apply only to the specified interface
        if 0 < interface_index < num_interfaces:
            ipv6.SetUp(interface_index) if state else ipv6.SetDown(
                interface_index)
            ipv6.GetRoutingProtocol().NotifyInterfaceDown(
                i) if not state else ipv6.GetRoutingProtocol().NotifyInterfaceUp(i)
            print(
                f"Router {r.GetId()} {'enabled' if state else 'disabled'} (interface {interface_index})")
        else:
            print(
                f"Invalid interface index {interface_index} for router {r.GetId()}")


def main(argv):
    cmd = ns.CommandLine()
    cmd.Parse(argv)

    # Create nodes
    print("Create nodes")
    all_nodes = ns.NodeContainer()
    all_nodes.Create(6)  # n0, r0, r1, r2, r3,n1

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

    # Install IPv6 Internet Stack
    internetv6 = ns.InternetStackHelper()
    ipv6RoutingHelper = ns.Ipv6ListRoutingHelper()

    ripng = ns.RipNgHelper()

    ipv6RoutingHelper.Add(ripng, 10)  # RIPng priority set to 10

    internetv6.SetRoutingHelper(ipv6RoutingHelper)
    internetv6.Install(all_nodes)

    # Create channels
    csma2 = ns.CsmaHelper()
    csma2.SetChannelAttribute(
        "DataRate", ns.DataRateValue(ns.DataRate(5000000)))
    csma2.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(2)))

    csma = ns.CsmaHelper()
    csma.SetChannelAttribute(
        "DataRate", ns.DataRateValue(ns.DataRate(2000000)))
    csma.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(4)))
    csma.SetQueue("ns3::DropTailQueue", "MaxSize",
                  ns.QueueSizeValue(ns.QueueSize("10p")))

    d1 = csma.Install(net1)  # n0 - r0
    d2 = csma2.Install(net2)  # n0 - r1
    d3 = csma.Install(net3)  # r0 - r3
    d4 = csma2.Install(net4)  # r1 - r2
    d5 = csma2.Install(net5)  # r2 - r3
    d6 = csma.Install(net6)  # r3 - n1

    # Assign IPv6 Addresses
    print("Addressing")
    ipv6 = ns.Ipv6AddressHelper()

    ipv6.SetBase(ns.Ipv6Address("2001:1::"), ns.Ipv6Prefix(64))
    i1 = ipv6.Assign(d1)

    ipv6.SetBase(ns.Ipv6Address("2001:2::"), ns.Ipv6Prefix(64))
    i2 = ipv6.Assign(d2)

    ipv6.SetBase(ns.Ipv6Address("2001:3::"), ns.Ipv6Prefix(64))
    i3 = ipv6.Assign(d3)

    ipv6.SetBase(ns.Ipv6Address("2001:4::"), ns.Ipv6Prefix(64))
    i4 = ipv6.Assign(d4)

    ipv6.SetBase(ns.Ipv6Address("2001:5::"), ns.Ipv6Prefix(64))
    i5 = ipv6.Assign(d5)

    ipv6.SetBase(ns.Ipv6Address("2001:6::"), ns.Ipv6Prefix(64))
    i6 = ipv6.Assign(d6)

    # set_interface_state(r1, -1, False)
    # set_interface_state(r0, -1, False)

    # Create UDP Server on n1
    print("Setting up UDP Server")
    udpServer = ns.UdpEchoServerHelper(9)  # Port 9
    serverApps = udpServer.Install(n1)
    serverApps.Start(ns.Seconds(1.0))
    serverApps.Stop(ns.Seconds(300.0))

    # Create UDP Client on n0
    print("Setting up UDP Client")
    udpClient = ns.UdpEchoClientHelper(i6.GetAddress(1, 1).ConvertTo(), 9)
    udpClient.SetAttribute(
        "MaxPackets", ns.UintegerValue(20000000))  # Send 20 packets
    udpClient.SetAttribute("Interval", ns.TimeValue(
        ns.Seconds(1.0)))  # 1-second interval
    udpClient.SetAttribute(
        "PacketSize", ns.UintegerValue(1024))  # 1024-byte packets

    clientApps = udpClient.Install(n0)
    clientApps.Start(ns.Seconds(2.0))
    clientApps.Stop(ns.Seconds(300.0))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("udp_rip6.tr"))
    csma.EnablePcapAll("udp_rip6", True)

    ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
    ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

    # animFile = "./animated-umbrella/udp_rip6.xml"  # Output XML file for NetAnim
    # anim = ns.AnimationInterface(animFile)

    # # 🚀 Optional: Set node descriptions in NetAnim
    # anim.SetConstantPosition(all_nodes.Get(0), 0.0, 3.0)  # n0
    # anim.SetConstantPosition(all_nodes.Get(1), 6.0, 0.0)  # r0
    # anim.SetConstantPosition(all_nodes.Get(2), 6.0, 6.0)  # r1
    # anim.SetConstantPosition(all_nodes.Get(3), 12.0, 3.0)  # r2
    # anim.SetConstantPosition(all_nodes.Get(4), 18.0, 3.0)  # n1
    # anim.SetConstantPosition(all_nodes.Get(5), 18.0, 3.0)  # n1

    print_time = ns.Seconds(10)
    end_time = ns.Seconds(300)  # Stop simulation at 25s

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
            # Schedule next event if within simulation time
            self.current_time += self.interval
            if self.current_time < self.end_time:
                ns.Simulator.Schedule(self.interval, self)
    # Schedule first event
    event = EventImpl("r0 turned off", print_time, end_time)
    # print("mmd")

    ns.Simulator.Stop(ns.Seconds(300.0))
    ns.Simulator.Schedule(print_time, event)
    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    import sys
    main(sys.argv)
