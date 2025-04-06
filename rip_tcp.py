try:
    from ns import ns
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
            # ipv6.GetRoutingProtocol().NotifyAddRoute(i)  # Force update

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
    all_nodes.Create(5)  # n0, r0, r1, r2,n1

    n0 = all_nodes.Get(0)
    r0 = all_nodes.Get(1)
    r1 = all_nodes.Get(2)
    r2 = all_nodes.Get(3)
    n1 = all_nodes.Get(4)

    net1 = ns.NodeContainer()
    net1.Add(n0)
    net1.Add(r0)

    net2 = ns.NodeContainer()
    net2.Add(n0)
    net2.Add(r1)

    net3 = ns.NodeContainer()
    net3.Add(r0)
    net3.Add(r2)

    net4 = ns.NodeContainer()
    net4.Add(r1)
    net4.Add(r2)

    net5 = ns.NodeContainer()
    net5.Add(r2)
    net5.Add(n1)

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

    d1 = csma.Install(net1)  # n0 - r0
    d2 = csma2.Install(net2)  # n0 - r1
    d3 = csma.Install(net3)  # r0 - r2
    d4 = csma2.Install(net4)  # r1 - r2
    d5 = csma.Install(net5)  #r2 - n1

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

    # set_interface_state(r1, -1, False)
    # set_interface_state(r0, -1, False)

    print("Setting up TCP Server")
    tcpServer = ns.PacketSinkHelper("ns3::TcpSocketFactory", ns.InetSocketAddress(ns.Ipv6Address.GetAny(), 9))
    serverApps = tcpServer.Install(n1)
    serverApps.Start(ns.Seconds(1.0))
    serverApps.Stop(ns.Seconds(60.0))


    print("Setting up TCP Client")
    tcpClient = ns.BulkSendHelper("ns3::TcpSocketFactory",
                                ns.InetSocketAddress(i5.GetAddress(1, 1).ConvertTo(), 9))
    tcpClient.SetAttribute("MaxBytes", ns.UintegerValue(0))  # 0 means unlimited data
    tcpClient.SetAttribute("SendSize", ns.UintegerValue(1024))  # Packet size

    clientApps = tcpClient.Install(n0)
    clientApps.Start(ns.Seconds(2.0))
    clientApps.Stop(ns.Seconds(60.0))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("dual-router-udp.tr"))
    csma.EnablePcapAll("dual-router-udp", True)

    ns.LogComponentEnable("BulkSendApplication", ns.LOG_LEVEL_INFO)
    ns.LogComponentEnable("PacketSink", ns.LOG_LEVEL_INFO)
    # ns.LogComponentEnable("RipNg", ns.LOG_LEVEL_ALL)


    print_time = ns.Seconds(10)
    end_time = ns.Seconds(60)  # Stop simulation at 25s
    class EventImpl(ns.EventImpl):
        def __init__(self, message, interval, end_time):
            super().__init__()
            self.message = message
            self.interval = interval
            self.end_time = end_time
            self.current_time = ns.Simulator.Now()

        def Notify(self):
            print(f"---------------- {self.message} at {self.current_time.GetSeconds()}s ----------------")
            set_interface_state(r1, -1, False)
            # Schedule next event if within simulation time
            self.current_time += self.interval
            if self.current_time < self.end_time:
                ns.Simulator.Schedule(self.interval, self)

    # Schedule first event
    event = EventImpl("r0 turned off", print_time, end_time)


        # ✅ Step 1: Install mobility and explicitly set node positions
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(all_nodes)

    # ✅ Assign correct positions before AnimationInterface is initialized
    positions = [
        ns.Vector(0, 3, 0),  # n0
        ns.Vector(6, 0, 0),  # r0
        ns.Vector(6, 6, 0),  # r1
        ns.Vector(12, 3, 0),  # r2
        ns.Vector(18, 3, 0)   # n1
    ]

    for i in range(all_nodes.GetN()):
        mobility_model = all_nodes.Get(i).GetObject[ns.MobilityModel]()
        if mobility_model:
            mobility_model.SetPosition(positions[i])
        else:
            print(f"🚨 Node {i} does not have a mobility model!")

    # ✅ Step 2: Initialize AnimationInterface after mobility setup
    animFile = "./animated-umbrella/rip_udp.xml"
    anim = ns.AnimationInterface(animFile)

    # ✅ Step 3: Set NetAnim positions using correct values
    anim.SetConstantPosition(all_nodes.Get(0), 0, 3)  # n0
    anim.SetConstantPosition(all_nodes.Get(1), 6, 0)  # r0
    anim.SetConstantPosition(all_nodes.Get(2), 6, 6)  # r1
    anim.SetConstantPosition(all_nodes.Get(3), 12, 3)  # r2
    anim.SetConstantPosition(all_nodes.Get(4), 18, 1.5)  # n1

    # ✅ Step 4: Print final node positions to verify
    print("\n✅ Final Node Positions:")
    for i in range(all_nodes.GetN()):
        pos = all_nodes.Get(i).GetObject[ns.MobilityModel]().GetPosition()
        print(f"Node {i}: x={pos.x}, y={pos.y}")

    def get_all_ipv6_addresses(node):
        """ Retrieves all IPv6 addresses assigned to a node's interfaces. """
        ipv6 = node.GetObject[ns.Ipv6]()
        if ipv6 is None:
            return ["No IPv6"]

        num_interfaces = ipv6.GetNInterfaces()
        addresses = []

        for i in range(1,num_interfaces):
            for j in range(ipv6.GetNAddresses(i)):
                addr = ipv6.GetAddress(i, j).GetAddress()    
                if not addr.IsLinkLocal():  # Ignore link-local addresses
                    addresses.append(f"{addr.ConvertTo()} (iface {i})\n")

        return addresses

    for i, node in enumerate([n0, r0, r1, r2, n1]):
        ip_list = get_all_ipv6_addresses(node)
        ip_text = "&shgmnghmnb#10;".join(ip_list)  # Format multiple IPs
        anim.SetConstantPosition(node, positions[i].x, positions[i].y)
        anim.UpdateNodeDescription(node, f"Node {i} &ssderftyhjh#10; {ip_text}")  # Show all interfaces

    print("\n✅ Assigned IPv6 Addresses for Each Interface:")
    for i, node in enumerate([n0, r0, r1, r2, n1]):
        ip_list = get_all_ipv6_addresses(node)
        print(f"Node {i} Interfaces:\n" + "\n".join(ip_list))

    ns.Simulator.Stop(ns.Seconds(60.0))
    ns.Simulator.Schedule(print_time, event)
    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    import sys
    main(sys.argv)
