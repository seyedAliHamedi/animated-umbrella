try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "Error: ns3 Python module not found;"
        " Python bindings may not be enabled"
        " or your PYTHONPATH might not be properly configured"
    )


def main(argv):
    cmd = ns.CommandLine()
    cmd.Parse(argv)

    # Create nodes
    print("Create nodes")
    all_nodes = ns.NodeContainer()
    all_nodes.Create(4)  # n0, r0, r1, n1

    n0 = all_nodes.Get(0)
    r0 = all_nodes.Get(1)
    r1 = all_nodes.Get(2)
    n1 = all_nodes.Get(3)

    net1 = ns.NodeContainer()
    net1.Add(n0)
    net1.Add(r0)

    net2 = ns.NodeContainer()
    net2.Add(n0)
    net2.Add(r1)

    net3 = ns.NodeContainer()
    net3.Add(r0)
    net3.Add(n1)

    net4 = ns.NodeContainer()
    net4.Add(r1)
    net4.Add(n1)

    # Install IPv6 Internet Stack
    internetv6 = ns.InternetStackHelper()
    ipv6RoutingHelper = ns.Ipv6ListRoutingHelper()

    ripng = ns.RipNgHelper()
    ipv6RoutingHelper.Add(ripng, 10)  # RIPng priority set to 10

    internetv6.SetRoutingHelper(ipv6RoutingHelper)
    internetv6.Install(all_nodes)

    # Create channels
    csma = ns.CsmaHelper()
    csma.SetChannelAttribute(
        "DataRate", ns.DataRateValue(ns.DataRate(5000000)))
    csma.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(2)))
    csma2 = ns.CsmaHelper()
    csma2.SetChannelAttribute(
        "DataRate", ns.DataRateValue(ns.DataRate(2000000)))
    csma2.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(4)))

    d1 = csma.Install(net1)  # n0 - r0
    d2 = csma2.Install(net2)  # n0 - r1
    d3 = csma.Install(net3)  # r0 - n1
    d4 = csma2.Install(net4)  # r1 - n1

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

    # Create a Ping6 application (n0 -> n1)
    print("Application")
    packetSize = 1024
    maxPacketCount = 5
    interPacketInterval = ns.Seconds(1.0)

    ping = ns.PingHelper(i3.GetAddress(1, 1).ConvertTo())  # Ping n1
    ping.SetAttribute("Count", ns.UintegerValue(maxPacketCount))
    ping.SetAttribute("Interval", ns.TimeValue(interPacketInterval))
    ping.SetAttribute("Size", ns.UintegerValue(packetSize))

    apps = ping.Install(ns.NodeContainer(n0))
    apps.Start(ns.Seconds(2.0))
    apps.Stop(ns.Seconds(20.0))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("dual-router-ping6.tr"))
    csma.EnablePcapAll("dual-router-ping6", True)

    ns.Simulator.Stop(ns.Seconds(25.0))
    # Run Simulation
    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    import sys
    main(sys.argv)
