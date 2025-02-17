try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "Error: ns3 Python module not found; Python bindings may not be enabled"
    )



def main(argv):
    cmd = ns.CommandLine()
    cmd.Parse(argv)

    print("Create nodes")
    all = ns.NodeContainer(3)
    net1 = ns.NodeContainer()
    net1.Add(all.Get(0))
    net1.Add(all.Get(1))
    net2 = ns.NodeContainer()
    net2.Add(all.Get(1))
    net2.Add(all.Get(2))



    # Create IPv4 Internet Stack
    internetv4 = ns.InternetStackHelper()
    internetv4.Install(all)

    # Create channels
    csma = ns.CsmaHelper()
    csma.SetChannelAttribute("DataRate", ns.DataRateValue(ns.DataRate(5000000)))
    csma.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(2)))
    d1 = csma.Install(net1)
    d2 = csma.Install(net2)

    # Create networks and assign IPv4 Addresses
    print("Addressing")
    ipv4 = ns.Ipv4AddressHelper()
    ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))  # Use Ipv4Mask
    ipv4.Assign(d1)
    node1_ipv4 = net1.Get(0).GetObject[ns.Ipv4]()
    node1_ipv4.SetForwarding(1, True)
    #node1_ipv4.SetDefaultRouteInAllNodes(1)


    ipv4.SetBase(ns.Ipv4Address("192.168.2.0"), ns.Ipv4Mask("255.255.255.0"))  # Use Ipv4Mask
    ipv4.Assign(d2)
    node2_ipv4 = net2.Get(0).GetObject[ns.Ipv4]()
    node2_ipv4.SetForwarding(0, True)
    #i2.SetDefaultRouteInAllNodes(0)

    # Create a Ping application to send ICMP echo request from n0 to n1 via r
    print("Application")
    packetSize = 1024
    maxPacketCount = 5
    interPacketInterval = ns.Seconds(1)
    local_address = node1_ipv4.GetAddress(0, 0)  # Get the first address of the first node
    remote_address = node1_ipv4.GetAddress(1, 0)  # Get the first address of the second node
    print(type(node1_ipv4),node1_ipv4)
    print(type(remote_address),remote_address)
    print(type(remote_address.GetLocal()),remote_address.GetLocal())
    print(type(remote_address.GetLocal().ConvertTo()),remote_address.GetLocal().ConvertTo())
    ping = ns.PingHelper(remote_address.GetLocal().ConvertTo())
    # ping.SetLocal(local_address.GetLocal())  # Local address of n0
    # ping.SetRemote(remote_address.GetLocal())  # Remote address of n1

    ping.SetAttribute("Count", ns.UintegerValue(maxPacketCount))
    ping.SetAttribute("Interval", ns.TimeValue(interPacketInterval))
    ping.SetAttribute("Size", ns.UintegerValue(packetSize))

    apps = ping.Install(ns.NodeContainer(net1.Get(0)))
    apps.Start(ns.Seconds(2))
    apps.Stop(ns.Seconds(20))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("simple-routing-ping.tr"))
    csma.EnablePcapAll("simple-routing-ping", True)

    # Run Simulation
    ns.Simulator.Run()
    ns.Simulator.Destroy()

if __name__ == "__main__":
    import sys
    main(sys.argv)
