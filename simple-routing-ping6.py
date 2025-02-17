#
# IPv4 Simple Routing Example
#

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
    all_nodes.Create(3)  # n0, r (router), n1

    net1 = ns.NodeContainer()
    net1.Add(all_nodes.Get(0))
    net1.Add(all_nodes.Get(1))  # Connect n0 to router

    net2 = ns.NodeContainer()
    net2.Add(all_nodes.Get(1))
    net2.Add(all_nodes.Get(2))  # Connect router to n1

    # Create IPv4 Internet Stack
    internet = ns.InternetStackHelper()
    internet.Install(all_nodes)

    # Create channels using CSMA
    csma = ns.CsmaHelper()
    csma.SetChannelAttribute("DataRate", ns.StringValue("5Mbps"))
    csma.SetChannelAttribute("Delay", ns.StringValue("2ms"))

    d1 = csma.Install(net1)
    d2 = csma.Install(net2)

    # Assign IPv4 Addresses
    print("Addressing")
    ipv4 = ns.Ipv4AddressHelper()

    ipv4.SetBase(ns.Ipv4Address("192.168.1.0"), ns.Ipv4Mask("255.255.255.0"))
    i1 = ipv4.Assign(d1)

    ipv4.SetBase(ns.Ipv4Address("192.168.2.0"), ns.Ipv4Mask("255.255.255.0"))
    i2 = ipv4.Assign(d2)

    # Enable IPv4 Forwarding on the Router (r)
    r = all_nodes.Get(1)  # Router node
    r_ipv4 = r.GetObject(ns.Ipv4)  # Get router's Ipv4 object
    r_ipv4.SetForwarding(1, True)  # Enable packet forwarding on interface 1
    r_ipv4.SetForwarding(0, True)  # Enable forwarding on interface 0

    # Create a Ping application (ICMP Echo Request) from n0 to n1 via r
    print("Application")
    packetSize = 1024
    maxPacketCount = 5
    interPacketInterval = ns.Seconds(1.0)

    ping = ns.V4PingHelper(i2.GetAddress(1))  # Destination = n1's IP

    ping.SetAttribute("Count", ns.UintegerValue(maxPacketCount))
    ping.SetAttribute("Interval", ns.TimeValue(interPacketInterval))
    ping.SetAttribute("Size", ns.UintegerValue(packetSize))

    apps = ping.Install(ns.NodeContainer(net1.Get(0)))  # Install on n0
    apps.Start(ns.Seconds(2.0))
    apps.Stop(ns.Seconds(20.0))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("simple-routing-ping4.tr"))
    csma.EnablePcapAll("simple-routing-ping4", True)

    ns.Simulator.Stop(ns.Seconds(25.0))
    # Run Simulation
    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    import sys
    main(sys.argv)
