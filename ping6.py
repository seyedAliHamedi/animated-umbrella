#
# Copyright (c) 2008-2009 Strasbourg University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: David Gross <gdavid.devel@gmail.com>
#         Sebastien Vincent <vincent@clarinet.u-strasbg.fr>
#

#
# Network topology:
#
#             n0   r    n1
#             |    _    |
#             ====|_|====
#                router
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

    all = ns.NodeContainer(3)
    net1 = ns.NodeContainer()
    net1.Add(all.Get(0))
    net1.Add(all.Get(1))
    net2 = ns.NodeContainer()
    net2.Add(all.Get(1))
    net2.Add(all.Get(2))

    # Create IPv6 Internet Stack
    internetv6 = ns.InternetStackHelper()
    internetv6.Install(all)

    # Create channels
    csma = ns.CsmaHelper()
    csma.SetChannelAttribute(
        "DataRate", ns.DataRateValue(ns.DataRate(5000000)))
    csma.SetChannelAttribute("Delay", ns.TimeValue(ns.MilliSeconds(2)))
    d1 = csma.Install(net1)
    d2 = csma.Install(net2)
    # p2p = ns.PointToPointHelper()
    # p2p.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
    # p2p.SetChannelAttribute("Delay", ns.StringValue("2ms"))

    # d1 = p2p.Install(net1)  # Link between n0 and router
    # d2 = p2p.Install(net2)  # Link between router and n1
    # Create networks and assign IPv6 Addresses
    print("Addressing")
    ipv6 = ns.Ipv6AddressHelper()
    ipv6.SetBase(ns.Ipv6Address("2001:1::"), ns.Ipv6Prefix(64))
    i1 = ipv6.Assign(d1)
    i1.SetForwarding(1, True)
    i1.SetDefaultRouteInAllNodes(1)
    ipv6.SetBase(ns.Ipv6Address("2001:2::"), ns.Ipv6Prefix(64))
    i2 = ipv6.Assign(d2)
    i2.SetForwarding(0, True)
    i2.SetDefaultRouteInAllNodes(0)

    # Create a Ping6 application to send ICMPv6 echo request from n0 to n1 via r
    print("Application")
    packetSize = 1024
    maxPacketCount = 5
    interPacketInterval = ns.Seconds(1.0)
    # ping = ns.PingHelper(i2.GetAddress(1, 1).ConvertTo())
    ping = ns.PingHelper(i2.GetAddress(1, 1).ConvertTo())

    # ping6.SetLocal(i1.GetAddress(0, 1))
    # ping6.SetRemote(i2.GetAddress(1, 1))

    ping.SetAttribute("Count", ns.UintegerValue(maxPacketCount))
    ping.SetAttribute("Interval", ns.TimeValue(interPacketInterval))
    ping.SetAttribute("Size", ns.UintegerValue(packetSize))

    apps = ping.Install(ns.NodeContainer(net1.Get(0)))
    apps.Start(ns.Seconds(2.0))
    apps.Stop(ns.Seconds(20.0))

    print("Tracing")
    ascii = ns.AsciiTraceHelper()
    csma.EnableAsciiAll(ascii.CreateFileStream("simple-routing-ping6.tr"))
    csma.EnablePcapAll("simple-routing-ping6", True)

    # p2p.EnableAsciiAll(ascii.CreateFileStream("simple-routing-ping6.tr"))
    # p2p.EnablePcapAll("simple-routing-ping6", True)

    flowmonHelper = ns.FlowMonitorHelper()
    monitor = flowmonHelper.InstallAll()
    
    monitor.CheckForLostPackets()
    flowmonHelper.SerializeToXmlFile("r.xml", True, True)

   # 🚀 Enable Animation Interface for XML Output
    animFile = "simple-routing-animation.xml"  # Output XML file for NetAnim
    anim = ns.AnimationInterface(animFile)
    
    # 🚀 Optional: Set node descriptions in NetAnim
    anim.SetConstantPosition(all.Get(0), 0.0, 1.0)  # n0
    anim.SetConstantPosition(all.Get(1), 5.0, 2.0)  # router
    anim.SetConstantPosition(all.Get(2), 10.0, 1.0) 

    ns.Simulator.Stop(ns.Seconds(25.0))
    # Run Simulation
    ns.Simulator.Run()
    ns.Simulator.Destroy()


if __name__ == "__main__":
    import sys

    main(sys.argv)
