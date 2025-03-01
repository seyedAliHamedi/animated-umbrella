from ns import ns


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


def fix_xml(animFile="./animated-umbrella/rip_udp.xml"):
    print("\n ------------------------- fixing XML --------------------------")
    with open(animFile,"r") as file:
        data = file.read()
    data = data.replace("&amp;#10","&#10")
    with open(animFile,"w") as file:
        file.write(data)
    print("XML FIXED")

def create_xml(all_nodes, positions, animFile="./animated-umbrella/rip_udp.xml"):
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(all_nodes)

    positions_vec = [ns.Vector(pos[0], pos[1], 0) for pos in positions]

    for i in range(all_nodes.GetN()):
        mobility_model = all_nodes.Get(i).GetObject[ns.MobilityModel]()
        if mobility_model:
            mobility_model.SetPosition(positions_vec[i])
        else:
            print(f"🚨 Node {i} does not have a mobility model!")

    anim = ns.AnimationInterface(animFile)

    for i in range(all_nodes.GetN()):  
        node = all_nodes.Get(i)   
        anim.SetConstantPosition(node, positions[i][0], positions[i][1],0)

    print("\n✅ Final Node Positions in NetAnim:")
    for i in range(all_nodes.GetN()):
        pos = all_nodes.Get(i).GetObject[ns.MobilityModel]().GetPosition()
        print(f"Node {i}: x={pos.x}, y={pos.y}")

    return all_nodes, anim


    