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


def create_xml( all_nodes, positions,animFile = "./animated-umbrella/rip_udp.xml"):
    """
    Set up node positions and create NetAnim visualization file for an NS-3 simulation.
    
    Args:
        all_nodes (ns3.NodeContainer): Container of nodes to be positioned and animated
        positions (list): List of [x, y] coordinates for each node
        animFile (str, optional): Path where the animation XML file will be saved
                                 Default: "./animated-umbrella/rip_udp.xml"
    
    Returns:
        tuple: (ns3.NodeContainer, ns3.AnimationInterface) - The node container and 
               animation interface object that must be kept alive during the simulation
    
    Note:
        The returned animation interface object must be stored in a variable to prevent
        garbage collection, or segmentation faults may occur during packet tracing.
    """

    # ✅ Step 1: Install mobility and explicitly set node positions
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(all_nodes)


    positions_vec =[]
    for i in range(len(positions)):
        positions_vec.append(ns.Vector(positions[i][0], positions[i][1], 0))

    

    for i in range(all_nodes.GetN()):
        mobility_model = all_nodes.Get(i).GetObject[ns.MobilityModel]()
        if mobility_model:
            mobility_model.SetPosition(positions_vec[i])
        else:
            print(f"🚨 Node {i} does not have a mobility model!")

    # ✅ Step 2: Initialize AnimationInterface after mobility setup
    
    anim = ns.AnimationInterface(animFile)

    # ✅ Step 3: Set NetAnim positions using correct values

    # router_img_id = anim.AddResource("./images/router.png")
    # pc_img_id = anim.AddResource("./images/PC.png") 

    for i in range(all_nodes.GetN()):  
        node = all_nodes.Get(i)   
        anim.SetConstantPosition(node , positions[i][0], positions[i][1],0)  # n0
        # if i < 3:  # Assuming first 3 nodes are routers
        #     anim.UpdateNodeImage(node.GetId(), router_img_id)
        # else:
        #     anim.UpdateNodeImage(node.GetId(), pc_img_id) 
    
    # ✅ Step 4: Print final node positions to verify
    print("\n✅ Final Node Positions:")
    for i in range(all_nodes.GetN()):
        pos = all_nodes.Get(i).GetObject[ns.MobilityModel]().GetPosition()
        print(f"Node {i}: x={pos.x}, y={pos.y}")


    for i in range(all_nodes.GetN()):
        node = all_nodes.Get(i)
        ip_list = get_all_ipv6_addresses(node)
        ip_text = "&#10;".join(ip_list)  # Format multiple IPs
        anim.SetConstantPosition(node, positions_vec[i].x, positions_vec[i].y,0)
        anim.UpdateNodeDescription(node, f"Node {i} &#10; {ip_text}")  # Show all interfaces

    return all_nodes , anim



    