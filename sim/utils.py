from ns import ns

import re
import os
import csv
import cppyy
import random
import json





sample_data = {
    "topology_adj_matrix" : [
    [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
    ],
    "topology_links_type":['csma', 'p2p'],
    "topology_links_rate":['5Mbps', '10Mbps', '1Mbps'],
    "topology_links_delay" :['5ms', '10ms', '10ms',],
    "topology_links_queue" :['5000','10000'],
    "topology_links_errors" : [0,0.1,0],
    "topology_base_network" : "192.166.1.0/24",
    "topology_xml_file" : "./sim/monitor/xml/topology.xml",
    
    
    "app_n_servers": 4,
    "app_n_clients": 5,
    "app_links_type": ['csma', 'p2p'],
    "app_links_rate": ['5Mbps', '10Mbps', '1Mbps'],
    "app_links_delay": ['5ms', '10ms', '10ms',],
    "app_type": "udp_echo",
    "app_max_packets": 10000,
    "app_interval": 1,
    "app_packet_size": 1024,
    "app_port": 9,
    "tcp_app_data_rate":500000,
    
    "app_animation_file": "./sim/monitor/xml/app.xml",
    
    "app_duration": 100,
    "app_start_time":10,
}

def distribute_values(values, count):
    return [random.choice(values) for _ in range(count)]

def calculate_subnet_mask(mask_bits):
    subnet_mask = [0, 0, 0, 0]
    for j in range(4):
        if mask_bits > 8:
            subnet_mask[j] = 255
            mask_bits -= 8
        else:
            subnet_mask[j] = 256 - 2 ** (8 - mask_bits)
            mask_bits = 0
    return ".".join(map(str, subnet_mask))


def fix_xml(animFile="./animated-umbrella/rip_udp.xml"):
    print("\n ------------------------- fixing XML --------------------------")
    with open(animFile,"r") as file:
        data = file.read()
    data = data.replace("&amp;#10","&#10")
    with open(animFile,"w") as file:
        file.write(data)
    print("XML Fixed")

def create_xml(all_nodes, positions, animFile):
    print("\n ------------------------- creating XML --------------------------")
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(all_nodes)

    positions_vec = [ns.Vector(pos[0], pos[1], 0) for pos in positions]

    for i in range(all_nodes.GetN()):
        mobility_model = all_nodes.Get(i).GetObject[ns.MobilityModel]()
        if mobility_model:
            mobility_model.SetPosition(positions_vec[i])

    anim = ns.AnimationInterface(animFile)

    for i in range(all_nodes.GetN()):  
        node = all_nodes.Get(i)   
        anim.SetConstantPosition(node, positions[i][0], positions[i][1],0)

    print("XML Created")
    return all_nodes, anim



    
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




def generate_node_files(num_nodes, output_dir="./sim/monitor/cpps"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate node files with unique function names
    for i in range(num_nodes):
        out_name = f"node{i}.cpp"
        out_path = os.path.join(output_dir, out_name)
        
        # Generate code with node-specific function names
        code = f'''#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include "ns3/simulator.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/tag.h"
#include "ns3/ipv4-header.h"
#include <vector>
#include <fstream>
#include <string>
#include <sstream> 

using namespace ns3;

// Node-specific identifiers to avoid conflicts
static uint32_t node{i}_src = {i};
struct PacketInfo{i} {{
    uint64_t uid;
    std::string type;
    uint16_t destPort;
    double time;
    uint32_t size;
    uint16_t offset;
}};

std::vector<PacketInfo{i}> node{i}_transmittedPackets;
std::vector<PacketInfo{i}> node{i}_receivedPackets;
std::ofstream node{i}_packetLogFile("./sim/monitor/logs/packets_log.txt", std::ios::out | std::ios::app);

// Callback for received packets
void node{i}_RxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {{
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    TcpHeader tcpHeader;
    std::string destIP;
    std::string srcIP;

    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    uint16_t identification = ipHeader.GetIdentification();
    uint16_t offset         = ipHeader.GetFragmentOffset(); // in 8-byte blocks
    bool moreFragments      = ipHeader.IsDontFragment();
    
    // Try to see if this is an IPv4 packet
    if (copy->PeekHeader(ipHeader)) {{
        // Remove the IP header so the next bytes align with L4


        if (copy->RemoveHeader(ipHeader)) {{

            Ipv4Address srcAddr = ipHeader.GetSource();
            std::ostringstream ossSrc;
            srcAddr.Print(ossSrc);
            srcIP = ossSrc.str();

            Ipv4Address destAddr = ipHeader.GetDestination();
            std::ostringstream ossDst;
            destAddr.Print(ossDst);
            destIP = ossDst.str();

            uint8_t protocol = ipHeader.GetProtocol(); // 6 = TCP, 17 = UDP, etc.
            if (protocol == 6) {{ // TCP
                if (copy->PeekHeader(tcpHeader)) {{
                    destPort = tcpHeader.GetDestinationPort();
                    packetType = "TCP";
                }}
            }}
            else if (protocol == 17) {{ // UDP
                if (copy->PeekHeader(udpHeader)) {{
                    destPort = udpHeader.GetDestinationPort();
                    packetType = "UDP";
                }}
            }}

        }}
    }}

    double time = Simulator::Now().GetSeconds();

    // Store packet info
    PacketInfo{i} pktInfo = {{ packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset }};
    node{i}_receivedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node{i}_packetLogFile.is_open()) {{
        node{i}_packetLogFile << "[Node " << node{i}_src << "] Packet: " << packet->GetUid()
                              << ", RX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << ", src IP: " << srcIP
                              << ", dest IP: " << destIP
                              << std::endl;
    }}

      //  std::cout << "Received Packet: " << packet
      //            << ", Type: " << packetType
      //            << ", Dest Port: " << destPort
      //            << ", Time: " << time
      //            << ", Size: " << packet->GetSize()
      //            << ", IP-ID=" << identification
      //            << ", FragOffset=" << offset
      //            << ", MoreFrag=" << (moreFragments ? 1 : 0)
      //            << std::endl;
}}

// Callback for transmitted packets
void node{i}_TxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {{
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    TcpHeader tcpHeader;
    std::string destIP;
    std::string srcIP;

    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    uint16_t identification = ipHeader.GetIdentification();
    uint16_t offset         = ipHeader.GetFragmentOffset(); // in 8-byte blocks
    bool moreFragments      = ipHeader.IsDontFragment();

    // Same IPv4-first approach
    if (copy->PeekHeader(ipHeader)) {{
        if (copy->RemoveHeader(ipHeader)) {{
        
            Ipv4Address srcAddr = ipHeader.GetSource();
            std::ostringstream ossSrc;
            srcAddr.Print(ossSrc);
            srcIP = ossSrc.str();

            Ipv4Address destAddr = ipHeader.GetDestination();
            std::ostringstream ossDst;
            destAddr.Print(ossDst);
            destIP = ossDst.str();
            
            uint8_t protocol = ipHeader.GetProtocol();

            if (protocol == 6) {{ // TCP
                if (copy->PeekHeader(tcpHeader)) {{
                    destPort = tcpHeader.GetDestinationPort();
                    packetType = "TCP";
                }}
            }}
            else if (protocol == 17) {{ // UDP
                if (copy->PeekHeader(udpHeader)) {{
                    destPort = udpHeader.GetDestinationPort();
                    packetType = "UDP";
                }}
            }}
        }}
    }}

    double time = Simulator::Now().GetSeconds();

    // Store packet info
    PacketInfo{i} pktInfo = {{ packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset }};
    node{i}_transmittedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node{i}_packetLogFile.is_open()) {{
        node{i}_packetLogFile << "[Node " << node{i}_src << "] Packet: "
                              << packet->GetUid()
                              << ", TX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << ", src IP: " << srcIP
                              << ", dest IP: " << destIP
                              << std::endl;
    }}

   //  std::cout << "Transmitted Packet: " << packet
          //     << ", Type: " << packetType
             //  << ", Dest Port: " << destPort
//<< ", Time: " << time
               //<< ", Size: " << packet->GetSize()
              //  << ", IP-ID=" << identification
            //    << ", FragOffset=" << offset
            //    << ", MoreFrag=" << (moreFragments ? 1 : 0)
           //     << std::endl;
}}

// Ensure the file closes properly at the end of the simulation
void node{i}_ClosePacketLog() {{
    if (node{i}_packetLogFile.is_open()) {{
        node{i}_packetLogFile.close();
    }}
}}

// Node-specific exported functions without C linkage
Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node{i}_CreateRxCallback() {{
    return MakeCallback(&node{i}_RxCallback);
}}

Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node{i}_CreateTxCallback() {{
    return MakeCallback(&node{i}_TxCallback);
}}

// Data access functions
std::vector<PacketInfo{i}> node{i}_GetTransmittedPackets() {{
    return node{i}_transmittedPackets;
}}

std::vector<PacketInfo{i}> node{i}_GetReceivedPackets() {{
    return node{i}_receivedPackets;
}}

'''
        
        # Write the code to the file
        with open(out_path, 'w') as f:
            f.write(code)
            
    print(f"Generated {num_nodes} node files in {output_dir}")



def run_cpp_file(cpp_file_path):
    """
    Read a C++ file and execute it using cppyy
    
    Args:
        cpp_file_path: Path to the C++ file
        
    Returns:
        Any results from the execution
    """
    # Check if file exists
    if not os.path.exists(cpp_file_path):
        raise FileNotFoundError(f"File not found: {cpp_file_path}")
    
    # Read the C++ file
    with open(cpp_file_path, 'r') as file:
        cpp_code = file.read()
    
    # Load the C++ code
    cppyy.cppdef(cpp_code)
    
    # Return the loaded namespace
    return cppyy.gbl



def setup_packet_tracing_for_router(router, trace_modules):
    node_id = router.GetId()
    print(f"Setting up IP-layer packet tracing for router {node_id}")

    # Get the Ipv4 object from this node
    ipv4 = router.GetObject[ns.Ipv4]()
    if ipv4 is None:
        print(f"No Ipv4 object found on node {node_id}, skipping IP-layer hookup.")
        return

    # Assuming your module is at index 0 in trace_modules,
    # or adapt if you have per-interface modules.
    module = trace_modules[node_id]

    # Build the C++ callbacks for this node
    rx_callback_func = getattr(module, f"node{node_id}_CreateRxCallback")
    tx_callback_func = getattr(module, f"node{node_id}_CreateTxCallback")

    rx_callback = rx_callback_func()
    tx_callback = tx_callback_func()

    # Connect the IP-layer "Rx" and "Tx" trace sources:
    ipv4.TraceConnectWithoutContext("Rx", rx_callback)
    ipv4.TraceConnectWithoutContext("Tx", tx_callback)


def create_csv(input_file):
    # Regular expression to parse each line
    log_pattern = re.compile(
        r"\[Node (\d+)\] Packet: (\S+),\s*(TX|RX):\s*(UDP|TCP),\s*Port:\s*(\d+),"
        r"\s*Time:\s*([0-9.]+),\s*Size:\s*(\d+),\s*Offset=\s*(\d+),"
        r"\s*src IP:\s*([\d\.]+),\s*dest IP:\s*([\d\.]+)"
    )
    # Define output file name based on input file
    output_file = os.path.splitext(input_file)[0] + ".csv"

    # Open input file and parse contents
    data = []
    with open(input_file, "r+") as file:
        for line in file:
            match = log_pattern.match(line.strip())
            if match:
                node, packet , direction, protocol, port, time, size, offset, src_ip, dest_ip = match.groups()
                data.append([node,packet, direction, protocol, port, time, size, offset, src_ip, dest_ip])


    # Write to CSV
    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Node","Packet", "Direction", "Protocol", "Port", "Time", "Size" , "Offset","src IP","dest IP"])
        writer.writerows(data)

    print(f"CSV file '{output_file}' has been created successfully.")



def get_ip_to_node(ip_list):
    ip_to_node = {ip: node_id for node_id, ips in ip_list.items() for ip in ips}
    return ip_to_node


def parse_traceroute_blocks(log_file):
    blocks = []
    current_block = {"dest": None, "hops": []}
    traceroute_header_pattern = re.compile(r"Traceroute to ([\d\.]+)")

    with open(log_file, "r") as file:
        for line in file:
            line = line.strip()

            if line.startswith("Traceroute to"):
                if current_block["dest"]:
                    blocks.append(current_block)
                match = traceroute_header_pattern.match(line)
                current_block = {"dest": match.group(1), "hops": []}
                continue

            if line.startswith("Trace Complete"):
                if current_block["dest"]:
                    blocks.append(current_block)
                    current_block = {"dest": None, "hops": []}
                continue

            hop_match = re.match(r"\d+\s+([\d\.]+)", line)
            if hop_match:
                ip = hop_match.group(1)
                if ip != "*" and ip not in current_block["hops"]:
                    current_block["hops"].append(ip)

    return blocks


def convert_all_to_node_paths(blocks, ip_mapping_path, src_ip_list):
    with open(ip_mapping_path, "r") as f:
        ip_to_node = json.load(f)

    if len(src_ip_list) != len(blocks):
        raise ValueError("Length of src_ip_list must match number of traceroute blocks.")

    route_dict = {}

    for i, block in enumerate(blocks):
        src_ip = src_ip_list[i]
        dest_ip = block["dest"]
        hop_ips = block["hops"]

        # Move destination IP to the end
        if dest_ip in hop_ips:
            hop_ips.remove(dest_ip)
        hop_ips.append(dest_ip)

        src_node = ip_to_node.get(src_ip)
        if src_node is None:
            print(f"⚠️ Source IP {src_ip} not found in mapping, skipping.")
            continue

        node_path = [src_node]
        for ip in hop_ips:
            node = ip_to_node.get(ip)
            if node is not None and node != node_path[-1]:
                node_path.append(node)

        route_dict[(src_ip, dest_ip)] = node_path

    return route_dict


def get_routes(src_ips,ip_map_path='animated-umbrella/src/monitor/logs/ip_mapping.json',log_path='animated-umbrella/mytrace.log'):        
    blocks = parse_traceroute_blocks(log_path)
    routes = convert_all_to_node_paths(blocks, ip_map_path, src_ips)
    return routes

