from ns import ns

import re
import os
import csv
import cppyy
import random
import json


sample_data = {
    "cpp_code_f": "",
    "topology_adj_matrix": [
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
    "topology_links_type": ['csma', 'p2p'],
    "topology_links_rate": ['5Mbps', '10Mbps', '1Mbps'],
    "topology_links_delay": ['5ms', '10ms', '10ms',],
    "topology_links_queue": ['5000', '10000'],
    "topology_links_errors": [0, 0.1, 0],
    "topology_base_network": "192.166.1.0/24",
    "topology_xml_file": "./sim/monitor/xml/topology.xml",


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
    "tcp_app_data_rate": 500000,

    "app_animation_file": "./sim/monitor/xml/app.xml",
    "q_list": {"q1": {"max_packets": [200, 400], "packet_size": [100, 1000], "score_th": 0.7},
               "q2": {"max_packets": [200, 400], "packet_size": [100, 1000], "score_th": 0.7},
               "q3": {"max_packets": [200, 400], "packet_size": [100, 1000], "score_th": 0.7},
               "q4": {"max_packets": [200, 400], "packet_size": [100, 1000], "score_th": 0.7}
               },

    "app_duration": 100,
    "app_start_time": 10,



    "xml_animation_file": "./sim/monitor/xml/animation.xml",
    "routing_table_file": "./sim/monitor/xml/routes.xml",
    "pcap_files_prefix": "./sim/monitor/pcap/capture",
    "flow_stats_file": "./sim/monitor/xml/flow-stats.xml",
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
    with open(animFile, "r") as file:
        data = file.read()
    data = data.replace("&amp;#10", "&#10")
    with open(animFile, "w") as file:
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
        anim.SetConstantPosition(node, positions[i][0], positions[i][1], 0)

    print("XML Created")
    return all_nodes, anim


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
      //           << std::endl;
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
    with open(cpp_file_path, 'r') as file:
        cpp_code = file.read()

    cppyy.cppdef(cpp_code)

    return cppyy.gbl


def setup_packet_tracing_for_router(router, trace_modules):
    node_id = router.GetId()

    ipv4 = router.GetObject[ns.Ipv4]()
    module = trace_modules[node_id]

    rx_callback_func = getattr(module, f"node{node_id}_CreateRxCallback")
    tx_callback_func = getattr(module, f"node{node_id}_CreateTxCallback")

    rx_callback = rx_callback_func()
    tx_callback = tx_callback_func()

    ipv4.TraceConnectWithoutContext("Rx", rx_callback)
    ipv4.TraceConnectWithoutContext("Tx", tx_callback)


def create_csv(input_file, routing_paths=None):
    log_pattern = re.compile(
        r"\[Node (\d+)\] Packet: (\S+),\s*(TX|RX):\s*(UDP|TCP),\s*Port:\s*(\d+),"
        r"\s*Time:\s*([0-9.]+),\s*Size:\s*(\d+),\s*Offset=\s*(\d+),"
        r"\s*src IP:\s*([\d\.]+),\s*dest IP:\s*([\d\.]+)"
    )

    output_file = os.path.splitext(input_file)[0] + ".csv"
    data = []

    paths_map = {}
    if routing_paths:
        for path_info in routing_paths:
            src_ip = path_info["src_ip"]
            dest_ip = path_info["dest_ip"]
            path = path_info["path"]
            paths_map[(src_ip, dest_ip)] = path

    with open(input_file, "r+") as file:
        for line in file:
            match = log_pattern.match(line.strip())
            if match:
                node, packet, direction, protocol, port, time, size, offset, src_ip, dest_ip = match.groups()
                node = int(node)

                prev_hop = "Null"
                next_hop = "Null"
                total_hops = 0

                try:
                    path = paths_map.get((src_ip, dest_ip), [])

                    node_index = path.index(node)

                    prev_hop = path[node_index - 1]
                    next_hop = path[node_index + 1]
                    if node_index-1 == 0:
                        if port == "9":
                            prev_hop = "Client"
                        if port == "49153":
                            prev_hop = "Server"

                    if node_index + 1 == len(path) - 1:
                        if port == "9":
                            next_hop = "Server"
                        if port == "49153":
                            next_hop = "Client"

                    total_hops = len(path)-2
                except:
                    pass

                data.append([node, packet, direction, protocol, port, time,
                            size, offset, src_ip, dest_ip, prev_hop, next_hop, total_hops])

    with open(output_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Node", "Packet", "Direction", "Protocol", "Port",
                        "Time", "Size", "Offset", "src IP", "dest IP", "prev_hop", "next_hop", "total_hops"])
        writer.writerows(data)


def get_ip_to_node(ip_list):
    ip_to_node = {ip: node_id for node_id, ips in ip_list.items()
                  for ip in ips}
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


def find_path(start_node, dest_ip, routing_tables, ip_to_node):
    # Extract the network portion of the destination IP
    dest_parts = dest_ip.split('.')
    dest_net = f"{dest_parts[0]}.{dest_parts[1]}.{dest_parts[2]}.0"

    path = [start_node]
    current_node = start_node
    visited = set([current_node])
    max_hops = 30

    while max_hops > 0:
        # Check if we have routing information for the current node
        if current_node not in routing_tables:
            print(f"  No routing table for node {current_node}")
            return None

        # Check if we have a route to the destination network
        if dest_net not in routing_tables[current_node]:
            print(f"  No route from node {current_node} to {dest_net}")
            return None

        # Get the next hop IP
        next_hop = routing_tables[current_node][dest_net]

        # Check if we've reached the destination's network (direct delivery)
        if next_hop == '0.0.0.0' or next_hop == dest_ip:
            dest_node = ip_to_node.get(dest_ip)
            if dest_node:
                path.append(int(dest_node))
            return path

        # Find the node ID corresponding to the next hop IP
        next_node = None
        for ip, node in ip_to_node.items():
            if ip == next_hop:
                next_node = int(node)
                break

        if next_node is None:
            print(f"  Unknown node for IP {next_hop}")
            return None

        if next_node in visited:
            print(f"  Loop detected at {current_node} → {next_node}")
            return None

        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

        # Check if we've reached the destination
        node_ips = [ip for ip, node in ip_to_node.items() if int(node)
                    == next_node]
        for ip in node_ips:
            if ip.startswith(dest_parts[0] + '.' + dest_parts[1] + '.' + dest_parts[2]):
                # We're in the same network as the destination
                dest_node = ip_to_node.get(dest_ip)
                if dest_node:
                    path.append(int(dest_node))
                return path

        max_hops -= 1

    print("  Maximum hop count exceeded")
    return None


def parse_routes_manually(file_path):
    """Parse the routes XML file manually with careful handling of the format."""
    routing_tables = {}
    # Read the file as text
    with open(file_path, 'r') as f:
        xml_content = f.read()

    # Process each <rt> tag manually with a more precise pattern
    import re

    # Split the file by <rt> tags to process each one individually
    rt_blocks = xml_content.split('<rt ')

    for block in rt_blocks[1:]:  # Skip the first split which is the header
        # Extract the attributes and content
        match = re.match(
            r't="(\d+)" id="(\d+)" info="(.*?)" />', block, re.DOTALL)
        if match:
            time, node_id, info = match.groups()
            node_id = int(node_id)
            time = int(time)

            if time == 10:  # Only use the t=5 routing tables
                if node_id not in routing_tables:
                    routing_tables[node_id] = {}

                # Process each line in the routing table
                for line in info.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('Node:') and not line.startswith('Destination'):
                        parts = line.split()
                        if len(parts) >= 3:
                            dest_net = parts[0]
                            gateway = parts[1]

                            if gateway != '0.0.0.0' and gateway != 'Genmask':
                                routing_tables[node_id][dest_net] = gateway

    print(f"Found routing data for {len(routing_tables)} nodes")
    return routing_tables

    # Define a C++ module that stores data in memory and provides access methods
sample_data['cpp_code_f'] = '''
    #include "ns3/callback.h"
    #include "ns3/packet.h"
    #include "ns3/ipv4-header.h"
    #include "ns3/udp-header.h"
    #include "ns3/tcp-header.h"
    #include "ns3/simulator.h"
    #include <vector>
    #include <map>
    #include <string>
    #include <sstream>
    
    using namespace ns3;
    
    // Single packet info structure
    struct PacketInfo {
        uint32_t nodeId;
        uint64_t uid;
        std::string type;
        uint16_t destPort;
        double time;
        uint32_t size;
        uint16_t offset;
        std::string srcIP;
        std::string destIP;
        std::string direction;
    };
    
    // Global vector to store all packets - using a single container is more efficient
    std::vector<PacketInfo> allPackets;
    
    // Process packet and store in memory
    void ProcessPacket(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interfaceIndex, 
                      uint32_t nodeId, bool isRx) {
        Ipv4Header ipHeader;
        UdpHeader udpHeader;
        TcpHeader tcpHeader;
        std::string destIP = "";
        std::string srcIP = "";
        std::string packetType = "Unknown";
        uint16_t destPort = 0;
        
        Ptr<Packet> copy = packet->Copy();
        uint16_t offset = 0;
        
        if (copy->PeekHeader(ipHeader)) {
            if (copy->RemoveHeader(ipHeader)) {
                Ipv4Address srcAddr = ipHeader.GetSource();
                std::ostringstream ossSrc;
                srcAddr.Print(ossSrc);
                srcIP = ossSrc.str();
                
                Ipv4Address destAddr = ipHeader.GetDestination();
                std::ostringstream ossDst;
                destAddr.Print(ossDst);
                destIP = ossDst.str();
                
                offset = ipHeader.GetFragmentOffset();
                uint8_t protocol = ipHeader.GetProtocol();
                
                if (protocol == 6) { // TCP
                    if (copy->PeekHeader(tcpHeader)) {
                        destPort = tcpHeader.GetDestinationPort();
                        packetType = "TCP";
                    }
                }
                else if (protocol == 17) { // UDP
                    if (copy->PeekHeader(udpHeader)) {
                        destPort = udpHeader.GetDestinationPort();
                        packetType = "UDP";
                    }
                }
            }
        }
        
        double time = Simulator::Now().GetSeconds();
        
        // Create packet info
        PacketInfo pktInfo = {
            nodeId, 
            packet->GetUid(), 
            packetType, 
            destPort, 
            time, 
            packet->GetSize(),
            offset,
            srcIP,
            destIP,
            isRx ? "RX" : "TX"
        };
        
        // Store in global container
        allPackets.push_back(pktInfo);
    }
    
    // Callbacks
    void RxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interfaceIndex) {
        uint32_t nodeId = ipv4->GetObject<Node>()->GetId();
        ProcessPacket(packet, ipv4, interfaceIndex, nodeId, true);
    }
    
    void TxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interfaceIndex) {
        uint32_t nodeId = ipv4->GetObject<Node>()->GetId();
        ProcessPacket(packet, ipv4, interfaceIndex, nodeId, false);
    }
    
    // Get packet data - direct access
    size_t GetPacketCount() {
        return allPackets.size();
    }
    
    uint32_t GetPacketNodeId(size_t index) {
        return allPackets[index].nodeId;
    }
    
    uint64_t GetPacketUid(size_t index) {
        return allPackets[index].uid;
    }
    
    std::string GetPacketType(size_t index) {
        return allPackets[index].type;
    }
    
    uint16_t GetPacketPort(size_t index) {
        return allPackets[index].destPort;
    }
    
    double GetPacketTime(size_t index) {
        return allPackets[index].time;
    }
    
    uint32_t GetPacketSize(size_t index) {
        return allPackets[index].size;
    }
    
    uint16_t GetPacketOffset(size_t index) {
        return allPackets[index].offset;
    }
    
    std::string GetPacketSrcIp(size_t index) {
        return allPackets[index].srcIP;
    }
    
    std::string GetPacketDestIp(size_t index) {
        return allPackets[index].destIP;
    }
    
    std::string GetPacketDirection(size_t index) {
        return allPackets[index].direction;
    }
    
    // Clear data when done
    void ClearPacketData() {
        allPackets.clear();
    }
    
    // Create callbacks
    Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> CreateRxCallback() {
        return MakeCallback(&RxCallback);
    }
    
    Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> CreateTxCallback() {
        return MakeCallback(&TxCallback);
    }
    '''
