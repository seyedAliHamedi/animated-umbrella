from ns import ns
import re
import os
import csv
import cppyy
import random
import json


sample_data = {
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
    "q_list": {
        "voice": {
            "max_packets": [2000, 3000],
            "packet_size": [160, 200],
            "data_rate": [0.06, 0.08],
            "score_th": 0.80,
            "w_l": 0.20,
            "w_j": 0.40,
            "w_d": 0.40,
            "w_b": 0.00,
            "p": 1.0
        },
        "live_video": {
            "max_packets": [5000, 10000],
            "packet_size": [1200, 1500],
            "data_rate": [1.0, 2.6],
            "score_th": 0.75,
            "w_l": 0.40,
            "w_j": 0.20,
            "w_d": 0.30,
            "w_b": 0.10,
            "p": 0.8
        },
        "gaming": {
            "max_packets": [1000, 2000],
            "packet_size": [50, 200],
            "data_rate": [0.05, 0.20],
            "score_th": 0.85,
            "w_l": 0.25,
            "w_j": 0.25,
            "w_d": 0.50,
            "w_b": 0.00,
            "p": 1.0
        },
        "bulk_data": {
            "max_packets": [20000, 50000],
            "packet_size": [500, 1500],
            "data_rate": [5.0, 20.0],
            "score_th": 0.70,
            "w_l": 0.20,
            "w_j": 0.15,
            "w_d": 0.15,
            "w_b": 0.50,
            "p": 0.5
        }
    },
    "routers": {
        0: {'name': 'Cisco ASR 9904', 'P_idle': 4.0, 'P_rx': 35.0, 'P_tx': 35.0, 'P_base': 1150.0},
        1: {'name': 'Cisco ASR 1001-X', 'P_idle': 0.75, 'P_rx': 7.0, 'P_tx': 5.0, 'P_base': 175.0},
        2: {'name': 'Cisco NCS 5508', 'P_idle': 5.0, 'P_rx': 27.5, 'P_tx': 27.5, 'P_base': 650.0},
        3: {'name': 'Cisco Nexus 9300', 'P_idle': 3.0, 'P_rx': 17.5, 'P_tx': 17.5, 'P_base': 200.0},
        4: {'name': 'Cisco Catalyst 9500', 'P_idle': 1.5, 'P_rx': 11.5, 'P_tx': 11.5, 'P_base': 110.0},
        5: {'name': 'Juniper MX960', 'P_idle': 5.0, 'P_rx': 17.5, 'P_tx': 17.5, 'P_base': 1350.0},
        6: {'name': 'Juniper MX204', 'P_idle': 0.75, 'P_rx': 17.5, 'P_tx': 17.5, 'P_base': 350.0},
        7: {'name': 'Juniper PTX10008', 'P_idle': 8.5, 'P_rx': 35.0, 'P_tx': 35.0, 'P_base': 4000.0},
        8: {'name': 'Juniper PTX1000', 'P_idle': 0.75, 'P_rx': 20.0, 'P_tx': 20.0, 'P_base': 225.0},
        9: {'name': 'Juniper QFX5200', 'P_idle': 3.5, 'P_rx': 20.0, 'P_tx': 20.0, 'P_base': 105.0},
        10: {'name': 'Arista 7500R3', 'P_idle': 7.5, 'P_rx': 42.5, 'P_tx': 42.5, 'P_base': 1500.0},
        11: {'name': 'Nokia 7750 SR-12e', 'P_idle': 4.0, 'P_rx': 15.0, 'P_tx': 15.0, 'P_base': 1250.0},
        12: {'name': 'Nokia 7750 SR-1', 'P_idle': 1.5, 'P_rx': 3.0, 'P_tx': 3.0, 'P_base': 225.0},
        13: {'name': 'Nokia 7950 XRS-20', 'P_idle': 7.5, 'P_rx': 40.0, 'P_tx': 40.0, 'P_base': 6500.0},
        14: {'name': 'Nokia 7250 IXR-6/10', 'P_idle': 3.0, 'P_rx': 32.5, 'P_tx': 32.5, 'P_base': 600.0},
        15: {'name': 'Extreme SLX 9850', 'P_idle': 4.0, 'P_rx': 12.5, 'P_tx': 12.5, 'P_base': 650.0},
        16: {'name': 'Extreme MLXe-16', 'P_idle': 4.0, 'P_rx': 18.0, 'P_tx': 18.0, 'P_base': 900.0},
        17: {'name': 'Huawei NE40E-X8A', 'P_idle': 6.5, 'P_rx': 32.5, 'P_tx': 32.5, 'P_base': 4100.0}
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


def get_ip_to_node(ip_list):
    ip_to_node = {ip: node_id for node_id, ips in ip_list.items()
                  for ip in ips}
    return ip_to_node


def find_path(start_node, dest_ip, routing_tables, ip_to_node):
    dest_parts = dest_ip.split('.')
    dest_net = f"{dest_parts[0]}.{dest_parts[1]}.{dest_parts[2]}.0"

    path = [start_node]
    current_node = start_node
    visited = {current_node}
    max_hops = 30

    while max_hops > 0:
        if current_node not in routing_tables:
            return None

        if dest_net not in routing_tables[current_node]:
            return None

        next_hop = routing_tables[current_node][dest_net]

        if next_hop == '0.0.0.0' or next_hop == dest_ip:
            dest_node = ip_to_node.get(dest_ip)
            if dest_node:
                path.append(int(dest_node))
            return path

        next_node = ip_to_node.get(next_hop)
        if next_node is None:
            return None

        next_node = int(next_node)

        if next_node in visited:
            return None

        path.append(next_node)
        visited.add(next_node)
        current_node = next_node

        node_ips = [ip for ip, node in ip_to_node.items() if int(node)
                    == next_node]
        for ip in node_ips:
            if ip.startswith('.'.join(dest_parts[:3])):
                dest_node = ip_to_node.get(dest_ip)
                if dest_node:
                    path.append(int(dest_node))
                return path

        max_hops -= 1

    return None


def parse_routes_manually(file_path):
    routing_tables = {}

    with open(file_path, 'r') as f:
        xml_content = f.read()

    rt_blocks = xml_content.split('<rt ')

    for block in rt_blocks[1:]:
        match = re.match(
            r't="(\d+)" id="(\d+)" info="(.*?)" />', block, re.DOTALL)
        if match:
            time, node_id, info = match.groups()
            node_id = int(node_id)
            time = int(time)

            if time == 10:
                if node_id not in routing_tables:
                    routing_tables[node_id] = {}

                for line in info.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('Node:') and not line.startswith('Destination'):
                        parts = line.split()
                        if len(parts) >= 3:
                            dest_net = parts[0]
                            gateway = parts[1]

                            if gateway != '0.0.0.0' and gateway != 'Genmask':
                                routing_tables[node_id][dest_net] = gateway

    return routing_tables


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
    
    // Optimized packet info structure
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
    
    // Reserve space for packets to avoid frequent allocations
    std::vector<PacketInfo> allPackets;
    
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
                // Efficiently convert IPs to strings
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
        
        // Store packet info efficiently
        allPackets.push_back({
            nodeId, 
            packet->GetUid(), 
            packetType, 
            destPort, 
            Simulator::Now().GetSeconds(), 
            packet->GetSize(),
            offset,
            srcIP,
            destIP,
            isRx ? "RX" : "TX"
        });
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
    
    // Direct access methods for efficiency
    size_t GetPacketCount() { return allPackets.size(); }
    uint32_t GetPacketNodeId(size_t i) { return allPackets[i].nodeId; }
    uint64_t GetPacketUid(size_t i) { return allPackets[i].uid; }
    std::string GetPacketType(size_t i) { return allPackets[i].type; }
    uint16_t GetPacketPort(size_t i) { return allPackets[i].destPort; }
    double GetPacketTime(size_t i) { return allPackets[i].time; }
    uint32_t GetPacketSize(size_t i) { return allPackets[i].size; }
    uint16_t GetPacketOffset(size_t i) { return allPackets[i].offset; }
    std::string GetPacketSrcIp(size_t i) { return allPackets[i].srcIP; }
    std::string GetPacketDestIp(size_t i) { return allPackets[i].destIP; }
    std::string GetPacketDirection(size_t i) { return allPackets[i].direction; }
    
    // Clear and reserve space
    void ClearPacketData() {
        allPackets.clear();
        allPackets.reserve(100000); // Pre-allocate space
    }
    
    // Create callbacks
    Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> CreateRxCallback() {
        return MakeCallback(&RxCallback);
    }
    
    Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> CreateTxCallback() {
        return MakeCallback(&TxCallback);
    }
    '''
