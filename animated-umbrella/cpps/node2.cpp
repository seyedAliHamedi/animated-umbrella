#include "ns3/callback.h"
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

using namespace ns3;

// Node-specific identifiers to avoid conflicts
static uint32_t node2_src = 2;
struct PacketInfo2 {
    uint64_t uid;
    std::string type;
    uint16_t destPort;
    double time;
    uint32_t size;
    uint16_t offset;
};

std::vector<PacketInfo2> node2_transmittedPackets;
std::vector<PacketInfo2> node2_receivedPackets;
std::ofstream node2_packetLogFile("animated-umbrella/cpps/packets_log.txt", std::ios::out | std::ios::app);

// Callback for received packets
void node2_RxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    TcpHeader tcpHeader;

    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    uint16_t identification = ipHeader.GetIdentification();
    uint16_t offset         = ipHeader.GetFragmentOffset(); // in 8-byte blocks
    bool moreFragments      = ipHeader.IsDontFragment();
    
    // Try to see if this is an IPv4 packet
    if (copy->PeekHeader(ipHeader)) {
        // Remove the IP header so the next bytes align with L4


        if (copy->RemoveHeader(ipHeader)) {
            uint8_t protocol = ipHeader.GetProtocol(); // 6 = TCP, 17 = UDP, etc.
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

    // Store packet info
    PacketInfo2 pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset };
    node2_receivedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node2_packetLogFile.is_open()) {
        node2_packetLogFile << "[Node " << node2_src << "] Packet: " << packet->GetUid()
                              << ", RX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << std::endl;
    }

        std::cout << "Received Packet: " << packet
                  << ", Type: " << packetType
                  << ", Dest Port: " << destPort
                  << ", Time: " << time
                  << ", Size: " << packet->GetSize()
                  << ", IP-ID=" << identification
                  << ", FragOffset=" << offset
                  << ", MoreFrag=" << (moreFragments ? 1 : 0)
                  << std::endl;
}

// Callback for transmitted packets
void node2_TxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {
    Ipv4Header ipHeader;
    UdpHeader udpHeader;
    TcpHeader tcpHeader;

    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    uint16_t identification = ipHeader.GetIdentification();
    uint16_t offset         = ipHeader.GetFragmentOffset(); // in 8-byte blocks
    bool moreFragments      = ipHeader.IsDontFragment();

    // Same IPv4-first approach
    if (copy->PeekHeader(ipHeader)) {
        if (copy->RemoveHeader(ipHeader)) {
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

    // Store packet info
    PacketInfo2 pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset };
    node2_transmittedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node2_packetLogFile.is_open()) {
        node2_packetLogFile << "[Node " << node2_src << "] Packet: "
                              << packet->GetUid()
                              << ", TX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << std::endl;
    }

     std::cout << "Transmitted Packet: " << packet
               << ", Type: " << packetType
               << ", Dest Port: " << destPort
               << ", Time: " << time
               << ", Size: " << packet->GetSize()
                << ", IP-ID=" << identification
                << ", FragOffset=" << offset
                << ", MoreFrag=" << (moreFragments ? 1 : 0)
                << std::endl;
}

// Ensure the file closes properly at the end of the simulation
void node2_ClosePacketLog() {
    if (node2_packetLogFile.is_open()) {
        node2_packetLogFile.close();
    }
}

// Node-specific exported functions without C linkage
Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node2_CreateRxCallback() {
    return MakeCallback(&node2_RxCallback);
}

Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node2_CreateTxCallback() {
    return MakeCallback(&node2_TxCallback);
}

// Data access functions
std::vector<PacketInfo2> node2_GetTransmittedPackets() {
    return node2_transmittedPackets;
}

std::vector<PacketInfo2> node2_GetReceivedPackets() {
    return node2_receivedPackets;
}

