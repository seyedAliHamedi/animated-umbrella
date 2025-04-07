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
#include <sstream> 

using namespace ns3;

// Node-specific identifiers to avoid conflicts
static uint32_t node0_src = 0;
struct PacketInfo0 {
    uint64_t uid;
    std::string type;
    uint16_t destPort;
    double time;
    uint32_t size;
    uint16_t offset;
};

std::vector<PacketInfo0> node0_transmittedPackets;
std::vector<PacketInfo0> node0_receivedPackets;
std::ofstream node0_packetLogFile("./animated-umbrella/src/monitor/logs/packets_log.txt", std::ios::out | std::ios::app);

// Callback for received packets
void node0_RxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {
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
    if (copy->PeekHeader(ipHeader)) {
        // Remove the IP header so the next bytes align with L4


        if (copy->RemoveHeader(ipHeader)) {

            Ipv4Address srcAddr = ipHeader.GetSource();
            std::ostringstream ossSrc;
            srcAddr.Print(ossSrc);
            srcIP = ossSrc.str();

            Ipv4Address destAddr = ipHeader.GetDestination();
            std::ostringstream ossDst;
            destAddr.Print(ossDst);
            destIP = ossDst.str();

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
    PacketInfo0 pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset };
    node0_receivedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node0_packetLogFile.is_open()) {
        node0_packetLogFile << "[Node " << node0_src << "] Packet: " << packet->GetUid()
                              << ", RX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << ", src IP: " << srcIP
                              << ", dest IP: " << destIP
                              << std::endl;
    }

      //  std::cout << "Received Packet: " << packet
      //            << ", Type: " << packetType
      //            << ", Dest Port: " << destPort
      //            << ", Time: " << time
      //            << ", Size: " << packet->GetSize()
      //            << ", IP-ID=" << identification
      //            << ", FragOffset=" << offset
      //            << ", MoreFrag=" << (moreFragments ? 1 : 0)
      //            << std::endl;
}

// Callback for transmitted packets
void node0_TxCallback(Ptr<const Packet> packet, Ptr<Ipv4> ipv4,uint32_t interfaceIndex) {
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
    PacketInfo0 pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize(),offset };
    node0_transmittedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (node0_packetLogFile.is_open()) {
        node0_packetLogFile << "[Node " << node0_src << "] Packet: "
                              << packet->GetUid()
                              << ", TX: " << packetType
                              << ", Port: " << destPort
                              << ", Time: " << time
                              << ", Size: " << packet->GetSize()
                              << ", Offset=" << offset
                              << ", src IP: " << srcIP
                              << ", dest IP: " << destIP
                              << std::endl;
    }

   //  std::cout << "Transmitted Packet: " << packet
          //     << ", Type: " << packetType
             //  << ", Dest Port: " << destPort
//<< ", Time: " << time
               //<< ", Size: " << packet->GetSize()
              //  << ", IP-ID=" << identification
            //    << ", FragOffset=" << offset
            //    << ", MoreFrag=" << (moreFragments ? 1 : 0)
           //     << std::endl;
}

// Ensure the file closes properly at the end of the simulation
void node0_ClosePacketLog() {
    if (node0_packetLogFile.is_open()) {
        node0_packetLogFile.close();
    }
}

// Node-specific exported functions without C linkage
Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node0_CreateRxCallback() {
    return MakeCallback(&node0_RxCallback);
}

Callback<void, Ptr<const Packet>, Ptr<Ipv4>, uint32_t> node0_CreateTxCallback() {
    return MakeCallback(&node0_TxCallback);
}

// Data access functions
std::vector<PacketInfo0> node0_GetTransmittedPackets() {
    return node0_transmittedPackets;
}

std::vector<PacketInfo0> node0_GetReceivedPackets() {
    return node0_receivedPackets;
}

