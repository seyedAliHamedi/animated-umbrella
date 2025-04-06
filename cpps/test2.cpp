#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/udp-header.h"
#include "ns3/tcp-header.h"
#include "ns3/simulator.h"
#include "ns3/net-device.h"
#include "ns3/node.h"
#include "ns3/tag.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace ns3;

static uint32_t src = 0;
struct PacketInfo {
    uint64_t uid;
    std::string type;
    uint16_t destPort;
    double time;
    uint32_t size;
};

std::vector<PacketInfo> transmittedPackets;
std::vector<PacketInfo> receivedPackets;

std::ofstream packetLogFile("animated-umbrella/cpps/packets_log.txt", std::ios::out | std::ios::app);


// Callback for received packets
void RxCallback(Ptr<const Packet> packet) {
    UdpHeader udpHeader;
    TcpHeader tcpHeader;
    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    if (copy->PeekHeader(udpHeader)) {
        destPort = udpHeader.GetDestinationPort();
        packetType = "UDP";
    } else if (copy->PeekHeader(tcpHeader)) {
        destPort = tcpHeader.GetDestinationPort();
        packetType = "TCP";
    }

    double time = Simulator::Now().GetSeconds();

    // Store packet info
    PacketInfo pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize() };
    receivedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (packetLogFile.is_open()) {
        packetLogFile << "[Node " << src << "] Packet: " << packet->GetUid() << ", RX: " << packetType 
                      << ", Port: " << destPort 
                      << ", Time: " << time 
                      << ", Size: " << packet->GetSize() << std::endl;
    }
}

void TxCallback(Ptr<const Packet> packet) {
    UdpHeader udpHeader;
    TcpHeader tcpHeader;
    std::string packetType = "Unknown";
    uint16_t destPort = 0;

    Ptr<Packet> copy = packet->Copy();
    if (copy->PeekHeader(udpHeader)) {
        destPort = udpHeader.GetDestinationPort();
        packetType = "UDP";
    } else if (copy->PeekHeader(tcpHeader)) {
        destPort = tcpHeader.GetDestinationPort();
        packetType = "TCP";
    }


    double time = Simulator::Now().GetSeconds();

    // Store packet info
    PacketInfo pktInfo = { packet->GetUid(), packetType, destPort, time, packet->GetSize() };
    transmittedPackets.push_back(pktInfo);

    // Log the packet to the file
    if (packetLogFile.is_open()) {
        packetLogFile << "[Node " << src << "] Packet: " << packet->GetUid() << ", TX: " << packetType 
                      << ", Port: " << destPort 
                      << ", Time: " << time 
                      << ", Size: " << packet->GetSize() << std::endl;
    }
}


// Ensure the file closes properly at the end of the simulation
void ClosePacketLog() {
    if (packetLogFile.is_open()) {
        packetLogFile.close();
    }
}

// Create Callbacks
static Callback<void, Ptr<const Packet>> CreateRxCallback() {
    return MakeCallback(&RxCallback);
}

static Callback<void, Ptr<const Packet>> CreateTxCallback() {
    return MakeCallback(&TxCallback);
}
