#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/net-device.h"
#include "ns3/trace-source-accessor.h"
#include <iostream>
#include <map>
#include <Python.h>  // ✅ Correctly include Python API

using namespace ns3;

// ✅ Explicitly declare the Python functions before use
extern "C" {
    void updateTxCounter(int node_id);
    void updateRxCounter(int node_id);
}

// Global packet count maps
std::map<uint32_t, uint32_t> nodeTxCount;
std::map<uint32_t, uint32_t> nodeRxCount;

// ✅ Function to trace packet transmission
void TracePacketSent(Ptr<const Packet> packet, Ptr<NetDevice> device) {
    Ptr<Node> node = device->GetNode();
    uint32_t nodeId = node->GetId();
    nodeTxCount[nodeId]++;

    // ✅ Call Python function safely
    if (Py_IsInitialized()) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        try {
            updateTxCounter(nodeId);
        } catch (...) {
            PyErr_Print();
        }
        PyGILState_Release(gstate);
    }
}

// ✅ Function to trace packet reception
void TracePacketReceived(Ptr<const Packet> packet, Ptr<NetDevice> device) {
    Ptr<Node> node = device->GetNode();
    uint32_t nodeId = node->GetId();
    nodeRxCount[nodeId]++;

    // ✅ Call Python function safely
    if (Py_IsInitialized()) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        try {
            updateRxCounter(nodeId);
        } catch (...) {
            PyErr_Print();
        }
        PyGILState_Release(gstate);
    }
}

// ✅ Function to attach tracing to all nodes
void AttachPacketTracing(Node* node) {
    for (uint32_t i = 0; i < node->GetNDevices(); i++) {
        Ptr<NetDevice> netDevice = node->GetDevice(i);
        if (netDevice) {
            netDevice->TraceConnectWithoutContext("MacTx", MakeCallback(&TracePacketSent));
            netDevice->TraceConnectWithoutContext("MacRx", MakeCallback(&TracePacketReceived));
        }
    }
}

// ✅ Function to print packet statistics at the end of the simulation
extern "C" void PrintTracerouteResults() {
    std::cout << "\\n📊 Traceroute Simulation Results (Per-Node Packet Count):" << std::endl;
    for (const auto& txEntry : nodeTxCount) {
        uint32_t nodeId = txEntry.first;
        uint32_t sent = txEntry.second;
        uint32_t received = nodeRxCount[nodeId]; // Default to 0 if missing
        std::cout << "Node " << nodeId << ":  📤 Sent: " << sent
                  << " packets   📥 Received: " << received << " packets" << std::endl;
    }
}

