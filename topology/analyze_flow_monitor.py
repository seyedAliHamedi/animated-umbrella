import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# Load the XML file
xml_file = "./flow-monitor-results.xml"  # Update this if your file is in a different location
tree = ET.parse(xml_file)
root = tree.getroot()

flows = []
throughput = []
packet_loss = []
delay = []

# Extract flow statistics
for flow in root.findall("FlowStats/Flow"):
    flow_id = int(flow.get("flowId"))
    tx_packets = int(flow.get("txPackets"))
    rx_packets = int(flow.get("rxPackets"))
    lost_packets = int(flow.get("lostPackets"))
    tx_bytes = int(flow.get("txBytes"))
    rx_bytes = int(flow.get("rxBytes"))

    duration = (
        float(flow.get("timeLastRxPacket")[:-2]) - float(flow.get("timeFirstTxPacket")[:-2])
    ) * 1e-9  # Convert ns to seconds

    if duration > 0:
        tput = (rx_bytes * 8) / duration / 1e6  # Convert to Mbps
    else:
        tput = 0

    if tx_packets > 0:
        loss_ratio = (lost_packets / (rx_packets + lost_packets)) * 100
    else:
        loss_ratio = 0

    if rx_packets > 0:
        avg_delay = (float(flow.get("delaySum")[:-2]) / rx_packets) * 1e-9  # Convert ns to sec
    else:
        avg_delay = 0

    flows.append(flow_id)
    throughput.append(tput)
    packet_loss.append(loss_ratio)
    delay.append(avg_delay)

# Plot Throughput
plt.figure(figsize=(10, 5))
plt.bar(flows, throughput, color="blue")
plt.xlabel("Flow ID")
plt.ylabel("Throughput (Mbps)")
plt.title("Flow Throughput Analysis")
plt.grid(axis="y")
plt.show()

# Plot Packet Loss
plt.figure(figsize=(10, 5))
plt.bar(flows, packet_loss, color="red")
plt.xlabel("Flow ID")
plt.ylabel("Packet Loss (%)")
plt.title("Packet Loss Analysis")
plt.grid(axis="y")
plt.show()

# Plot Delay
plt.figure(figsize=(10, 5))
plt.bar(flows, delay, color="green")
plt.xlabel("Flow ID")
plt.ylabel("Mean Delay (seconds)")
plt.title("Flow Delay Analysis")
plt.grid(axis="y")
plt.show()