from ns import ns

class Monitor:
    def __init__(self, node_container, devices, internet_stack_helper):
        self.node_container = node_container
        self.devices = devices
        self.internet_stack_helper = internet_stack_helper
        self.flow_monitor = None
        self.node_status = {}

    def generate_animation_xml(self, file_name):
        """Generates an XML file for use with NetAnim."""
        anim = ns.AnimationInterface(file_name)
        for i, node in enumerate(self.node_container):
            anim.SetConstantPosition(node, i * 10, i * 10)  # Simple grid layout
        print(f"Animation XML generated: {file_name}")

    def create_pcap_files(self, prefix):
        """Creates pcap files for each device in the simulation."""
        for i, device in enumerate(self.devices):
            pcap_filename = f"{prefix}_{i}.pcap"
            ns.CsmaHelper.EnablePcap(pcap_filename, device)
            print(f"PCAP file generated: {pcap_filename}")

    def setup_flow_monitor(self):
        """Sets up NS-3 FlowMonitor."""
        flow_helper = ns.FlowMonitorHelper()
        self.flow_monitor = flow_helper.InstallAll()
        print("FlowMonitor setup completed.")

    def update_node_status(self):
        """Updates the dictionary with live status of all nodes."""
        for i, node in enumerate(self.node_container):
            self.node_status[node.GetId()] = {
                "is_up": node.IsUp(),
                "ip_addresses": self.get_ip_addresses(node),
            }

    def get_ip_addresses(self, node):
        """Helper function to retrieve IP addresses from a node."""
        addresses = []
        ipv4 = node.GetObject(ns.Ipv4.GetTypeId())
        if ipv4:
            for i in range(ipv4.GetNInterfaces()):
                for j in range(ipv4.GetNAddresses(i)):
                    addr = ipv4.GetAddress(i, j).GetLocal()
                    addresses.append(str(addr))
        return addresses

    def print_status(self):
        """Prints the current status of all nodes."""
        for node_id, status in self.node_status.items():
            print(f"Node {node_id}: Up={status['is_up']}, IPs={status['ip_addresses']}")
