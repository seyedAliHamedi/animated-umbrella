import math
from ns import ns


################# STAR TOPOLOGY #################
"""
            E1
            |
    E2-- [Router] --E3
            |
            E4
"""

class Star:
    def __init__(self, n_subnets=1, n_device_per_subnet=4, n_routers=1, type_of_link="p2p",
                 link_throughputs=["5Mbps"], link_delays=["2ms"], base_networks=["192.168.1.0/24"],animation_file="star_topology.xml"):

        self.n_subnets = n_subnets
        self.n_device_per_subnet = n_device_per_subnet
        self.n_routers = n_routers
        self.type_of_link = type_of_link
        self.link_throughputs = link_throughputs
        self.link_delays = link_delays
        self.base_networks = base_networks
        self.animation_file = animation_file
        self.nodes, self.devices, self.interfaces, self.ip_interfaces = self.initialize()

    def initialize(self):
        """Sets up the topology and returns nodes, devices, interfaces, and IP interfaces"""
        endDevices = []
        routers = ns.NodeContainer()
        routers.Create(self.n_routers)

        # Create subnets with multiple devices
        for _ in range(self.n_subnets):
            subnet_devices = ns.NodeContainer()
            subnet_devices.Create(self.n_device_per_subnet)
            endDevices.append(subnet_devices)

        all_nodes = ns.NodeContainer()
        all_nodes.Add(routers)
        for devices in endDevices:
            all_nodes.Add(devices)

        links = []
        devices = []

        link_throughputs = self._distribute_values(self.link_throughputs, self.n_subnets)
        link_delays = self._distribute_values(self.link_delays, self.n_subnets)

        for i in range(self.n_subnets):

            """Creates a link based on the specified type"""
            if self.type_of_link == "p2p":
                link = ns.PointToPointHelper()
            elif self.type_of_link == "csma":
                link = ns.CsmaHelper()
            else:
                raise ValueError("Unsupported link type")

            link.SetDeviceAttribute("DataRate", ns.StringValue(link_throughputs[i]))
            link.SetChannelAttribute("Delay", ns.StringValue(link_delays[i]))

            for j in range(self.n_device_per_subnet):
                dev_pair = link.Install(endDevices[i].Get(j), routers.Get(0))
                devices.append(dev_pair)
            links.append(link)

        stack = ns.InternetStackHelper()
        stack.Install(all_nodes)

        address = ns.Ipv4AddressHelper()
        ip_interfaces = []
        generated_subnets = {}

        for i in range(self.n_subnets):
            if i < len(self.base_networks):
                base_ip, base_mask = self.base_networks[i].split("/")
            else:
                base_ip, base_mask = self.base_networks[0].split("/")
                base_parts = list(map(int, base_ip.split(".")))
                base_parts[2] += i
                base_ip = f"{base_parts[0]}.{base_parts[1]}.{base_parts[2]}.0"

            mask_bits = int(base_mask)
            subnet_mask = self._calculate_subnet_mask(mask_bits)

            if base_ip in generated_subnets:
                generated_subnets[base_ip] += 1
                base_ip = f"{base_ip[:-1]}{generated_subnets[base_ip]}"
            else:
                generated_subnets[base_ip] = 1

            address.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask(subnet_mask))

            for j in range(self.n_device_per_subnet):
                ip_interfaces.append(address.Assign(devices[i * self.n_device_per_subnet + j]))

        ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

        return all_nodes, devices, stack, ip_interfaces

    # ========== HELPERS ==========
    def _distribute_values(self, values, count):
        """Evenly distributes values across a given count"""
        chunk_size = math.ceil(count / len(values))
        return [val for val in values for _ in range(chunk_size)][:count]

    def _calculate_subnet_mask(self, mask_bits):
        """Generates a subnet mask based on CIDR notation"""
        subnet_mask = [0, 0, 0, 0]
        for j in range(4):
            if mask_bits > 8:
                subnet_mask[j] = 255
                mask_bits -= 8
            else:
                subnet_mask[j] = 256 - 2 ** (8 - mask_bits)
                mask_bits = 0
        return ".".join(map(str, subnet_mask))
    
    def generate_animation_xml(self):
        anim = ns.AnimationInterface(self.animation_file)

        pos = ns.ListPositionAllocator()
        pos.Add(ns.Vector(50, 50, 0))

        angle_step = 360 / (self.n_subnets * self.n_device_per_subnet)
        angle = 0
        radius = 30  
        for i in range(self.n_subnets):
            for j in range(self.n_device_per_subnet):
                x = 50 + radius * math.cos(math.radians(angle))
                y = 50 + radius * math.sin(math.radians(angle))
                pos.Add(ns.Vector(x, y, 0))
                angle += angle_step  

        mobility = ns.MobilityHelper()
        mobility.SetPositionAllocator(pos)
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.nodes)

        print(f"📂 Animation XML file generated: {self.animation_file}")

    # ========== GETTERS ==========
    def get_nodes(self):
        """Returns all network nodes (end devices + routers)"""
        return self.nodes

    def get_routers(self):
        """Returns the routers in the topology"""
        return self.nodes.Get(0)  # First node is always the router

    def get_end_devices(self):
        """Returns all end devices"""
        return [self.nodes.Get(i) for i in range(1, self.nodes.GetN())]

    def get_devices(self):
        """Returns all network devices (nodes with links installed)"""
        return self.devices

    def get_interfaces(self):
        """Returns interfaces (Devices + Internet stack)"""
        return self.interfaces

    def get_ip_interfaces(self):
        """Returns all assigned IP interfaces"""
        return self.ip_interfaces
    
    def summary(self):
        print("\n" + "=" * 40)
        print("🔥 STAR TOPOLOGY NETWORK SUMMARY 🔥")
        print("=" * 40)
        print(f"🔹 Subnets: {self.n_subnets}")
        print(f"🔹 Devices per Subnet: {self.n_device_per_subnet}")
        print(f"🔹 Routers: {self.n_routers}")
        print(f"🔹 Link Type: {self.type_of_link}")
        print(f"🔹 Base Networks: {', '.join(self.base_networks)}")
        print("-" * 40)

        print("🔹 Routers:")
        for r in [self.nodes.Get(0)]:
            print(f"   - Router {r.GetId()}")

        print("\n🔹 End Devices:")
        for ed in self.get_end_devices():
            print(f"   - Device {ed.GetId()}")

        print("\n🔹 IP Interfaces:")
        for i, ip in enumerate(self.get_ip_interfaces()):
            print(f"   - Interface {i}: {ip}")

        print("\n✨ TOPOLOGY SHAPE ✨")
        print(self._generate_topology_graph())
        print("=" * 40 + "\n")


# ========================== EXAMPLE USAGE ==========================
star_topology = Star(n_subnets=1, n_device_per_subnet=6, n_routers=1, type_of_link="csma",
                     link_throughputs=['5Mbps', '6Mbps', '7Mbps'],
                     link_delays=['2ms', '3ms'],
                     base_networks=["192.168.1.0/24"])

star_topology.generate_animation_xml()
star_topology.summary()