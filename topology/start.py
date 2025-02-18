import math
from ns import ns

class Star:
    def __init__(self, n_subnets, n_device_per_subnet=1, n_routers=1, type_of_link="p2p",
                 link_throughputs=["5Mbps"], link_delays=["2ms"], base_networks=["192.168.1.0/24"]):

        self.endDevices = ns.NodeContainer()
        self.endDevices.Create(n_subnets)

        self.router = ns.NodeContainer()
        self.router.Create(n_routers)

        self.all_devices = ns.NodeContainer()
        self.all_devices.Add(self.endDevices)
        self.all_devices.Add(self.router)

        def distribute_values(values, count):
            chunk_size = math.ceil(count / len(values))
            return [val for val in values for _ in range(chunk_size)][:count]

        self.link_throughputs = distribute_values(link_throughputs, n_subnets)
        self.link_delays = distribute_values(link_delays, n_subnets)

        self.links = []
        self.devices = []

        for i in range(n_subnets):
            if type_of_link == "p2p":
                link = ns.PointToPointHelper()
            elif type_of_link == "csma":
                link = ns.CsmaHelper()

            link.SetDeviceAttribute("DataRate", ns.StringValue(self.link_throughputs[i]))
            link.SetChannelAttribute("Delay", ns.StringValue(self.link_delays[i]))

            device_pair = link.Install(self.endDevices.Get(i), self.router.Get(0))
            self.devices.append(device_pair)
            self.links.append(link)

        self.stack = ns.InternetStackHelper()
        self.stack.Install(self.all_devices)

        self.address = ns.Ipv4AddressHelper()
        self.ip_interfaces = []

        generated_subnets = []

        for i in range(n_subnets):
            if i < len(base_networks):  
                base_ip, base_mask = base_networks[i].split("/")
            else:
                base_ip, base_mask = base_networks[0].split("/")  # Use the first provided base network
                base_parts = list(map(int, base_ip.split(".")))
                base_parts[2] += i  # ✅ Increment third octet dynamically
                base_ip = f"{base_parts[0]}.{base_parts[1]}.{base_parts[2]}.0"

            mask_bits = int(base_mask)
            subnet_mask = [0, 0, 0, 0]
            for j in range(4):
                if mask_bits > 8:
                    subnet_mask[j] = 255
                    mask_bits -= 8
                else:
                    subnet_mask[j] = 256 - 2**(8 - mask_bits)
                    mask_bits = 0

            subnet_mask_str = ".".join(map(str, subnet_mask))
            print(base_ip,subnet_mask_str)
            self.address.SetBase(ns.Ipv4Address(base_ip), ns.Ipv4Mask(subnet_mask_str))
            self.ip_interfaces.append(self.address.Assign(self.devices[i]))

        ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
        
# Example Usage:
star_topology = Star(n_subnets=6, n_device_per_subnet=1, n_routers=1, type_of_link="p2p",
                     link_throughputs=['5Mbps', '6Mbps', '7Mbps'],
                     link_delays=['2ms', '3ms'],
                     base_networks=["192.168.1.0/24", "192.168.2.0/24", "192.168.3.0/24"])