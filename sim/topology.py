from ns import ns
import os
import sys


from sim.utils import *


class Topology:
    def __init__(self, adj_matrix=sample_data['topology_adj_matrix'], links_type=sample_data['topology_links_type'], links_rate=sample_data['topology_links_rate'], links_delay=sample_data['topology_links_delay'], links_queue=sample_data['topology_links_queue'], links_errors=sample_data['topology_links_errors'], xml_file=sample_data['topology_xml_file']):

        self.adj_matrix = adj_matrix
        self.N_routers = len(self.adj_matrix)
        self.N_links = sum(sum(row) for row in self.adj_matrix) // 2
        self.links_type = links_type
        self.links_rate = links_rate
        self.links_delays = links_delay
        self.links_queue = links_queue
        self.links_errors = links_errors
        self.xml_file = xml_file

        self.nodes, self.devices, self.interfaces, self.ip_interfaces = self.initialize()

    def initialize(self):
        routers = ns.NodeContainer()
        routers.Create(self.N_routers)

        devices = []

        # links_types = distribute_values(self.links_type, self.N_links)
        # links_rate = distribute_values(self.links_rate, self.N_links)
        # link_delays = distribute_values(self.links_delays, self.N_links)
        # links_queue = distribute_values(self.links_queue, self.N_links)
        # links_errors = distribute_values(self.links_errors, self.N_links)

        # x = 0
        # for i in range(self.N_routers):
        #     for j in range(i, self.N_routers):
        #         if self.adj_matrix[i][j] == 1:
        #             if links_types[x] == "p2p":
        #                 link = ns.PointToPointHelper()
        #                 link.SetDeviceAttribute(
        #                     "DataRate", ns.StringValue(links_rate[x]))
        #                 link.SetChannelAttribute(
        #                     "Delay", ns.StringValue(link_delays[x]))
        #                 link.SetQueue("ns3::DropTailQueue", "MaxSize", ns.QueueSizeValue(
        #                     ns.QueueSize(f"{links_queue[x]}p")))
        #             elif links_types[x] == "csma":
        #                 link = ns.CsmaHelper()
        #                 link.SetChannelAttribute(
        #                     "DataRate", ns.DataRateValue(ns.DataRate(links_rate[x])))
        #                 link.SetChannelAttribute(
        #                     "Delay", ns.StringValue(link_delays[x]))
        #                 link.SetQueue("ns3::DropTailQueue", "MaxSize", ns.QueueSizeValue(
        #                     ns.QueueSize(f"{links_queue[x]}p")))

        # node_pair = ns.NodeContainer()
        # node_pair.Add(routers.Get(i))
        # node_pair.Add(routers.Get(j))
        # dev_pair = link.Install(node_pair)
        # devices.append(dev_pair)

        # error_model = ns.CreateObject[ns.RateErrorModel]()
        # error_model.SetRate(links_errors[x])
        # error_model.SetUnit(ns.RateErrorModel.ERROR_UNIT_PACKET)
        # dev_pair.Get(1).SetAttribute("ReceiveErrorModel",
        #                              ns.PointerValue(error_model))
        # dev_pair.Get(0).SetAttribute("ReceiveErrorModel",
        #                              ns.PointerValue(error_model))

        # x = x+1
        # ────────────────────────────────────────────────────────────────
        # 0)  Explicit link table  ➜  put it once near your class __init__
        # ────────────────────────────────────────────────────────────────
        LINK_TABLE = [
            #  i, j,   type,  rate,     delay, queue_pkts,  err_rate
            # ---------- PATH-1  (fast / premium QoS) ----------
            (0, 1, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),
            (1, 2, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),
            (2, 3, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),

            # ---------- PATH-2  (balanced) ----------
            (0, 6, 'p2p',  '1Gbps',   '2ms',    500, 0.0),
            (6, 7, 'p2p',  '1Gbps',   '2ms',    500, 0.0),
            (7, 3, 'p2p',  '1Gbps',   '2ms',    500, 0.0),

            # ---------- PATH-3  (low-energy / weak QoS) ----------
            (0, 4, 'csma', '100Mbps', '5ms',    200, 0.0),
            (4, 5, 'csma', '100Mbps', '5ms',    200, 0.0),
            (5, 3, 'csma', '100Mbps', '5ms',    200, 0.0),

            # ---------- bridging / redundancy ----------
            (1, 6, 'p2p',  '1Gbps',   '3ms',    500, 0.0),
            (2, 7, 'p2p',  '1Gbps',   '3ms',    500, 0.0),
            (4, 6, 'p2p',  '1Gbps',   '3ms',    500, 0.0),
            (5, 7, 'p2p',  '1Gbps',   '3ms',    500, 0.0),
        ]

        # Build a quick look-up dictionary:  edge_key -> (type, rate, delay, queue, err)
        link_specs = {
            tuple(sorted((i, j))): (ltype, rate, delay, qpkts, err)
            for (i, j, ltype, rate, delay, qpkts, err) in LINK_TABLE
        }

        # ────────────────────────────────────────────────────────────────
        # 1)  Replace your old “random distribute” block with this loop
        # ────────────────────────────────────────────────────────────────
        x = 0
        for i in range(self.N_routers):
            for j in range(i, self.N_routers):
                if self.adj_matrix[i][j] != 1:
                    continue

                # --- fetch the spec for this (i,j) edge ---------------
                spec_key = (i, j)
                assert spec_key in link_specs, f"Edge {spec_key} missing in LINK_TABLE"
                link_type, rate, delay, q_pkts, err_rate = link_specs[spec_key]

                # --- create the helper --------------------------------
                if link_type == "p2p":
                    link = ns.PointToPointHelper()
                    link.SetDeviceAttribute("DataRate", ns.StringValue(rate))
                    link.SetChannelAttribute("Delay", ns.StringValue(delay))
                    link.SetQueue("ns3::DropTailQueue",
                                  "MaxSize",
                                  ns.QueueSizeValue(ns.QueueSize(f"{q_pkts}p")))
                elif link_type == "csma":
                    link = ns.CsmaHelper()
                    link.SetChannelAttribute("DataRate", ns.StringValue(rate))
                    link.SetChannelAttribute("Delay", ns.StringValue(delay))
                    link.SetQueue("ns3::DropTailQueue",
                                  "MaxSize",
                                  ns.QueueSizeValue(ns.QueueSize(f"{q_pkts}p")))
                else:
                    raise ValueError(f"Unknown link_type '{link_type}'")

                # --- install ------------------------------------------
                node_pair = ns.NodeContainer()
                node_pair.Add(routers.Get(i))
                node_pair.Add(routers.Get(j))
                dev_pair = link.Install(node_pair)
                devices.append(dev_pair)
                # print(f"[OK] link #{x:02d}  {i}-{j} "
                #       f"type={link_type:4s}  rate={rate:>7}  delay={delay:>4}  "
                #       f"queue={q_pkts:4d}p  err={err_rate:.3g}")

                # --- optional error model (keep identical to your code) ----
                error_model = ns.CreateObject[ns.RateErrorModel]()
                error_model.SetRate(err_rate)
                error_model.SetUnit(ns.RateErrorModel.ERROR_UNIT_PACKET)
                dev_pair.Get(1).SetAttribute("ReceiveErrorModel",
                                             ns.PointerValue(error_model))
                dev_pair.Get(0).SetAttribute("ReceiveErrorModel",
                                             ns.PointerValue(error_model))

                x += 1   # keep counter if you need it elsewhere

        # print("──────── summary ────────")
        # print(f"links installed : {x}")
        # print(f"links expected   : {len(LINK_TABLE)}")
        # missing = set(link_specs) - {tuple(sorted((i, j)))
        #                              for i in range(self.N_routers)
        #                              for j in range(i, self.N_routers)
        #                              if self.adj_matrix[i][j] == 1}
        # print(f"specs w/out edge : {missing or 'none'}")
        internet = ns.InternetStackHelper()
        ipv4RoutingHelper = ns.Ipv4ListRoutingHelper()

        rip = ns.RipHelper()

        ipv4RoutingHelper.Add(rip, 10)

        internet.SetRoutingHelper(ipv4RoutingHelper)
        internet.Install(routers)

        y = 0
        ip_interfaces = []
        for i in range(self.N_routers):
            for j in range(i, self.N_routers):
                address = ns.Ipv4AddressHelper()
                address.SetBase(ns.Ipv4Address(
                    f"1.{i}.{j}.0"), ns.Ipv4Mask('255.255.255.0'))
                if self.adj_matrix[i][j] == 1:
                    ip_interfaces.append(address.Assign(devices[y]))
                    y += 1

        return routers, devices, internet, ip_interfaces
