from ns import ns

import os

from app import App
from monitor import Monitor
from topology import Topology

# ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("PacketSink", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("OnOffApplication", ns.LOG_LEVEL_INFO)
import time


def main():
    duration = 100
    t = time.time()
    # clean the monitor dir
    for root, dirs, files in os.walk("./sim/monitor"):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

    print("\n--------------------------- INITIALIZING TOPOLOGY ---------------------------")
    topology = Topology()

    print("\n--------------------------- INITIALIZING APPLICATION ---------------------------")
    app = App(topology, n_clients=5, n_servers=4, app_duration=30,
              links_delay=['1ms'], links_type=['p2p'])

    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(app.topology.nodes)
    mobility.Install(app.clients)
    mobility.Install(app.servers)

    print("\n========================== INITIALIZING MONITOR ==========================")
    app.monitor = Monitor(app.topology, app)

    print("\n--------------------------- Setting up animation ---------------------------")
    anim = app.monitor.setup_animation(app.animFile)
    print(f"Enhanced animation setup complete. Output file")
    print("- Routing tables, IP counters, and queue information enabled for NetAnim")

    # print("\n--------------------------- Setting up PCAP capture ---------------------------")
    # app.monitor.setup_pcap_capture()
    # print(f"PCAP capture enabled.")

    print("\n--------------------------- Setting up packet logs ---------------------------")
    app.monitor.setup_packet_log()
    print("Packet log setup Completed")

    print("\n--------------------------- Setting up flow monitor ---------------------------")
    app.monitor.setup_flow_monitor()
    print("FlowMonitor setup completed.")
    app.monitor.position_nodes(anim)

    print(
        f"\n--------------------------- RUNNING SIMULATION : {duration} seconds ---------------------------")

    ns.Simulator.Stop(ns.Seconds(duration))
    ns.Simulator.Run()

    print("\n--------------------------- Getting node IPs ---------------------------")
    node_ips = app.monitor.get_node_ips_by_id()
    for node_id, ips in node_ips.items():
        print(f"Node {node_id}: {', '.join(ips)}")

    print("\n--------------------------- Tracing routes from clients to servers ---------------------------")
    app.monitor.trace_routes()
    print("Routing trace completed.")

    print("\n--------------------------- Collecting flow statistics ---------------------------")
    app.monitor.collect_flow_stats(
        app_port=app.app_port, filter_noise=True, log=True)

    print("\n--------------------------- Processing Packet Logs ---------------------------")
    app.monitor.get_packet_logs()
    print(f"CSV summary file has been created successfully with path information.")

    ns.Simulator.Destroy()

    print("\n--------------------------- SIMULATION COMPLETED ---------------------------")
    print(f"Animation file created at: {app.animFile}")
    print("xml files saved in: ./sim/monitor/xml/")
    print("PCAP files saved in: ./sim/monitor/pcap/")
    print("packet logs saved in: ./sim/monitor/logs/")
    print("------------------------------------------------------------------")
    print(time.time()-t)


if __name__ == "__main__":
    main()
