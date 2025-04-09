from app import App
from topology import Topology
from ns import ns

import os
from monitor import Monitor

ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("PacketSink", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("OnOffApplication", ns.LOG_LEVEL_INFO)


def main():

    for root, dirs, files in os.walk("./sim/monitor"):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)

    duration = 100
    a = []
    for i in range(15):
        row = []
        for j in range(15):
            row.append(1 if i != j else 0)
        a.append(row)

    print("\n========================== INITIALIZING TOPOLOGY ==========================")
    topology = Topology(adj_matrix=a, links_delay=['1ms'], links_type=['p2p'])

    print("\n========================== INITIALIZING APPLICATION ==========================")
    app = App(topology, n_clients=5, n_servers=4, app_duration=30,
              links_delay=['1ms'], links_type=['p2p'])

    print("\n========================== RUNNING SIMULATION ==========================")
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(app.topology.nodes)
    mobility.Install(app.clients)
    mobility.Install(app.servers)

    print(
        f"\n--------------------------- Running simulation for {duration} seconds ---------------------------")

    app.monitor = Monitor(app.topology, app)

    anim = app.monitor.setup_animation(app.animFile)
    app.monitor.setup_pcap_capture()
    app.monitor.setup_packet_log()
    app.monitor.setup_flow_monitor()
    app.monitor.position_nodes(anim)

    ns.Simulator.Stop(ns.Seconds(duration))
    ns.Simulator.Run()

    app.monitor.get_node_ips_by_id()
    app.monitor.trace_routes()

    app.monitor.collect_flow_stats(app_port=app.app_port, filter_noise=True)
    app.monitor.get_packet_logs()

    ns.Simulator.Destroy()

    print("\n--------------------------- Simulation Complete ---------------------------")
    print(f"Animation file created at: {app.animFile}")
    print("Flow statistics saved in: ./sim/monitor/xml/flow-stats.xml")
    print("CSV summary saved in: ./sim/monitor/logs/flow_summary.csv")
    print("PCAP files saved in: ./sim/monitor/pcap/")
    print("Run NetAnim to visualize the simulation (load XML files from ./sim/monitor/xml/)")
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
