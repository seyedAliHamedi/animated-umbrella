from monitor import Monitor
from ns import ns
import os
import sys
import json

# ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("PacketSink", ns.LOG_LEVEL_INFO)
# ns.LogComponentEnable("OnOffApplication", ns.LOG_LEVEL_INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topology import Topology
from app import App

def main():
    if os.path.exists('animated-umbrella/src/monitor/logs/packets_log.txt'):
        os.remove('animated-umbrella/src/monitor/logs/packets_log.txt')
    duration = 100 
    print("\n========================== INITIALIZING TOPOLOGY ==========================")
    a= [
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0],
    ]
    
    topology = Topology(links_delay=['1ms'],links_type=['p2p'])
    print("\n========================== INITIALIZING APPLICATION ==========================")
    app = App(topology,n_clients=5,n_servers=4,app_duration=30,links_delay=['1ms'],links_type=['p2p'])
    
    print("\n========================== RUNNING SIMULATION ==========================")
    
    mobility = ns.MobilityHelper()
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(app.topology.nodes)
    mobility.Install(app.clients)
    mobility.Install(app.servers)
    
    if not app.monitor:
        app.monitor = Monitor(app.topology, app)
        
    anim = app.monitor.setup_animation(app.animFile)
    app.monitor.setup_pcap_capture()
    app.monitor.setup_packet_log()
    flowmonitor = app.monitor.setup_flow_monitor()
    
    app.monitor.position_nodes(anim)
        
    print(f"\n--------------------------- Running simulation for {duration} seconds ---------------------------")
        
    ns.Simulator.Stop(ns.Seconds(duration))

    app.monitor.setup_tracert()

    ns.Simulator.Run()
        
    if app.monitor:
        node_ips = app.monitor.get_node_ips_by_id()

        app.monitor.collect_flow_stats(app_port=app.app_port, filter_noise=True)

        app.monitor.get_packet_logs()
        # routes = app.monitor.get_all_routes()
        # with open('animated-umbrella/src/monitor/logs/routes.json', 'w') as json_file:
        #     str_routes = {f"{src}->{dst}": path for (src, dst), path in routes.items()}
        #     json.dump(str_routes, json_file, indent=4)
        
        ns.Simulator.Destroy()
        
        print("\n--------------------------- Simulation Complete ---------------------------")
        print(f"Animation file created at: {app.animFile}")
        print("Flow statistics saved in: ./src/monitor/xml/flow-stats.xml")
        print("CSV summary saved in: ./src/monitor/logs/flow_summary.csv")
        print("PCAP files saved in: ./src/monitor/pcap/")
        print("Run NetAnim to visualize the simulation (load XML files from ./src/monitor/xml/)")
        print("------------------------------------------------------------------")
    

if __name__ == "__main__":
    main()