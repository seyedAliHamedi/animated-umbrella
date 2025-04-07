from ns import ns
import numpy as np
import traceback

from sim.topology import Topology
from sim.app import App
from sim.monitor import Monitor

class NetworkEnv:
    
    def __init__(self,topology_base_network, simulation_duration=100, min_throughput=1.0, max_latency=100.0, max_packet_loss=0.1, router_energy_cost=10, link_energy_cost=2,max_steps=20,each_step_duration=5,):
        self.simulation_duration = simulation_duration
        self.current_step = 0
        self.max_steps = max_steps
        self.each_step_duration = each_step_duration
        self.topology_base_network=topology_base_network
        
        self.min_throughput = min_throughput
        self.max_latency = max_latency
        self.max_packet_loss = max_packet_loss
        
        self.router_energy_cost = router_energy_cost
        self.link_energy_cost = link_energy_cost
        
        self.setup_environment()
    
    def setup_environment(self):
        
        self.topology = Topology()
        self.app = App(self.topology)
        self.monitor = Monitor(
            self.topology.nodes, 
            self.app
        )
            
        self.active_routers = [True] * self.topology.N_routers
        self.active_links = [True] * self.topology.N_links
            
        self.monitor.setup_flow_monitor()
    
    def reset(self):
        print("reset environment")
        ns.Simulator.Destroy()
        default_state = {
            'router_status': [True] * self.topology.N_routers,
            'link_status': [True] * self.topology.N_links,
            'throughput': [0.0] *  self.topology.N_routers,
            'latency': [float('inf')] *  self.topology.N_routers,
            'packet_loss': [1.0] *  self.topology.N_routers
        }
        
        self.setup_environment()
        
        self.current_step = 0
        return default_state  
    
    def step(self, action):
        self.apply_actions(action)
        self.run_simulation(self.each_step_duration)
        metrics = self.collect_metrics()
        reward = self.calculate_reward(metrics)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        state = {
            'router_status': self.active_routers,
            'link_status': self.active_links,
            'throughput': metrics['throughput'],
            'latency': metrics['latency'],
            'packet_loss': metrics['packet_loss']
        }
        
        energy = self.calculate_energy()
        info = {
            'metrics': metrics,
            'energy': energy
        }
        
        return state, reward, done, info
    
    def get_state(self):
        metrics = self.collect_metrics()
        return {
            'router_status': self.active_routers,
            'link_status': self.active_links,
            'throughput': metrics['throughput'],
            'latency': metrics['latency'],
            'packet_loss': metrics['packet_loss']
        }
    
    def run_simulation(self, duration):
        print(f"Running simulation for {duration} seconds...")
        ns.Simulator.Stop(ns.Seconds(duration))
        ns.Simulator.Run()
        print("Simulation completed")
    
    def apply_actions(self, action):
        if 'router_actions' in action and action['router_actions']:
            print(f"Applying router actions: {action['router_actions']}")
            for router_idx in action['router_actions']:
                if 0 <= router_idx < self.topology.N_routers:
                    self.active_routers[router_idx] = not self.active_routers[router_idx]
                    try:
                        router = self.topology.nodes.Get(router_idx)
                        state = self.active_routers[router_idx]
                        self.set_interface_state(router, -1, state)
                    except Exception as e:
                        print(f"Error toggling router {router_idx}: {e}")
            
        if 'interface_actions' in action and action['interface_actions']:
            print(f"Applying interface actions: {len(action['interface_actions'])} actions")
            for router_idx, interface_idx, state in action['interface_actions']:
                if 0 <= router_idx < self.topology.N_routers:
                    try:
                        router = self.topology.nodes.Get(router_idx)
                        self.set_interface_state(router, interface_idx, state)
                    except Exception as e:
                        print(f"Error setting interface state for router {router_idx}, interface {interface_idx}: {e}")
    
    def set_interface_state(self, r, interface_index, state):
        """
        Set the state of router interfaces.
        
        Parameters:
        r - The router node
        interface_index - The interface to modify (-1 for all interfaces)
        state - Boolean (True = UP, False = DOWN)
        """
        try:
            # Get IPv4 stack safely
            ipv4 = None
            try:
                ipv4 = r.GetObject[ns.Ipv4]()
            except Exception as e:
                print(f"Failed to get IPv4 object: {e}")
                return
                
            if ipv4 is None:
                print(f"Warning: No IPv4 stack on router {r.GetId()}")
                return
                
            num_interfaces = ipv4.GetNInterfaces()  # Get total interfaces
            
            if interface_index == -1:
                # Apply to all interfaces (except loopback, usually index 0)
                for i in range(1, num_interfaces):
                    try:
                        if state:
                            ipv4.SetUp(i)
                        else:
                            ipv4.SetDown(i)
                        
                        # Get routing protocol safely
                        routing_protocol = None
                        try:
                            routing_protocol = ipv4.GetRoutingProtocol()
                        except Exception as e:
                            print(f"Failed to get routing protocol: {e}")
                            continue
                            
                        if routing_protocol:
                            if not state:
                                routing_protocol.NotifyInterfaceDown(i)
                            else:
                                routing_protocol.NotifyInterfaceUp(i)
                    except Exception as e:
                        print(f"Error setting interface {i} state: {e}")
                
                print(f"Router {r.GetId()} {'enabled' if state else 'disabled'} (all {num_interfaces-1} interfaces)")
            else:
                # Apply only to the specified interface
                if 0 < interface_index < num_interfaces:
                    try:
                        if state:
                            ipv4.SetUp(interface_index)
                        else:
                            ipv4.SetDown(interface_index)
                        
                        # Get routing protocol safely
                        routing_protocol = None
                        try:
                            routing_protocol = ipv4.GetRoutingProtocol()
                        except Exception as e:
                            print(f"Failed to get routing protocol: {e}")
                            return
                            
                        if routing_protocol:
                            if not state:
                                routing_protocol.NotifyInterfaceDown(interface_index)
                            else:
                                routing_protocol.NotifyInterfaceUp(interface_index)
                    except Exception as e:
                        print(f"Error setting interface {interface_index} state: {e}")
                    
                    print(f"Router {r.GetId()} {'enabled' if state else 'disabled'} (interface {interface_index})")
                else:
                    print(f"Invalid interface index {interface_index} for router {r.GetId()}")
        except Exception as e:
            print(f"Error in set_interface_state: {e}")
            traceback.print_exc()

    def collect_metrics(self):
        """Collect network performance metrics."""
        # Initialize with default values
        n_clients = self.app.n_clients if hasattr(self.app, 'n_clients') else 1
        default_metrics = {
            'throughput': [0.0] * n_clients,
            'latency': [float('inf')] * n_clients,
            'packet_loss': [1.0] * n_clients
        }
        
        try:
            # Check if flow monitor is properly initialized
            if not hasattr(self.monitor, 'flow_monitor') or self.monitor.flow_monitor is None:
                print("Warning: Flow monitor not available")
                return default_metrics
                
            # Get flow statistics
            try:
                self.monitor.flow_monitor.CheckForLostPackets()
                stats = self.monitor.flow_monitor.GetFlowStats()
            except Exception as e:
                print(f"Error getting flow stats: {e}")
                return default_metrics
                
            # Initialize metrics lists
            throughput = []
            latency = []
            packet_loss = []
            
            # Process each flow - MODIFIED for Python bindings compatibility
            try:
                for flow_pair in stats:
                    # Get the FlowStats object directly without using operator[]
                    flow_id = flow_pair.first  # Get key
                    flow_stats = flow_pair.second  # Get value
                    
                    tx_packets = flow_stats.txPackets
                    rx_packets = flow_stats.rxPackets
                    tx_bytes = flow_stats.txBytes
                    
                    # Get delay sum safely
                    try:
                        delay_sum = flow_stats.delaySum.GetSeconds()
                    except Exception as e:
                        print(f"Error getting delay sum: {e}")
                        delay_sum = 0
                    
                    # Calculate metrics
                    if tx_packets > 0:
                        # Throughput in Mbps
                        current_throughput = (tx_bytes * 8.0) / (1024 * 1024)
                        
                        # Latency in ms
                        current_latency = (delay_sum / rx_packets) * 1000 if rx_packets > 0 else float('inf')
                        
                        # Packet loss ratio
                        current_packet_loss = (tx_packets - rx_packets) / tx_packets
                        
                        throughput.append(current_throughput)
                        latency.append(current_latency)
                        packet_loss.append(current_packet_loss)
            except Exception as e:
                print(f"Error iterating flows: {e}")
                traceback.print_exc()
            
            # If no flows were processed, use default values
            if not throughput:
                print("No flow statistics available, using defaults")
                return default_metrics
                
            # Return collected metrics
            return {
                'throughput': throughput,
                'latency': latency,
                'packet_loss': packet_loss
            }
        
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            traceback.print_exc()
            return default_metrics

    def calculate_reward(self, metrics):
        return -5
            
    def calculate_energy(self):
        return 5