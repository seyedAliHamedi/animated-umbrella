#!/usr/bin/env python3
"""
Ping/RTT Simulator for Network Topologies
Simulates ICMP echo requests across a network and generates RTT measurements.
"""

import csv
import random
import numpy as np
from collections import defaultdict
import heapq


def parse_rate(rate_str):
    """Convert rate string (e.g., '100Mbps', '10Gbps') to bits per second."""
    rate_str = rate_str.strip().upper()
    if 'GBPS' in rate_str:
        return float(rate_str.replace('GBPS', '')) * 1e9
    elif 'MBPS' in rate_str:
        return float(rate_str.replace('MBPS', '')) * 1e6
    elif 'KBPS' in rate_str:
        return float(rate_str.replace('KBPS', '')) * 1e3
    else:
        return float(rate_str)


def parse_delay(delay_str):
    """Convert delay string (e.g., '1ms', '0.5ms') to milliseconds."""
    delay_str = delay_str.strip().lower()
    if 'ms' in delay_str:
        return float(delay_str.replace('ms', ''))
    elif 's' in delay_str:
        return float(delay_str.replace('s', '')) * 1000
    else:
        return float(delay_str)


class NetworkSimulator:
    def __init__(self, link_table, adj_matrix):
        """
        Initialize the network simulator.
        
        Args:
            link_table: List of tuples (i, j, type, rate, delay, queue_pkts, err_rate)
            adj_matrix: Adjacency matrix representing network connectivity
        """
        self.num_nodes = len(adj_matrix)
        self.adj_matrix = adj_matrix
        self.links = {}
        
        # Parse link table
        for link in link_table:
            i, j, link_type, rate_str, delay_str, queue_pkts, err_rate = link
            
            rate_bps = parse_rate(rate_str)
            delay_ms = parse_delay(delay_str)
            
            # Store link properties (bidirectional)
            self.links[(i, j)] = {
                'type': link_type,
                'rate': rate_bps,
                'delay': delay_ms,
                'queue': queue_pkts,
                'err_rate': err_rate
            }
            self.links[(j, i)] = {
                'type': link_type,
                'rate': rate_bps,
                'delay': delay_ms,
                'queue': queue_pkts,
                'err_rate': err_rate
            }
    
    def dijkstra_shortest_path(self, src, dst):
        """
        Find shortest path using Dijkstra's algorithm based on propagation delay.
        
        Returns:
            (path, total_delay) or (None, None) if no path exists
        """
        if src == dst:
            return [src], 0.0
        
        # Priority queue: (delay, node, path)
        pq = [(0.0, src, [src])]
        visited = set()
        
        while pq:
            delay, node, path = heapq.heappop(pq)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            if node == dst:
                return path, delay
            
            # Check all neighbors
            for neighbor in range(self.num_nodes):
                if self.adj_matrix[node][neighbor] == 1 and neighbor not in visited:
                    link_key = (node, neighbor)
                    if link_key in self.links:
                        link_delay = self.links[link_key]['delay']
                        new_delay = delay + link_delay
                        new_path = path + [neighbor]
                        heapq.heappush(pq, (new_delay, neighbor, new_path))
        
        return None, None
    
    def calculate_base_rtt(self, src, dst):
        """
        Calculate the base RTT (round-trip time) between two nodes.
        
        Returns:
            Base RTT in milliseconds, or None if no path exists
        """
        if src == dst:
            return 0.0
        
        # Find shortest path
        path, one_way_delay = self.dijkstra_shortest_path(src, dst)
        
        if path is None:
            return None
        
        # RTT is round-trip, so multiply by 2
        base_rtt = one_way_delay * 2
        
        return base_rtt
    
    def add_network_variance(self, base_rtt, path_length):
        """
        Add realistic variance to RTT measurements.
        
        Variance includes:
        - Queuing delay variations
        - Processing delay jitter
        - Clock precision effects
        """
        if base_rtt == 0:
            return 0.0
        
        # Base variance (proportional to RTT and path length)
        variance_factor = 0.02 + (path_length * 0.005)  # 2-5% base variance
        
        # Add Gaussian noise
        noise = random.gauss(0, base_rtt * variance_factor * 0.3)
        
        # Add occasional small spikes (queuing)
        if random.random() < 0.1:  # 10% chance of slight queuing
            noise += random.uniform(0, base_rtt * 0.05)
        
        rtt_with_noise = base_rtt + noise
        
        # Ensure RTT is never negative
        return max(0.0, rtt_with_noise)
    
    def simulate_ping(self, src, dst, count=10):
        """
        Simulate ping between two nodes.
        
        Args:
            src: Source node index
            dst: Destination node index
            count: Number of ping packets to send
        
        Returns:
            Dictionary with RTT statistics
        """
        base_rtt = self.calculate_base_rtt(src, dst)
        
        if base_rtt is None:
            # No path exists
            return None
        
        # Calculate path length for variance
        path, _ = self.dijkstra_shortest_path(src, dst)
        path_length = len(path) - 1 if path else 0
        
        # Generate multiple RTT samples
        rtt_samples = []
        for _ in range(count):
            rtt = self.add_network_variance(base_rtt, path_length)
            rtt_samples.append(rtt)
        
        # Calculate statistics
        return {
            'samples': rtt_samples,
            'avg': np.mean(rtt_samples),
            'min': np.min(rtt_samples),
            'max': np.max(rtt_samples),
            'count': count
        }
    
    def run_full_simulation(self, ping_count=10, output_file='ping.csv'):
        """
        Run ping simulation for all node pairs and save to CSV.
        
        Args:
            ping_count: Number of pings to send for each node pair
            output_file: Output CSV filename
        """
        results = []
        
        print(f"Running ping simulation for {self.num_nodes} nodes...")
        
        # Simulate pings between all pairs of nodes
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                ping_result = self.simulate_ping(src, dst, ping_count)
                
                if ping_result is not None:
                    results.append({
                        'src': src,
                        'dst': dst,
                        'avg_rtt': ping_result['avg'],
                        'min_rtt': ping_result['min'],
                        'max_rtt': ping_result['max'],
                        'rtt_count': ping_result['count'],
                        'samples': ping_result['samples']
                    })
        
        # Write to CSV
        self.write_csv(results, output_file, ping_count)
        print(f"Results written to {output_file}")
    
    def write_csv(self, results, filename, ping_count):
        """Write simulation results to CSV file."""
        with open(filename, 'w', newline='') as csvfile:
            # Create header
            fieldnames = ['src', 'dst', 'avg_rtt', 'min_rtt', 'max_rtt', 'rtt_count']
            fieldnames += [f'rtt_{i}' for i in range(ping_count)]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write each result
            for result in results:
                row = {
                    'src': result['src'],
                    'dst': result['dst'],
                    'avg_rtt': result['avg_rtt'],
                    'min_rtt': result['min_rtt'],
                    'max_rtt': result['max_rtt'],
                    'rtt_count': result['rtt_count']
                }
                # Add individual RTT samples
                for i, sample in enumerate(result['samples']):
                    row[f'rtt_{i}'] = sample
                
                writer.writerow(row)


def main():
    """Main function to run the simulation with example topology."""
    
    # Example 1: Simple 8-node topology (the one that generated your ping.csv)
    LINK_TABLE_EXAMPLE1 = [
        #  i, j,   type,  rate,     delay, queue_pkts,  err_rate
        # ---------- PATH-1  (fast / premium QoS) ----------
        (0, 1, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),
        (1, 2, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),
        (2, 3, 'p2p',  '10Gbps',  '1ms',   1000, 0.0),
        # ---------- PATH-2  (balanced) ----------
        (0, 6, 'p2p',  '10Mbps',   '2ms',    500, 0.0),
        (6, 7, 'p2p',  '10Mbps',   '2ms',    500, 0.0),
        (7, 3, 'p2p',  '10Mbps',   '2ms',    500, 0.0),
        # ---------- PATH-3  (low-energy / weak QoS) ----------
        (0, 4, 'csma', '100Kbps', '5ms',    200, 0.0),
        (4, 5, 'csma', '100Kbps', '5ms',    200, 0.0),
        (5, 3, 'csma', '100Kbps', '5ms',    200, 0.0),
        # ---------- bridging / redundancy ----------
        (1, 6, 'p2p',  '100Kbps',   '3ms',    500, 0.0),
        (2, 7, 'p2p',  '100Kbps',   '3ms',    500, 0.0),
        (4, 6, 'p2p',  '100Kbps',   '3ms',    500, 0.0),
        (5, 7, 'p2p',  '100Kbps',   '3ms',    500, 0.0),
    ]
    
    adj_matrix_example1 = [
        [0, 1, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 1, 0],
    ]
    
    # Example 2: Japanese network topology
    # Node index → name
    # 0 Hiroshima, 1 Sakyo, 2 Dojima, 3 Nara, 4 Komatso, 5 NTT Otemachi,
    # 6 Tsukuba, 7 KDDI Otemachi, 8 Akihabara, 9 Nezu, 10 Yogami,
    # 11 Hiyoshi, 12 Fujisawa
    
    LINK_TABLE_JAPAN = [
        #  i,  j,   type,  rate,      delay,    queue_pkts, err_rate
        (0,  2,  'p2p', '100Mbps',  '1.60ms',  10000,        0.0),  # Hiroshima–Dojima
        (1,  2,  'p2p', '100Mbps',  '0.25ms',  10000,        0.0),  # Sakyo–Dojima
        (1,  3,  'p2p', '100Mbps',  '0.20ms',  10000,        0.0),  # Sakyo–Nara
        (2,  3,  'p2p', '10Gbps',   '0.20ms',  10000,        0.0),  # Dojima–Nara
        (2,  4,  'p2p', '100Gbps',  '1.30ms',  10000,        0.0),  # Dojima–Komatsu
        (2,  5,  'p2p', '100Gbps',  '2.20ms',  10000,        0.0),  # Dojima–Fujisawa
        (2,  9,  'p2p', '10Gbps',   '2.50ms',  10000,        0.0),  # Dojima–Tsukuba
        (3,  5,  'p2p', '10Gbps',   '2.25ms',  10000,        0.0),  # Nara–Fujisawa
        (3, 12,  'p2p', '10Gbps',   '2.25ms',  10000,        0.0),  # Nara–KDDI Otemachi
        (4,  5,  'p2p', '100Gbps',  '1.75ms',  10000,        0.0),  # Komatsu–Fujisawa
        (5,  6,  'p2p', '10Gbps',   '0.10ms',  10000,        0.0),  # Fujisawa–Hiyoshi
        (5,  7,  'p2p', '150Gbps',  '0.12ms',  10000,        0.0),  # Fujisawa–Yagami
        (5, 10,  'p2p', '10Gbps',   '0.30ms',  10000,        0.0),  # Fujisawa–Akihabara
        (5, 12,  'p2p', '10Gbps',   '0.275ms', 10000,        0.0),  # Fujisawa–KDDI Otemachi
        (7,  8,  'p2p', '150Gbps',  '0.175ms', 10000,        0.0),  # Yagami–Nezu
        (7,  9,  'p2p', '10Gbps',   '0.50ms',  10000,        0.0),  # Yagami–Tsukuba
        (9, 10,  'p2p', '100Gbps',  '0.275ms', 10000,        0.0),  # Tsukuba–Akihabara
        (10, 11, 'p2p', '10Gbps',   '0.02ms',  10000,        0.0),  # Akihabara–NTT Otemachi
        (10, 12, 'p2p', '100Gbps',  '0.02ms',  10000,        0.0),  # Akihabara–KDDI Otemachi
    ]
    
    adj_matrix_japan = [
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 Hiroshima
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 Sakyo
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],  # 2 Dojima
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 3 Nara
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 4 Komatso
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1],  # 5 NTT Otemachi
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 6 Tsukuba
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],  # 7 KDDI Otemachi
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 8 Akihabara
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 9 Nezu
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],  # 10 Yogami
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 11 Hiyoshi
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 12 Fujisawa
    ]
    
    print("=" * 60)
    print("Network Ping/RTT Simulator")
    print("=" * 60)
    print("\nSelect topology to simulate:")
    print("1. 8-node example topology (generates similar to your ping.csv)")
    print("2. 13-node Japanese network topology")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        print("\nRunning simulation for Japanese network topology...")
        sim = NetworkSimulator(LINK_TABLE_JAPAN, adj_matrix_japan)
        sim.run_full_simulation(ping_count=10, output_file='ping_japan.csv')
    else:
        print("\nRunning simulation for 8-node example topology...")
        sim = NetworkSimulator(LINK_TABLE_EXAMPLE1, adj_matrix_example1)
        sim.run_full_simulation(ping_count=10, output_file='ping_example.csv')
    
    print("\nSimulation complete!")


if __name__ == '__main__':
    main()
