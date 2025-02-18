# Extract base network and subnet mask
base_ip, base_mask = "192.168.1.0/15".split("/")
base_parts = list(map(int, base_ip.split(".")))
mask_bits = int(base_mask)

# Generate subnet mask dynamically
subnet_mask = [(0xFFFFFFFF << (32 - mask_bits) >> i) & 0xFF for i in (24, 16, 8, 0)]
subnet_mask = ".".join(map(str, subnet_mask))
print(subnet_mask)