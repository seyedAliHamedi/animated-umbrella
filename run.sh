#!/bin/bash

# Navigate to the correct directory
cd /Users/alihamedi/Desktop/work/ns3/ns-3-allinone/ns-3-dev/ || exit 1


# Start ns3 shell and run the Python script
./ns3 shell <<EOF
python ./sdn/main.py
exit
EOF