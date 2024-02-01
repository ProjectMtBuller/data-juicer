#！/bin/bash

/root/sync_tools
/root/tools/card/run.sh -hds /root/dataset/data.hds
streamlit run app_mb.py --server.address 0.0.0.0 --server.port 80