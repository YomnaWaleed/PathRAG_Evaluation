PathRAG_Evaluation/
├── data/
│   ├── AUTOSAR_SWS_ECUStateManager.pdf
│   └── Automotive_SPICE_PAM_31_EN.pdf
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics implementation
│   ├── multi_hop_simulator.py  # Multi-hop query simulation
│   └── test_cases.py       # Test cases and ground truth data
├── config/
│   └── settings.py         # Configuration and environment variables
├── utils/
│   ├── data_loader.py      # PDF loading and processing
│   └── visualization.py    # Results visualization
├── main.py                 # Main evaluation script
└── requirements.txt