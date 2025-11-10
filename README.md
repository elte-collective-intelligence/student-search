# Search & Rescue: Multi-Agent Reinforcement Learning (TorchRL)

[![CI](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/ci.yml)
[![Docker](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml/badge.svg)](https://github.com/elte-collective-intelligence/student-search/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/elte-collective-intelligence/student-search/branch/main/graph/badge.svg)](https://codecov.io/gh/elte-collective-intelligence/student-search)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](LICENSE)

This project simulates a multi-agent search-and-rescue mission using the PettingZoo MPE framework. Rescuers (adversaries) must guide victims to designated safe zones, navigating around obstacles and using cooperative intelligence to accomplish the task.

## Getting Started

### Prerequisites

Ensure you have Python 3.10 installed and necessary packages for running PettingZoo and reinforcement learning frameworks.

### Installation

1. Clone the repository:

    ```bash
    git clone https://gitlab.inf.elte.hu/student-projects-and-thesis/collective-intelligence/search.git
    cd search
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Overview

- **Objective**: Simulate a search-and-rescue operation where rescuers lead victims to specific safe zones based on clustering.
- **Framework**: Built using the PettingZoo MPE environment.
- **Agents**:
  - **Rescuers**: Adversarial agents guiding victims to the correct safe zones.
  - **Victims**: Agents that need to be rescued by being taken to matching safe zones.
- **Safe Zones**: Defined zones in the map’s four corners; each type of victim has a matching type of safe zone.

## Key Features

1. **Safe Zones**:
   - **Static Locations**: Safe zones are positioned at each corner of the map.
   - **Different Types**: Each victim type corresponds to a unique type of safe zone, introducing a clustering challenge.

2. **Reward System**:
   - Rescuers earn rewards for successfully moving victims to their designated safe zones.
   - Victims are incentivized to avoid capture, reinforcing the search-and-rescue dynamics.

3. **Collision Detection**:
   - Refined detection ensures rescues occur only when victims and rescuers are close.
   - Introduced obstacle collision logic to prevent straightforward rescues.

## Documentation  

For a detailed description of the environment, reward system, training pipeline, and evaluation process, refer to the complete project documentation:  
📄 [**Documentation.pdf**](Documentation.pdf)  

The documentation includes:

- Functionalities of the search-and-rescue environment.  
- Code structure and modular components.  
- Training and evaluation pipelines with performance metrics.  

## Usage

Run the following commands to start training or evaluating the environment:

```bash
# Training
python main.py train=true

# Evaluation
python main.py eval=true
```

 ![Run with 1 missing agent](images/Rescue1.mp4)

## Team Members

Jakab Etele, jangbo@inf.elte.hu​

Szarka Marcell, dia4sw@inf.elte.hu​

Pánczél Patrik, mbhr3h@inf.elte.hu

## License

This project is licensed under the MIT License. - see the [LICENSE](LICENSE) file for details.
