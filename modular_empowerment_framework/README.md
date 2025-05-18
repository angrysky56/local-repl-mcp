# Modular Empowerment Framework

## Overview

The Modular Empowerment Framework (MEF) integrates two powerful systems:

1. **Modular Decision Framework (MDF)**: A multi-layered decision-making system that evaluates thoughts based on multiple criteria and adapts through feedback.

2. **Empowerment Optimization (EO)**: A multi-agent reinforcement learning system that maximizes both individual and group empowerment.

This integration leverages the complementary strengths of both systems:
- MDF provides a structured, ethically-guided decision-making process
- EO provides a mathematical framework for evaluating and optimizing actions based on their impact on future possibilities

## Project Structure

```
modular_empowerment_framework/
├── src/
│   ├── mdf/           # Modular Decision Framework
│   ├── eo/            # Empowerment Optimization
│   └── integration/   # Integration layer between MDF and EO
├── examples/          # Example scripts
└── README.md          # This file
```

## Features

- **Enhanced Decision Evaluation**: Uses empowerment to evaluate decisions based on impact on future possibilities
- **Dual Adaptation Mechanisms**: Adapts through both MDF's feedback mechanisms and EO's reinforcement learning
- **Multi-Agent Support**: Handles multiple agents with individual and group perspectives
- **REPL Integration**: Each agent can have its own Python REPL for computational tasks
- **Ethical Framework**: Decisions are evaluated against ethical principles

## Core Components

### Modular Decision Framework (MDF)

- **Thought Generation**: Creates possible thoughts/actions
- **Evaluation**: Scores thoughts on multiple metrics
- **Multi-Layered Analysis**: Analyzes thoughts from different perspectives (ethical, cognitive, etc.)
- **Self-Reflection**: Assesses confidence and alignment with principles
- **Feedback and Adaptation**: Learns from outcomes to improve future decisions

### Empowerment Optimization (EO)

- **Empowerment Calculation**: Measures how actions affect future possibilities
- **Environment Simulation**: Simulates how actions affect the environment
- **Group Empowerment**: Balances individual and group benefits
- **Policy Learning**: Updates action policies through reinforcement learning

### Integration Components

- **Translation Layer**: Maps between MDF thoughts and EO actions
- **Enhanced Evaluation**: Adds empowerment metrics to MDF evaluation
- **Integrated Decision Process**: Combines both frameworks' decision processes
- **REPL Integration**: Uses Python REPLs for agent-specific computations

## Getting Started

### Prerequisites

- Python 3.6+
- NumPy

### Installation

```bash
git clone https://github.com/angrysky56/modular_empowerment_framework.git
cd modular_empowerment_framework
```

### Running Examples

```bash
python examples/repl_integration_example.py
```

## Usage

Here's a simple example of how to use the framework:

```python
from src.integration import ModularEmpowermentIntegration

# Initialize the framework
mef = ModularEmpowermentIntegration()

# Add agents
mef.add_agent('agent_1', {'position': 0.2, 'resources': 0.5})
mef.add_agent('agent_2', {'position': 0.8, 'resources': 0.3})

# Run a decision process
result = mef.integrated_decision_process(
    inputs={'task': 'resource_allocation'},
    agent_id='agent_1'
)

# Run a simulation with multiple steps
simulation = mef.run_simulation(num_steps=5)
```

## REPL Integration

The framework supports integration with Python REPL instances, allowing each agent to have its own computational environment:

```python
# Create a REPL for an agent
from examples.create_python_repl import create_python_repl
repl_id = create_python_repl()

# Run code in the agent's REPL
from examples.run_python_in_repl import run_python_in_repl
result = run_python_in_repl(repl_id, "print('Hello from agent REPL')")

# When done, clean up
from examples.delete_repl import delete_repl
delete_repl(repl_id)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
