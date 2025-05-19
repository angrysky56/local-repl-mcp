
# Permanent Agent System Startup Script
# This script loads any existing agents or creates them if they don't exist

import os
import json
import random
from datetime import datetime, timedelta
import sys

# Add the modular empowerment framework to the path if it exists
mef_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modular_empowerment_framework")
if os.path.exists(mef_path):
    sys.path.append(mef_path)
    try:
        import modular_empowerment_framework as mef
        # Check if the MEF module is available
        print(f"Loaded Modular Empowerment Framework from: {mef_path}")
    except ImportError:
        print(f"Failed to import MEF from: {mef_path}")
        mef = None
else:
    print(f"MEF path not found: {mef_path}")
    mef = None

def create_or_load_permanent_agent_system(mef_module):
    """Create or load the permanent agent system

    Args:
        mef_module: The Modular Empowerment Framework module

    Returns:
        The agent system instance
    """
    if mef_module is None:
        print("MEF module not available. Cannot create agent system.")
        return None

    # Set up the data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_data")
    os.makedirs(data_dir, exist_ok=True)

    # Create or load the agent system
    system = mef_module.AgentSystem(data_dir=data_dir)

    # If no agents exist, create the default set
    if len(system.agents) == 0:
        print("Creating default agents...")

        # Create a memory agent
        memory_agent = mef_module.MemoryAgent(
            agent_id="memory_adv",
            name="Advanced Memory Agent",
            description="Stores and retrieves information for other agents"
        )
        system.add_agent(memory_agent)

        # Create a workflow agent
        workflow_agent = mef_module.WorkflowAgent(
            agent_id="workflow",
            name="Workflow Manager",
            description="Manages complex workflows involving multiple agents"
        )
        system.add_agent(workflow_agent)

        # Create a function tester agent
        tester_agent = mef_module.FunctionAgent(
            agent_id="function_tester",
            name="Function Tester",
            description="Tests and evaluates code functions"
        )
        system.add_agent(tester_agent)

        # Create a research agent
        research_agent = mef_module.ResearchAgent(
            agent_id="researcher",
            name="Research Assistant",
            description="Conducts research on specified topics"
        )
        system.add_agent(research_agent)

        # Save all agents
        system.save_all_agents()

    return system

# Create or load the permanent agent system
permanent_agents = create_or_load_permanent_agent_system(mef)

if permanent_agents:
    print(f"Permanent Agent System initialized with {len(permanent_agents.agents)} agents")
    for summary in permanent_agents.get_agent_summary():
        print(f"- {summary['id']} ({summary['type']}): Energy={summary['energy']:.2f}, Empowerment={summary['empowerment']:.2f}")
else:
    print("Failed to initialize agent system")
