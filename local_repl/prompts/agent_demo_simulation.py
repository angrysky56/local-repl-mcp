"""
Example script demonstrating the Modular Empowerment Framework with REPL integration.

This script sets up a simple environment with agents and runs a simulation using the
integrated framework, with each agent having its own REPL instance for computational tasks.
"""
def agent_demo_simulation() -> str:

    return """
    import sys
    import os
    import random
    import numpy as np
    import json
    from typing import Dict, List, Any

    from modular_empowerment_framework.src.integration import ModularEmpowermentIntegration


    def run_example():
        \"\"\"Run the example demonstration.\"\"\"
        print("Setting up Modular Empowerment Framework...")

        # Create configuration
        config = {
            'individual_empowerment_weight': 0.4,
            'group_empowerment_weight': 0.4,
            'mdf_eo_balance': 0.6,
            'mdf_config': {
                'quality_score_threshold': 0.5
            },
            'eo_config': {
                'cooperation_factor': 0.6
            }
        }

        # Initialize the integrated framework
        mef = ModularEmpowermentIntegration(config)

        # Create agents with initial states
        print("Creating agents...")
        agent_states = {
            'agent_1': {'position': 0.2, 'resources': 0.5, 'knowledge': 0.3},
            'agent_2': {'position': 0.8, 'resources': 0.3, 'knowledge': 0.7},
            'agent_3': {'position': 0.5, 'resources': 0.6, 'knowledge': 0.5}
        }

        for agent_id, state in agent_states.items():
            mef.add_agent(agent_id, state)

        # Setup REPL instances for agents
        print("Setting up REPL instances for each agent...")
        repl_ids = setup_agent_repls(list(agent_states.keys()))

        # Run a simulation
        print("Running simulation...")
        inputs = {
            'task': 'resource_allocation',
            'context': 'Limited resources must be allocated among agents',
            'constraints': 'Total resources cannot exceed available amount'
        }

        simulation_results = mef.run_simulation(num_steps=5, inputs=inputs)

        # Use agent REPLs to analyze results
        print("Analyzing results with agent REPLs...")
        analysis_results = analyze_results_with_repls(simulation_results, repl_ids)

        # Display results
        print("Simulation complete. Summary:")
        print(json.dumps(simulation_results['summary'], indent=2))

        print("\nAgent REPL Analysis:")
        print(json.dumps(analysis_results, indent=2))

        # Clean up REPLs
        cleanup_repls(repl_ids)

        print("Example complete!")
        return simulation_results, analysis_results

    def setup_agent_repls(agent_ids: List[str]) -> Dict[str, str]:
        \"\"\"
        Setup REPL instances for each agent.

        Args:
            agent_ids: List of agent IDs

        Returns:
            Dictionary mapping agent IDs to REPL IDs
        \"\"\"
        from local_repl.repl_script_library.create_python_repl import create_python_repl

        repl_ids = {}
        for agent_id in agent_ids:
            repl_id = create_python_repl()
            repl_ids[agent_id] = repl_id

            # Initialize the REPL with agent-specific libraries and functions
            setup_code = f\"\"\"
    import numpy as np
    import random
    import json

    # Agent-specific variables
    agent_id = "{agent_id}"
    agent_memory = {{"observations": [], "decisions": []}}

    # Agent-specific functions
    def process_data(data):
        # Process data based on agent's perspective
        return {{"processed_by": agent_id, "data": data}}

    def calculate_optimal_action(state, possible_actions):
        # Calculate the best action based on the agent's perspective
        scores = {{action: random.uniform(0, 1) for action in possible_actions}}
        return max(scores.items(), key=lambda x: x[1])[0]

    def analyze_simulation_step(step_data):
        # Analyze a simulation step
        analysis = {{
            "agent_id": agent_id,
            "step": step_data.get("step"),
            "empowerment_delta": random.uniform(-0.1, 0.1),
            "assessment": "Positive" if random.random() > 0.3 else "Negative"
        }}
        agent_memory["observations"].append(analysis)
        return analysis

    print(f"REPL for {agent_id} initialized")
    \"\"\"
            from local_repl.repl_script_library.run_python_in_repl import run_python_in_repl
            result = run_python_in_repl(repl_id, setup_code)
            print(f"REPL for {agent_id} setup result: {result}")

        return repl_ids

    def analyze_results_with_repls(simulation_results: Dict[str, Any], repl_ids: Dict[str, str]) -> Dict[str, Any]:
        \"\"\"
        Use agent REPLs to analyze simulation results.

        Args:
            simulation_results: Results from the simulation
            repl_ids: Dictionary mapping agent IDs to REPL IDs

        Returns:
            Analysis results from the REPLs
        \"\"\"
        from local_repl.repl_script_library.run_python_in_repl import run_python_in_repl

        analysis_results = {}

        for agent_id, repl_id in repl_ids.items():
            # Extract relevant steps for this agent
            agent_steps = [step for step in simulation_results.get('step_results', [])
                           if step.get('agent_id') == agent_id]

            analysis_code = f\"\"\"
    # Analyze the agent's steps in the simulation
    agent_steps = {json.dumps(agent_steps)}

    # Perform analysis
    results = []
    for step in agent_steps:
        analysis = analyze_simulation_step(step)
        results.append(analysis)

    # Calculate overall agent performance
    if results:
        overall_performance = {{
            "agent_id": agent_id,
            "num_steps_analyzed": len(results),
            "avg_empowerment_delta": np.mean([r["empowerment_delta"] for r in results]),
            "positive_steps_percent": 100 * len([r for r in results if r["assessment"] == "Positive"]) / len(results),
            "final_assessment": "Successful" if random.random() > 0.3 else "Needs Improvement"
        }}
    else:
        overall_performance = {{
            "agent_id": agent_id,
            "error": "No steps to analyze"
        }}

    # Store in agent memory
    agent_memory["decisions"].append(overall_performance)

    print(json.dumps(overall_performance))
    \"\"\"

            result = run_python_in_repl(repl_id, analysis_code)
            try:
                analysis_results[agent_id] = json.loads(result.strip())
            except json.JSONDecodeError:
                analysis_results[agent_id] = {"error": "Failed to decode REPL output", "output": result}

        return analysis_results

    def cleanup_repls(repl_ids: Dict[str, str]) -> None:
        \"\"\"
        Clean up REPL instances.

        Args:
            repl_ids: Dictionary mapping agent IDs to REPL IDs
        \"\"\"
        from local_repl.repl_script_library.delete_repl import delete_repl

        for agent_id, repl_id in repl_ids.items():
            try:
                delete_repl(repl_id)
                print(f"Deleted REPL for {agent_id}")
            except Exception as e:
                print(f"Error deleting REPL for {agent_id}: {e}")

    if __name__ == "__main__":
        run_example()
    """