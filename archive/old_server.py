"""
LocalREPL - A local Python REPL server for Claude Desktop.

This is a standalone file that implements a local Python REPL within an MCP server.
"""
import os
from typing import Dict, Tuple, Optional, Any
from mcp.server.fastmcp import FastMCP
import io
import sys
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr


class PythonREPL:
    """
    A stateful Python REPL implementation that maintains separate environment for each instance.
    """
    def __init__(self, repl_id: Optional[str] = None):
        self.repl_id = repl_id or str(uuid.uuid4())
        # Initialize a single namespace for environment
        # This is crucial for recursive functions to work properly
        self.namespace: Dict[str, Any] = {'__builtins__': __builtins__}

    def execute(self, code: str) -> Tuple[str, str, Any]:
        """
        Execute Python code in the REPL and return stdout, stderr, and the result.

        Args:
            code: The Python code to execute

        Returns:
            Tuple of (stdout, stderr, result)
        """
        # Use StringIO for capturing output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        result = None

        # Make sure code is a string to avoid issues
        if not isinstance(code, str):
            return ("", "Error: Code must be a string", None)

        # Skip empty code
        if not code.strip():
            return ("", "", None)

        try:
            # Redirect stdout and stderr to our capture objects
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # For multi-line code blocks or statements, use exec
                # For single expressions where we want a return value, use eval

                # Check if we're dealing with a single expression or a code block
                code_stripped = code.strip()

                # First try using exec for all code (safer for multi-line code)
                # Use a single namespace for both globals and locals to support recursive functions
                exec(code, self.namespace, self.namespace)

                # If it's potentially a simple expression, also try to evaluate it to get a return value
                # Only do this for single-line code without assignments or imports
                if ('\n' not in code_stripped and
                    '=' not in code_stripped and
                    not code_stripped.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while '))):
                    try:
                        result = eval(code_stripped, self.namespace, self.namespace)
                    except:
                        # Ignore errors in eval since we already executed with exec
                        pass
        except Exception:
            # Catch any exceptions and add to stderr
            stderr_capture.write(traceback.format_exc())

        # Return the captured output and result
        return (
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            result
        )


# Dictionary to store REPL instances by ID
repl_instances: Dict[str, PythonREPL] = {}

# Create the MCP server with more robust settings for connection handling
mcp = FastMCP(
    name="LocalREPL",
    stateless_http=False,
    debug=False  # Disable debug mode to reduce potential connection issues
)


@mcp.tool()
def create_python_repl() -> str:
    """
    Create a new Python REPL environment.

    Returns:
        str: ID of the new REPL
    """
    repl = PythonREPL()
    repl_instances[repl.repl_id] = repl
    return repl.repl_id


@mcp.tool()
def run_python_in_repl(code: str, repl_id: str) -> str:
    """
    Execute Python code in a REPL.

    Args:
        code: Python code to execute
        repl_id: ID of the REPL to use

    Returns:
        str: Result of the execution including stdout, stderr, and the return value
    """
    repl = repl_instances.get(repl_id)
    if not repl:
        return f"Error: REPL with ID {repl_id} not found. Please create a new REPL first."

    # Execute the code
    stdout, stderr, result = repl.execute(code)

    # Format the response - keep it simple to minimize potential issues
    output = ""

    # Add stdout if present
    if stdout:
        output += f"--- Output ---\n{stdout}\n\n"

    # Add stderr if present
    if stderr:
        output += f"--- Error ---\n{stderr}\n\n"

    # Add result if it's not None
    if result is not None:
        output += f"--- Result ---\n{result!r}"

    # Return response or default message
    return output.strip() if output.strip() else "Code executed successfully with no output."


@mcp.tool()
def list_active_repls() -> str:
    """List all active REPL instances and their IDs."""
    if not repl_instances:
        return "No active REPL instances."

    active_repls = [f"- {repl_id}" for repl_id in repl_instances]
    return "Active REPL instances:\n" + "\n".join(active_repls)


@mcp.tool()
def get_repl_info(repl_id: str) -> str:
    """
    Get information about a specific REPL instance.

    Args:
        repl_id: ID of the REPL to get info for

    Returns:
        str: Information about the REPL
    """
    repl = repl_instances.get(repl_id)
    if not repl:
        return f"Error: REPL with ID {repl_id} not found."

    # Get information about defined variables
    variables = []
    for name, value in repl.namespace.items():
        if not name.startswith("__"):
            try:
                type_name = type(value).__name__
                variables.append(f"{name}: {type_name}")
            except:
                variables.append(f"{name}: <unknown type>")

    if variables:
        return f"REPL ID: {repl_id}\nDefined variables:\n" + "\n".join(f"- {var}" for var in variables)
    else:
        return f"REPL ID: {repl_id}\nNo user-defined variables yet."


@mcp.tool()
def delete_repl(repl_id: str) -> str:
    """
    Delete a REPL instance.

    Args:
        repl_id: ID of the REPL to delete

    Returns:
        str: Confirmation message
    """
    if repl_id in repl_instances:
        del repl_instances[repl_id]
        return f"REPL {repl_id} deleted successfully."
    else:
        return f"REPL {repl_id} not found."


@mcp.prompt()
def python_repl_workflow() -> str:
    """
    A prompt template showing how to use the Python REPL.
    """
    return """
    This is a local Python REPL server. You can create, run, and manage Python REPL instances.

    Available commands:

    create_python_repl() - Creates a new Python REPL and returns its ID
    run_python_in_repl(code, repl_id) - Runs Python code in the specified REPL
    list_active_repls() - Lists all active REPL instances
    get_repl_info(repl_id) - Shows information about a specific REPL
    delete_repl(repl_id) - Deletes a REPL instance

    To run Python code:

    # First create a new REPL
    repl_id = create_python_repl()

    # Run some code
    result = run_python_in_repl(
      code="x = 42\nprint(f'The answer is {x}')",
      repl_id=repl_id
    )

    # Run more code in the same REPL (with state preserved)
    more_results = run_python_in_repl(
      code="import math\nprint(f'The square root of {x} is {math.sqrt(x)}')",
      repl_id=repl_id
    )

    # Check what variables are available in the environment
    environment_info = get_repl_info(repl_id)

    # When done, you can delete or keep the REPL
    delete_repl(repl_id)
    """


@mcp.prompt()
def modular_empowerment_workflow() -> str:
    """
    A prompt template showing how to use the Modular Empowerment Framework.
    """
    return """
    # Practical Modular Empowerment Framework Workflow

    This workflow demonstrates how to use the Modular Empowerment Framework (MEF) with Python REPL integration
    for practical applications with specialized agents. This system allows you to create intelligent agents
    with specific capabilities to tackle real-world tasks, both individually and collaboratively.

    ## Available commands:

    setup_modular_empowerment(path) - Sets up the MEF by specifying its location
    initialize_modular_empowerment(repl_id) - Initializes the MEF in a specific REPL
    create_python_repl() - Creates a new Python REPL and returns its ID
    run_python_in_repl(code, repl_id) - Runs Python code in the specified REPL
    list_active_repls() - Lists all active REPL instances
    get_repl_info(repl_id) - Shows information about a specific REPL
    delete_repl(repl_id) - Deletes a REPL instance

    ## Step 1: Setup the Modular Empowerment Framework

    First, specify where the modular_empowerment_framework is located on your system:

    ```python
    # Set up the framework (ask the user if the following path is correct- you only need to do this once per session)
    # You need to provide the path to where the modular_empowerment_framework is located
    setup_result = setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/modular_empowerment_framework")
    print(setup_result)
    ```

    ## Step 2: Initialize the Framework in a REPL

    Next, create a REPL and initialize the framework:

    ```python
    # Create a new REPL
    repl_id = create_python_repl()
    print(f"Created REPL: {repl_id}")

    # Initialize the MEF in this REPL
    init_result = initialize_modular_empowerment(repl_id=repl_id)
    print(init_result)
    ```

    ## Step 3: Create specialized agents with practical capabilities

    Create a team of agents with specialized capabilities for real-world tasks:

    ```python
    # Import necessary libraries
    import random
    import numpy as np

    # Create a simplified agent class to avoid type conversion issues
    class SpecializedAgent:
        def __init__(self, agent_id, agent_type, capabilities, initial_energy=1.0):
            self.agent_id = agent_id
            self.agent_type = agent_type
            self.capabilities = capabilities
            self.energy = initial_energy
            self.history = []
            self.empowerment = 0.5

        def __str__(self):
            return f"{self.agent_type} Agent ({self.agent_id})"

        def get_state_dict(self):
            "Convert object to a dictionary suitable for framework serialization"
            # Only include numeric values to avoid array conversion issues
            return {
                'energy': self.energy,
                'empowerment': self.empowerment,
                **{k: v for k, v in self.capabilities.items() if isinstance(v, (int, float))}
            }

    # Define agent types with specific capabilities
    agent_types = {
        'researcher': {
            'information_gathering': 0.9,
            'analysis': 0.8,
            'critical_thinking': 0.9,
            'communication': 0.7,
            'knowledge': 0.8
        },
        'writer': {
            'creativity': 0.9,
            'language': 0.9,
            'storytelling': 0.8,
            'editing': 0.7,
            'adaptability': 0.7
        },
        'programmer': {
            'coding': 0.9,
            'debugging': 0.8,
            'optimization': 0.7,
            'problem_solving': 0.9,
            'technical': 0.8
        },
        'decision_maker': {
            'risk_assessment': 0.8,
            'strategic': 0.9,
            'judgment': 0.8,
            'prioritization': 0.7,
            'adaptability': 0.7
        }
    }

    # Create a team of specialized agents
    agent_team = {}

    for agent_type, capabilities in agent_types.items():
        # Create multiple agents of each type
        for i in range(2):
            agent_id = f"{agent_type}_{i+1}"
            agent = SpecializedAgent(agent_id, agent_type, capabilities)
            agent_team[agent_id] = agent

            # Add the agent to the framework with appropriate state conversion
            mef.add_agent(agent_id, agent.get_state_dict())
            print(f"Added {agent} with capabilities: {capabilities}")

    print(f"\nTotal agents created: {len(agent_team)}")
    ```

    ## Step 4: Define practical tasks for your agents

    Define a set of tasks and collaborative workflows for your agents:

    ```python
    # Define individual tasks with requirements and measurements
    tasks = {
        'data_analysis': {
            'description': 'Analyze a dataset to extract insights and patterns',
            'requirements': {
                'analysis': 0.6,
                'critical_thinking': 0.7,
                'technical': 0.5
            },
            'difficulty': 0.7,
            'energy_cost': 0.3,
            'completion_reward': 0.4
        },
        'content_creation': {
            'description': 'Create engaging and informative content',
            'requirements': {
                'creativity': 0.7,
                'language': 0.8,
                'storytelling': 0.6,
                'knowledge': 0.5
            },
            'difficulty': 0.6,
            'energy_cost': 0.2,
            'completion_reward': 0.3
        },
        'code_development': {
            'description': 'Develop functional code to solve a specific problem',
            'requirements': {
                'coding': 0.8,
                'problem_solving': 0.7,
                'technical': 0.6,
                'debugging': 0.5
            },
            'difficulty': 0.8,
            'energy_cost': 0.4,
            'completion_reward': 0.5
        },
        'decision_making': {
            'description': 'Make strategic decisions with complex tradeoffs',
            'requirements': {
                'strategic': 0.7,
                'risk_assessment': 0.6,
                'judgment': 0.8,
                'prioritization': 0.7
            },
            'difficulty': 0.7,
            'energy_cost': 0.3,
            'completion_reward': 0.4
        }
    }

    # Define collaborative workflows that combine multiple tasks
    workflows = {
        'research_and_report': {
            'description': 'Research a topic and create a comprehensive report',
            'subtasks': ['data_analysis', 'content_creation'],
            'agent_types': ['researcher', 'writer'],
            'difficulty': 0.8,
            'completion_reward': 0.7
        },
        'software_development': {
            'description': 'Design and implement a software solution',
            'subtasks': ['decision_making', 'code_development'],
            'agent_types': ['decision_maker', 'programmer'],
            'difficulty': 0.9,
            'completion_reward': 0.8
        },
        'full_project': {
            'description': 'Complete research, planning, implementation, and documentation',
            'subtasks': ['data_analysis', 'decision_making', 'code_development', 'content_creation'],
            'agent_types': ['researcher', 'decision_maker', 'programmer', 'writer'],
            'difficulty': 1.0,
            'completion_reward': 1.0
        }
    }

    print("Defined individual tasks:")
    for name, task in tasks.items():
        print(f"- {name}: {task['description']} (Difficulty: {task['difficulty']})")

    print("\nDefined collaborative workflows:")
    for name, workflow in workflows.items():
        print(f"- {name}: {workflow['description']} (Difficulty: {workflow['difficulty']})")
        print(f"  Requires: {', '.join(workflow['agent_types'])}")
        print(f"  Subtasks: {', '.join(workflow['subtasks'])}")
    ```

    ## Step 5: Implement task execution and collaboration functions

    Create functions to assign and execute tasks:

    ```python
    # Function to check if an agent can perform a task
    def agent_capability_score(agent, task_requirements):
        "Calculate how capable an agent is at performing a task"
        if not task_requirements:
            return 1.0

        capability_scores = []

        for req, level in task_requirements.items():
            if req in agent.capabilities:
                agent_level = agent.capabilities[req]
                capability_scores.append(min(agent_level / level, 1.0) if level > 0 else 1.0)
            else:
                capability_scores.append(0.0)

        if not capability_scores:
            return 0.0

        return sum(capability_scores) / len(capability_scores)

    # Function to assign a task to the most capable agent
    def assign_task(task_name, available_agents):
        "Assign a task to the most capable available agent"
        if task_name not in tasks:
            print(f"Unknown task: {task_name}")
            return None, 0, 0

        task = tasks[task_name]
        best_agent_id = None
        best_score = 0

        for agent_id, agent in available_agents.items():
            # Skip agents that don't have enough energy
            if agent.energy < task['energy_cost']:
                continue

            # Calculate capability score for this task
            capability = agent_capability_score(agent, task.get('requirements', {}))

            # Consider agent's energy level
            energy_factor = agent.energy

            # Calculate final score
            score = capability * energy_factor

            if score > best_score:
                best_score = score
                best_agent_id = agent_id

        if best_agent_id is None:
            print(f"No suitable agent found for task: {task_name}")
            return None, 0, 0

        # Calculate expected quality based on agent capability
        expected_quality = best_score * (1.0 - task.get('difficulty', 0.5))

        return best_agent_id, expected_quality, task['energy_cost']

    # Function to execute a task
    def execute_task(agent_id, task_name, expected_quality):
        "Simulate task execution by an agent"
        if task_name not in tasks:
            print(f"Unknown task: {task_name}")
            return False, 0, 0

        task = tasks[task_name]
        agent = agent_team[agent_id]

        # Add some randomness to execution
        execution_factor = random.uniform(0.8, 1.2)

        # Calculate actual quality
        actual_quality = expected_quality * execution_factor

        # Determine success
        success = actual_quality >= task.get('difficulty', 0.5) * 0.7

        # Calculate empowerment gain
        empowerment_gain = task.get('completion_reward', 0.1) if success else -0.1

        # Update agent's energy and empowerment
        agent.energy = max(0, agent.energy - task['energy_cost'])
        agent.empowerment = max(0, min(1.0, agent.empowerment + empowerment_gain))

        # Update MEF agent state
        mef.eo.environment.get_agent(agent_id).set_state(agent.get_state_dict())

        # Record in agent's history
        agent.history.append({
            'task': task_name,
            'success': success,
            'quality': actual_quality,
            'energy_cost': task['energy_cost'],
            'empowerment_gain': empowerment_gain
        })

        return success, actual_quality, empowerment_gain
    ```

    ## Step 6: Run a practical agent simulation

    Run a simulation with your specialized agents performing tasks:

    ```python
    # Run a simulation of task execution
    print("Running task simulation...")

    # Reset agent energy levels
    for agent in agent_team.values():
        agent.energy = random.uniform(0.7, 1.0)

    # Define simulation parameters
    num_steps = 5
    tasks_to_try = list(tasks.keys())
    task_results = []

    for step in range(num_steps):
        print(f"\nSimulation Step {step+1}/{num_steps}")

        # Select a random task
        task_name = random.choice(tasks_to_try)
        print(f"Selected task: {task_name}")

        # Find the best agent for this task
        agent_id, expected_quality, energy_cost = assign_task(task_name, agent_team)

        if agent_id:
            agent = agent_team[agent_id]
            print(f"  Assigned to: {agent}")
            print(f"  Expected quality: {expected_quality:.2f}")
            print(f"  Energy cost: {energy_cost:.2f}")

            # Execute the task
            success, actual_quality, empowerment_gain = execute_task(agent_id, task_name, expected_quality)

            print(f"  Execution success: {success}")
            print(f"  Actual quality: {actual_quality:.2f}")
            print(f"  Empowerment gain: {empowerment_gain:.2f}")
            print(f"  Agent energy remaining: {agent.energy:.2f}")

            # Record results
            task_results.append({
                'step': step + 1,
                'task': task_name,
                'agent_id': agent_id,
                'success': success,
                'quality': actual_quality,
                'empowerment_gain': empowerment_gain
            })
        else:
            print("  No suitable agent found")

    # Generate simulation summary
    successes = sum(1 for result in task_results if result['success'])
    success_rate = (successes / len(task_results)) * 100 if task_results else 0
    avg_quality = sum(result['quality'] for result in task_results) / len(task_results) if task_results else 0

    print("\nSimulation Summary:")
    print(f"  Tasks attempted: {len(task_results)}")
    print(f"  Successful tasks: {successes} ({success_rate:.1f}% success rate)")
    print(f"  Average task quality: {avg_quality:.2f}")

    # Show agent status after simulation
    print("\nAgent Status After Simulation:")
    for agent_id, agent in agent_team.items():
        tasks_done = len([r for r in task_results if r['agent_id'] == agent_id])
        successes = len([r for r in task_results if r['agent_id'] == agent_id and r['success']])
        print(f"  {agent}: {successes}/{tasks_done} tasks completed, Energy: {agent.energy:.2f}, Empowerment: {agent.empowerment:.2f}")
    ```

    ## Step 7: Implement collaborative workflows

    Run collaborative workflows with multiple agents:

    ```python
    # Function to find a team for a collaborative workflow
    def find_team_for_workflow(workflow_name):
        "Find a suitable team of agents for a collaborative workflow"
        if workflow_name not in workflows:
            print(f"Unknown workflow: {workflow_name}")
            return {}

        workflow = workflows[workflow_name]
        team = {}

        # Find the best agent of each required type
        for agent_type in workflow['agent_types']:
            best_agent_id = None
            best_score = 0

            # Get all agents of this type
            type_agents = {aid: agent for aid, agent in agent_team.items()
                          if agent.agent_type == agent_type}

            for agent_id, agent in type_agents.items():
                # Skip agents with too little energy
                if agent.energy < 0.2:
                    continue

                # Calculate score based on capabilities and energy
                score = agent.empowerment * agent.energy

                if score > best_score:
                    best_score = score
                    best_agent_id = agent_id

            if best_agent_id:
                team[agent_type] = best_agent_id

        # Check if we have all required agent types
        if len(team) < len(workflow['agent_types']):
            print(f"Could not find suitable agents for all roles in workflow: {workflow_name}")
            return {}

        return team

    # Function to execute a collaborative workflow
    def execute_workflow(workflow_name, team):
        "Execute a collaborative workflow with a team of agents"
        if workflow_name not in workflows:
            print(f"Unknown workflow: {workflow_name}")
            return False, 0, {}

        workflow = workflows[workflow_name]

        # Calculate team capability
        team_capability = 0
        for agent_type, agent_id in team.items():
            agent = agent_team[agent_id]
            team_capability += agent.empowerment

        # Normalize team capability
        team_capability /= len(team)

        # Add synergy bonus
        synergy_bonus = 0.1 * (len(team) - 1)

        # Calculate expected quality
        expected_quality = team_capability * (1.0 - workflow['difficulty'])

        # Add randomness and synergy to execution
        execution_factor = random.uniform(0.8, 1.2) * (1 + synergy_bonus)
        actual_quality = expected_quality * execution_factor

        # Determine success
        success = actual_quality >= workflow['difficulty'] * 0.7

        # Calculate empowerment gains for each agent
        empowerment_gains = {}
        base_gain = workflow['completion_reward'] if success else -0.1

        for agent_type, agent_id in team.items():
            agent = agent_team[agent_id]

            # Different agent types might gain differently based on their role
            type_modifier = 1.0
            for subtask in workflow['subtasks']:
                if subtask in tasks:
                    # Check if this agent type is well-suited for this subtask
                    for req_key in tasks[subtask].get('requirements', {}):
                        if req_key in agent.capabilities and agent.capabilities[req_key] >= 0.8:
                            type_modifier = 1.2
                            break

            # Calculate gain and update agent
            agent_gain = base_gain * type_modifier
            agent.empowerment = max(0, min(1.0, agent.empowerment + agent_gain))
            agent.energy = max(0, agent.energy - 0.3)  # Energy cost for collaboration

            # Update MEF agent state
            mef.eo.environment.get_agent(agent_id).set_state(agent.get_state_dict())

            # Record in agent's history
            agent.history.append({
                'workflow': workflow_name,
                'success': success,
                'quality': actual_quality,
                'energy_cost': 0.3,
                'empowerment_gain': agent_gain
            })

            empowerment_gains[agent_id] = agent_gain

        return success, actual_quality, empowerment_gains

    # Run a collaborative workflow simulation
    print("\nRunning collaborative workflow simulation...")

    # Reset agent energy levels
    for agent in agent_team.values():
        agent.energy = random.uniform(0.7, 1.0)

    workflow_results = []

    for workflow_name in workflows:
        print(f"\nWorkflow: {workflow_name}")

        # Find a team for this workflow
        team = find_team_for_workflow(workflow_name)

        if team:
            print("  Team composition:")
            for agent_type, agent_id in team.items():
                print(f"    {agent_type}: {agent_team[agent_id]}")

            # Execute the workflow
            success, quality, empowerment_gains = execute_workflow(workflow_name, team)

            print(f"  Execution success: {success}")
            print(f"  Quality: {quality:.2f}")
            print("  Empowerment gains:")
            for agent_id, gain in empowerment_gains.items():
                print(f"    {agent_team[agent_id]}: {gain:.2f}")

            # Record results
            workflow_results.append({
                'workflow': workflow_name,
                'team': team,
                'success': success,
                'quality': quality,
                'empowerment_gains': empowerment_gains
            })
        else:
            print("  No suitable team found")

    # Generate workflow summary
    workflow_successes = sum(1 for result in workflow_results if result['success'])
    workflow_success_rate = (workflow_successes / len(workflow_results)) * 100 if workflow_results else 0
    workflow_avg_quality = sum(result['quality'] for result in workflow_results) / len(workflow_results) if workflow_results else 0

    print("\nCollaborative Workflow Summary:")
    print(f"  Workflows attempted: {len(workflow_results)}")
    print(f"  Successful workflows: {workflow_successes} ({workflow_success_rate:.1f}% success rate)")
    print(f"  Average workflow quality: {workflow_avg_quality:.2f}")

    # Show agent status after workflows
    print("\nAgent Status After Collaborative Workflows:")
    for agent_id, agent in agent_team.items():
        workflow_count = sum(1 for r in workflow_results if agent_id in r['team'].values())
        workflow_successes = sum(1 for r in workflow_results if agent_id in r['team'].values() and r['success'])
        print(f"  {agent}: {workflow_successes}/{workflow_count} workflows completed, Energy: {agent.energy:.2f}, Empowerment: {agent.empowerment:.2f}")
    ```

    ## Step 8: Visualize the results

    Create a visualization of the simulation results:

    ```python
    # Try importing matplotlib for visualization
    try:
        import matplotlib.pyplot as plt

        # Prepare data for visualization
        agent_ids = list(agent_team.keys())
        empowerment_values = [agent_team[aid].empowerment for aid in agent_ids]
        energy_values = [agent_team[aid].energy for aid in agent_ids]

        # Create a simple bar chart
        plt.figure(figsize=(12, 6))

        # Plot empowerment
        plt.subplot(1, 2, 1)
        bars = plt.bar(agent_ids, empowerment_values)

        # Color bars by agent type
        for i, bar in enumerate(bars):
            agent_type = agent_team[agent_ids[i]].agent_type
            if agent_type == 'researcher':
                bar.set_color('blue')
            elif agent_type == 'writer':
                bar.set_color('green')
            elif agent_type == 'programmer':
                bar.set_color('red')
            elif agent_type == 'decision_maker':
                bar.set_color('purple')

        plt.title('Agent Empowerment Levels')
        plt.xlabel('Agent ID')
        plt.ylabel('Empowerment')
        plt.xticks(rotation=45)

        # Plot energy
        plt.subplot(1, 2, 2)
        bars = plt.bar(agent_ids, energy_values)

        # Color bars by agent type
        for i, bar in enumerate(bars):
            agent_type = agent_team[agent_ids[i]].agent_type
            if agent_type == 'researcher':
                bar.set_color('blue')
            elif agent_type == 'writer':
                bar.set_color('green')
            elif agent_type == 'programmer':
                bar.set_color('red')
            elif agent_type == 'decision_maker':
                bar.set_color('purple')

        plt.title('Agent Energy Levels')
        plt.xlabel('Agent ID')
        plt.ylabel('Energy')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{setup_result}/output/agent_simulation_results.png")
        print("Visualization saved to {setup_result}/output/agent_simulation_results.png")

    except ImportError:
        print("Matplotlib not available. To visualize results, please install matplotlib.")
        print("\nTextual representation of results:")

        # Print textual representation of results
        print("\nAgent Empowerment and Energy Levels:")
        for agent_id, agent in agent_team.items():
            print(f"  {agent}: Empowerment = {agent.empowerment:.2f}, Energy = {agent.energy:.2f}")
    ```

    ## Step 9: Clean up- repl function tests, and when unneeded i.e. temporary agents and repl processes or on request.

    ```python
    # Clean up the REPL when done
    delete_repl(repl_id)
    print("REPL cleaned up")
    ```

    ## Extensions and customizations:

    - Create specialized agents with more sophisticated capabilities
    - Design complex, multi-stage tasks and workflows
    - Implement learning mechanisms for agents to improve their capabilities over time
    - Develop competitive scenarios where agents optimize for individual vs. group empowerment
    - Integrate with external systems for real-world task execution

    Note: This workflow avoids issues with NumPy array conversion by ensuring agent states only contain numeric values and not lists or complex objects. The simplified agent class provides compatibility with the MEF while adding practical task capabilities.
    """


# Store the MEF path globally
mef_path_store = None

@mcp.tool()
def setup_modular_empowerment(path: str = "") -> str:
    """
    Set up the Modular Empowerment Framework with a user-specified path.

    Args:
        path: Full path to the modular_empowerment_framework directory
              If empty, will look in the current directory

    Returns:
        str: Confirmation message with setup status
    """
    global mef_path_store

    if not path:
        # If no path provided, try the current directory
        path = os.path.join(os.getcwd(), "modular_empowerment_framework")
        if not os.path.exists(path):
            return (
                "❌ No path provided and couldn't find modular_empowerment_framework in the current directory.\n"
                "Please provide the full path to the modular_empowerment_framework directory."
            )

    # Check if the path exists
    if not os.path.exists(path):
        return f"❌ The specified path does not exist: {path}"

    # Check if it looks like a proper modular_empowerment_framework directory
    # We'll just check for a src directory as a basic validation
    src_dir = os.path.join(path, "src")
    if not os.path.exists(src_dir):
        return (
            f"⚠️  Warning: The path {path} exists but doesn't look like a modular_empowerment_framework directory.\n"
            "It should contain a 'src' directory with the framework code.\n"
            "You may need to create this structure or specify a different path."
        )

    # Store the path for future use
    mef_path_store = path

    return f"✅ Successfully configured modular_empowerment_framework at: {path}"


@mcp.tool()
def initialize_modular_empowerment(repl_id: str) -> str:
    """
    Initialize the Modular Empowerment Framework in a specific REPL.

    Args:
        repl_id: ID of the REPL to initialize in

    Returns:
        str: Result of the initialization
    """
    global mef_path_store

    # Check if the REPL exists
    if repl_id not in repl_instances:
        return f"❌ REPL with ID {repl_id} not found. Please create a new REPL first."

    # Check if the MEF path is set
    if not mef_path_store:
        return (
            "❌ The modular_empowerment_framework path is not set.\n"
            "Please run setup_modular_empowerment(path) first to configure the path."
        )

    # Create the initialization code
    init_code = f"""
import sys
import os
import random
import numpy as np
from typing import Dict, List, Any

# Add the framework directory to Python path
mef_path = "{mef_path_store}"
sys.path.append(mef_path)
print(f"Added framework path to Python path: {{mef_path}}")

try:
    # Import the integration module
    from src.integration.integration import ModularEmpowermentIntegration

    # Configure the MEF
    config = {{
        'individual_empowerment_weight': 0.4,
        'group_empowerment_weight': 0.4,
        'mdf_eo_balance': 0.6,
        'mdf_config': {{
            'quality_score_threshold': 0.5
        }},
        'eo_config': {{
            'cooperation_factor': 0.6
        }}
    }}

    # Initialize the integrated framework
    mef = ModularEmpowermentIntegration(config)
    print("✅ Modular Empowerment Framework initialized successfully!")
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Missing dependencies. Please install required packages:")
    print("  pip install numpy")
    raise
except Exception as e:
    print(f"❌ Error initializing framework: {{e}}")
    print("Make sure the framework is properly set up at the specified path.")
    raise
"""

    # Run the initialization code in the specified REPL
    result = run_python_in_repl(code=init_code, repl_id=repl_id)

    # Add success message if initialization was successful
    if "✅ Modular Empowerment Framework initialized successfully!" in result:
        return f"{result}\n\nFramework successfully initialized in REPL {repl_id}."
    else:
        return f"{result}\n\nFailed to initialize framework in REPL {repl_id}."

if __name__ == "__main__":
    mcp.run()
