"""
Empowerment Optimization (EO) framework implementation.

This module contains the base components for the EO system, focusing on
calculating and optimizing empowerment in multi-agent environments.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    """Represents an individual agent in the empowerment optimization system."""

    def __init__(self, agent_id: str, initial_state: Optional[Dict[str, Any]] = None):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state of the agent
        """
        self.agent_id = agent_id
        self.state = initial_state or {}
        self.policy: Optional[Policy] = None
        self.replay_buffer = []
        self.history = []

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"Agent {self.agent_id}"

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return self.state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the agent.

        Args:
            state: New state for the agent
        """
        self.state = state

    def update_state(self, state_update: Dict[str, Any]) -> None:
        """
        Update part of the agent's state.

        Args:
            state_update: Partial state update
        """
        self.state.update(state_update)


class Environment:
    """Represents the environment in which agents operate."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the environment.

        Args:
            config: Configuration for the environment
        """
        self.config = config or {}
        self.agents: Dict[str, Agent] = {}
        self.global_state: Dict[str, Any] = {}
        self.history = []

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the environment.

        Args:
            agent: Agent to add
        """
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} to environment")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent by ID.

        Args:
            agent_id: ID of the agent to get

        Returns:
            The agent if found, None otherwise
        """
        return self.agents.get(agent_id)

    def get_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the states of all agents.

        Returns:
            Dictionary mapping agent IDs to their states
        """
        return {agent_id: agent.get_state() for agent_id, agent in self.agents.items()}

    def simulate_transition(self, agent_id: str, action: Any) -> Dict[str, Any]:
        """
        Simulate a state transition for an agent given an action.

        Args:
            agent_id: ID of the agent
            action: Action to simulate

        Returns:
            Predicted next state
        """
        # Placeholder implementation - would use actual environment dynamics
        agent = self.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found in environment")
            return {}

        current_state = agent.get_state()
        # Simple placeholder transition function
        next_state = {k: v + np.random.normal(0, 0.1) for k, v in current_state.items()}

        logger.debug(f"Simulated transition for agent {agent_id}: {action} -> {next_state}")
        return next_state

    def execute_action(self, agent_id: str, action: Any) -> Dict[str, Any]:
        """
        Execute an action for an agent in the environment.

        Args:
            agent_id: ID of the agent
            action: Action to execute

        Returns:
            Next state after execution
        """
        agent = self.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found in environment")
            return {}

        # Simulate the action's effect
        next_state = self.simulate_transition(agent_id, action)

        # Update the agent's state
        agent.set_state(next_state)

        # Record in history
        self.history.append({
            'agent_id': agent_id,
            'action': action,
            'next_state': next_state,
            'timestamp': len(self.history)
        })

        logger.debug(f"Executed action for agent {agent_id}: {action} -> {next_state}")
        return next_state


class EmpowermentModel:
    """Model for estimating empowerment from state-action pairs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the empowerment model.

        Args:
            config: Configuration for the model
        """
        self.config = config or {}
        self.weights = np.random.randn(10)  # Placeholder for model weights

    def estimate_empowerment(self, state: Dict[str, Any], action: Any) -> float:
        """
        Estimate empowerment for a state-action pair.

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Estimated empowerment value
        """
        # Filter out non-numeric values to prevent array conversion issues
        numeric_values = []
        for value in state.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(value)
            elif isinstance(value, (list, tuple)) and all(isinstance(v, (int, float)) for v in value):
                # If it's a list/tuple of numbers, use the average
                numeric_values.append(sum(value) / len(value) if value else 0.0)
        
        # If no numeric values found, use a default approach
        if not numeric_values:
            return 0.5  # Default empowerment value
            
        # Convert to numpy array of safe numeric values
        state_vector = np.array(numeric_values)
        action_value = float(hash(str(action)) % 1000) / 1000.0

        # Simple linear model for demonstration
        # Ensure dimensions match by using the minimum length
        weights_subset = self.weights[:min(len(state_vector), len(self.weights))]
        state_subset = state_vector[:len(weights_subset)]
        
        empowerment = np.dot(state_subset, weights_subset) + action_value

        # Normalize to [0, 1] range
        empowerment = 1.0 / (1.0 + np.exp(-empowerment))  # Sigmoid function

        logger.debug(f"Estimated empowerment: {empowerment}")
        return empowerment

    def update(self, experiences: List[Dict[str, Any]]) -> None:
        """
        Update the model based on experiences.

        Args:
            experiences: List of experience dictionaries with state, action, reward info
        """
        # Placeholder implementation - would use actual learning algorithm
        for _ in range(min(10, len(experiences))):
            self.weights += np.random.normal(0, 0.01, size=len(self.weights))

        logger.debug("Updated empowerment model")


class Policy:
    """Policy for selecting actions based on states."""

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the policy.

        Args:
            agent_id: ID of the agent this policy belongs to
            config: Configuration for the policy
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.weights = np.random.randn(10)  # Placeholder for policy weights

    def sample_actions(self, state: Dict[str, Any], num_samples: int = 5) -> List[Any]:
        """
        Sample possible actions from the policy.

        Args:
            state: Current state
            num_samples: Number of actions to sample

        Returns:
            List of sampled actions
        """
        # Placeholder implementation - would use actual policy sampling
        actions = [f"action_{i}" for i in range(num_samples)]

        logger.debug(f"Sampled {num_samples} actions for agent {self.agent_id}")
        return actions

    def update(self, experiences: List[Dict[str, Any]]) -> None:
        """
        Update the policy based on experiences.

        Args:
            experiences: List of experience dictionaries with state, action, reward info
        """
        # Placeholder implementation - would use actual policy optimization
        for _ in range(min(10, len(experiences))):
            self.weights += np.random.normal(0, 0.01, size=len(self.weights))

        logger.debug(f"Updated policy for agent {self.agent_id}")


class EOCore:
    """Core implementation of the Empowerment Optimization framework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EO system.

        Args:
            config: Configuration parameters for the EO system
        """
        # Default configuration
        self.config = {
            'cooperation_factor': 0.5,  # Lambda in the paper
            'num_action_samples': 10,
            'simulation_horizon': 5,
            'batch_size': 32,
            'learning_rate': 0.01
        }

        # Override defaults with provided config
        if config:
            self.config.update(config)

        self.environment = Environment()
        self.model = EmpowermentModel()
        self.policies = {}
        self.replay_buffer = []

        logger.info("Initialized Empowerment Optimization framework")

    def add_agent(self, agent_id: str, initial_state: Optional[Dict[str, Any]] = None) -> Agent:
        """
        Add a new agent to the system.

        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state of the agent

        Returns:
            The created agent
        """
        agent = Agent(agent_id, initial_state)
        policy = Policy(agent_id)

        self.environment.add_agent(agent)
        self.policies[agent_id] = policy
        agent.policy = policy

        logger.info(f"Added agent {agent_id} to EO system")
        return agent

    def calculate_individual_empowerment(self, agent_id: str, action: Any) -> float:
        """
        Calculate empowerment for an individual agent.

        Args:
            agent_id: ID of the agent
            action: Action to evaluate

        Returns:
            Empowerment value
        """
        agent = self.environment.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return 0.0

        state = agent.get_state()
        empowerment = self.model.estimate_empowerment(state, action)

        logger.debug(f"Calculated individual empowerment for agent {agent_id}: {empowerment}")
        return empowerment

    def calculate_group_empowerment(self, actions: Dict[str, Any]) -> float:
        """
        Calculate empowerment for the group of agents.

        Args:
            actions: Dictionary mapping agent IDs to their actions

        Returns:
            Group empowerment value
        """
        individual_empowerments = {}

        for agent_id, action in actions.items():
            individual_empowerments[agent_id] = self.calculate_individual_empowerment(agent_id, action)

        # Simple weighted sum for group empowerment
        group_empowerment = sum(individual_empowerments.values()) / max(1, len(individual_empowerments))

        logger.debug(f"Calculated group empowerment: {group_empowerment}")
        return group_empowerment

    def select_best_action(self, agent_id: str) -> Any:
        """
        Select the best action for an agent based on empowerment.

        Args:
            agent_id: ID of the agent

        Returns:
            The selected action
        """
        agent = self.environment.get_agent(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return None

        state = agent.get_state()
        policy = self.policies[agent_id]

        # Sample actions from policy
        possible_actions = policy.sample_actions(state, self.config['num_action_samples'])

        # Calculate empowerment for each action
        action_empowerments = {}
        for action in possible_actions:
            empowerment = self.calculate_individual_empowerment(agent_id, action)
            action_empowerments[action] = empowerment

        # Select action with highest empowerment
        if not action_empowerments:
            logger.warning(f"No actions available for agent {agent_id}")
            return None

        best_action = max(action_empowerments.items(), key=lambda x: x[1])[0]

        logger.debug(f"Selected best action for agent {agent_id}: {best_action}")
        return best_action

    def execute_actions(self, actions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Execute actions for multiple agents.

        Args:
            actions: Dictionary mapping agent IDs to their actions

        Returns:
            Dictionary mapping agent IDs to their next states
        """
        next_states = {}

        for agent_id, action in actions.items():
            next_state = self.environment.execute_action(agent_id, action)
            next_states[agent_id] = next_state

        logger.debug(f"Executed actions for {len(actions)} agents")
        return next_states

    def calculate_reward(self, agent_id: str, individual_empowerment: float, group_empowerment: float) -> float:
        """
        Calculate reward for an agent based on individual and group empowerment.

        Args:
            agent_id: ID of the agent
            individual_empowerment: Empowerment value for the agent
            group_empowerment: Empowerment value for the group

        Returns:
            Reward value
        """
        lambda_value = self.config['cooperation_factor']

        # r_i = (1 - λ) * E_i + λ * E_group
        reward = (1 - lambda_value) * individual_empowerment + lambda_value * group_empowerment

        logger.debug(f"Calculated reward for agent {agent_id}: {reward}")
        return reward

    def store_experience(self, agent_id: str, state: Dict[str, Any], action: Any,
                       reward: float, next_state: Dict[str, Any]) -> None:
        """
        Store an experience in the replay buffer.

        Args:
            agent_id: ID of the agent
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        experience = {
            'agent_id': agent_id,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'timestamp': len(self.replay_buffer)
        }

        self.replay_buffer.append(experience)

        # Also store in agent's personal buffer
        agent = self.environment.get_agent(agent_id)
        if agent:
            agent.replay_buffer.append(experience)

        logger.debug(f"Stored experience for agent {agent_id}")

    def update_policies(self) -> None:
        """Update all agent policies based on experiences."""
        for agent_id, policy in self.policies.items():
            agent = self.environment.get_agent(agent_id)
            if not agent:
                continue

            # Get agent's experiences
            experiences = agent.replay_buffer[-self.config['batch_size']:]
            if not experiences:
                continue

            # Update policy
            policy.update(experiences)

        logger.debug("Updated all agent policies")

    def update_model(self) -> None:
        """Update the empowerment model based on experiences."""
        # Get recent experiences
        experiences = self.replay_buffer[-self.config['batch_size']:]
        if not experiences:
            return

        # Update model
        self.model.update(experiences)

        logger.debug("Updated empowerment model")

    def step(self) -> Dict[str, Any]:
        """
        Perform a single step of the EO process for all agents.

        Returns:
            Results of the step
        """
        logger.info("Starting EO step")

        # Get all agent IDs
        agent_ids = list(self.environment.agents.keys())

        # Select best actions for all agents
        actions = {}
        for agent_id in agent_ids:
            actions[agent_id] = self.select_best_action(agent_id)

        # Calculate group empowerment before execution
        group_empowerment = self.calculate_group_empowerment(actions)

        # Store original states
        original_states = {}
        for agent_id in agent_ids:
            agent = self.environment.get_agent(agent_id)
            if agent is not None:
                original_states[agent_id] = agent.get_state()

        # Execute actions
        next_states = self.execute_actions(actions)

        # Calculate rewards and store experiences
        for agent_id in agent_ids:
            if agent_id in original_states:
                individual_empowerment = self.calculate_individual_empowerment(agent_id, actions[agent_id])
                reward = self.calculate_reward(agent_id, individual_empowerment, group_empowerment)

                self.store_experience(
                    agent_id,
                    original_states[agent_id],
                    actions[agent_id],
                    reward,
                    next_states[agent_id]
                )

        # Update policies and model
        self.update_policies()
        self.update_model()

        results = {
            'actions': actions,
            'group_empowerment': group_empowerment,
            'next_states': next_states
        }

        logger.info("Completed EO step")
        return results
    def run(self, num_steps: int = 10) -> Dict[str, Any]:
        """
        Run the EO process for multiple steps.

        Args:
            num_steps: Number of steps to run

        Returns:
            Results of the run
        """
        logger.info(f"Starting EO run for {num_steps} steps")

        step_results = []
        for i in range(num_steps):
            logger.info(f"EO step {i+1}/{num_steps}")
            result = self.step()
            step_results.append(result)

        run_results = {
            'num_steps': num_steps,
            'step_results': step_results,
            'final_group_empowerment': step_results[-1]['group_empowerment'] if step_results else 0,
            'agent_states': self.environment.get_all_agent_states()
        }

        logger.info("Completed EO run")
        return run_results
