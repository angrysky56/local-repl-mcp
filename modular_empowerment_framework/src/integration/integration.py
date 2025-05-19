"""
Integration module for the Modular Decision Framework (MDF) and Empowerment Optimization (EO).

This module provides the necessary classes and functions to bridge the two frameworks,
allowing them to work together in a unified system.
"""

import numpy as np
from typing import Dict, List, Any, Callable, Tuple, Optional
import logging

from src.mdf.core import MDFCore, Thought
from src.eo.core import EOCore, Agent, Environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModularEmpowermentIntegration:
    """
    Main integration class that combines MDF with EO techniques.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration.

        Args:
            config: Configuration parameters for the integration
        """
        # Default configuration
        self.config = {
            # Core weights for empowerment in MDF evaluation
            'individual_empowerment_weight': 0.3,
            'group_empowerment_weight': 0.3,

            # Balance between MDF and EO in decision making
            'mdf_eo_balance': 0.5,  # 0: Pure MDF, 1: Pure EO

            # Scale conversion between EO and MDF
            'empowerment_to_metric_scale_factor': 10.0  # Scale [0,1] to [0,10]
        }

        # Override defaults with provided config
        if config:
            self.config.update(config)

        # Initialize MDF
        mdf_config = config.get('mdf_config', {}) if config else {}
        self.mdf = MDFCore(mdf_config)

        # Initialize EO
        eo_config = config.get('eo_config', {}) if config else {}
        self.eo = EOCore(eo_config)

        # Add empowerment metrics to MDF metric weights
        self.mdf.config['metric_weights']['individual_empowerment'] = self.config['individual_empowerment_weight']
        self.mdf.config['metric_weights']['group_empowerment'] = self.config['group_empowerment_weight']

        # Initialize translation mappings
        self._init_translation_mappings()

        logger.info("Initialized Modular Empowerment Integration")

    def _init_translation_mappings(self):
        """Initialize the mapping functions between MDF and EO concepts."""
        self.translation = {
            'thought_to_action': self._thought_to_action,
            'action_to_thought': self._action_to_thought,
            'state_to_data': self._state_to_data,
            'data_to_state': self._data_to_state
        }

    def _thought_to_action(self, thought: Thought) -> Any:
        """
        Convert a MDF thought to an EO action.

        Args:
            thought: The thought to convert

        Returns:
            The converted action
        """
        # Simple implementation - in a real system this would be more sophisticated
        return {
            'content': thought.content,
            'context': thought.context
        }

    def _action_to_thought(self, action: Any) -> Thought:
        """
        Convert an EO action to an MDF thought.

        Args:
            action: The action to convert

        Returns:
            The converted thought
        """
        # Simple implementation - in a real system this would be more sophisticated
        if isinstance(action, dict) and 'content' in action:
            return Thought(action['content'], action.get('context', {}))
        return Thought(str(action))

    def _state_to_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an EO state to MDF data.

        Args:
            state: The state to convert

        Returns:
            The converted data
        """
        # Simple implementation - in a real system this would be more sophisticated
        return state.copy()

    def _data_to_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MDF data to an EO state.

        Args:
            data: The data to convert

        Returns:
            The converted state
        """
        # Simple implementation - in a real system this would be more sophisticated
        return data.copy()

    def add_agent(self, agent_id: str, initial_state: Dict[str, Any] = {}) -> Agent:
        """
        Add a new agent to the integrated system.

        Args:
            agent_id: Unique identifier for the agent
            initial_state: Initial state of the agent

        Returns:
            The created agent
        """
        return self.eo.add_agent(agent_id, initial_state)
    def enhance_mdf_evaluation(self, thought: Thought, agent_id: str) -> Dict[str, float]:
        """
        Enhance MDF evaluation metrics with empowerment calculations.

        Args:
            thought: Thought to evaluate
            agent_id: ID of the agent whose perspective is being used

        Returns:
            Enhanced metrics dictionary
        """
        # Get initial metrics from MDF
        metrics = self.mdf.evaluate_metrics(thought)

        # Convert thought to action
        action = self.translation['thought_to_action'](thought)

        # Calculate individual empowerment
        individual_empowerment = self.eo.calculate_individual_empowerment(agent_id, action)

        # Calculate group empowerment
        # For simplicity, we're using a placeholder with just this agent's action
        group_empowerment = self.eo.calculate_group_empowerment({agent_id: action})

        # Scale empowerment values to MDF metric scale (0-10)
        scale_factor = self.config['empowerment_to_metric_scale_factor']
        metrics['individual_empowerment'] = individual_empowerment * scale_factor
        metrics['group_empowerment'] = group_empowerment * scale_factor

        logger.debug(f"Enhanced evaluation with empowerment for thought: {thought.content}")
        return metrics

    def enhanced_multi_layered_analysis(self, thought: Thought, agent_id: str) -> Dict[str, float]:
        """
        Perform enhanced multi-layered analysis including empowerment impact.

        Args:
            thought: Thought to analyze
            agent_id: ID of the agent whose perspective is being used

        Returns:
            Enhanced analysis scores
        """
        # Get initial analysis from MDF
        scores = self.mdf.multi_layered_analysis(thought)

        # Add empowerment-based analysis
        action = self.translation['thought_to_action'](thought)

        # Calculate empowerment impact score
        empowerment_impact = self._assess_empowerment_impact(action, agent_id)
        scores['empowerment_impact'] = empowerment_impact

        # Calculate agent cooperation score
        agent_cooperation = self._assess_agent_cooperation(action)
        scores['agent_cooperation'] = agent_cooperation

        logger.debug(f"Enhanced multi-layered analysis with empowerment aspects")
        return scores

    def _assess_empowerment_impact(self, action: Any, agent_id: str) -> float:
        """
        Assess the impact of an action on empowerment.

        Args:
            action: The action to assess
            agent_id: ID of the agent

        Returns:
            Empowerment impact score (0-1)
        """
        # Placeholder implementation
        # In a real system, this would simulate short and long-term empowerment effects
        return np.random.uniform(0, 1)

    def _assess_agent_cooperation(self, action: Any) -> float:
        """
        Assess how the action affects cooperation among agents.

        Args:
            action: The action to assess

        Returns:
            Agent cooperation score (0-1)
        """
        # Placeholder implementation
        # In a real system, this would analyze resource sharing, information exchange, etc.
        return np.random.uniform(0, 1)

    def integrated_decision_process(self, inputs: Any, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete integrated decision process.

        Args:
            inputs: Raw input data
            agent_id: Optional ID of the agent's perspective to use

        Returns:
            Decision results
        """
        logger.info("Starting integrated decision process")

        # If no agent_id is provided, use the first available agent
        if agent_id is None:
            if not self.eo.environment.agents:
                logger.error("No agents available for decision process")
                return {'status': 'failure', 'reason': 'No agents available'}
            agent_id = str(next(iter(self.eo.environment.agents.keys())))

        # Module 2: Data Collection and Preprocessing
        data, user_attributes, user_interests = self.mdf.preprocess_inputs(inputs)

        # Module 3: Dilemma Classification
        dilemma_type = self.mdf.classify_dilemma(data)

        # Module 4: Thought Generation
        thoughts = self.mdf.generate_thoughts(data, user_attributes, user_interests)

        # Module 5: Enhanced Evaluation with Empowerment
        for thought in thoughts:
            thought.metrics = self.enhance_mdf_evaluation(thought, agent_id)
            thought.quality_score = self.mdf.calculate_quality_score(thought.metrics)

        # Module 6: Pruning
        thoughts = self.mdf.prune_thoughts(thoughts)

        # Module 7: Thought Selection
        best_thought = self.mdf.select_best_thought(thoughts)
        if not best_thought:
            logger.error("No viable thoughts found")
            return {'status': 'failure', 'reason': 'No viable thoughts found'}

        # Module 8: Enhanced Multi-Layered Analysis
        best_thought.scores = self.enhanced_multi_layered_analysis(best_thought, agent_id)

        # Module 9: Anomaly Detection and Handling
        anomalies = self.mdf.detect_anomalies(thoughts)
        if anomalies:
            self.mdf.handle_anomalies(anomalies)
            # Re-evaluate thoughts after handling anomalies
            for thought in thoughts:
                thought.metrics = self.enhance_mdf_evaluation(thought, agent_id)
                thought.quality_score = self.mdf.calculate_quality_score(thought.metrics)

            best_thought = self.mdf.select_best_thought(thoughts)
            if not best_thought:
                logger.error("No viable thoughts found after handling anomalies")
                return {'status': 'failure', 'reason': 'No viable thoughts after anomaly handling'}

        # Module 10: Self-Reflection
        best_thought.self_reflection_score = self.mdf.self_reflection(best_thought)

        # Module 11: Decision Integration and Execution
        decision = self.mdf.integrate_scores_and_decide(best_thought)

        # Convert thought to action and execute in EO environment
        action = self.translation['thought_to_action'](best_thought)
        next_state = self.eo.environment.execute_action(agent_id, action)

        # Calculate empowerment for feedback
        individual_empowerment = self.eo.calculate_individual_empowerment(agent_id, action)
        group_empowerment = self.eo.calculate_group_empowerment({agent_id: action})

        # Calculate reward
        reward = self.eo.calculate_reward(agent_id, individual_empowerment, group_empowerment)

        # Store experience in EO
        agent = self.eo.environment.get_agent(agent_id)
        if agent is None:
            logger.error(f"Agent {agent_id} not found")
            return {'status': 'failure', 'reason': f'Agent {agent_id} not found'}

        self.eo.store_experience(
            agent_id,
            agent.get_state(),
            action,
            reward,
            next_state
        )

        # Update EO components
        self.eo.update_policies()
        self.eo.update_model()

        # Module 12: Outcome and Feedback
        execution_results = {
            'executed': True,
            'status': 'success',
            'details': f"Executed: {decision['thought']}",
            'next_state': next_state,
            'empowerment': individual_empowerment,
            'group_empowerment': group_empowerment,
            'reward': reward
        }

        outcome = self.mdf.gather_outcome(execution_results)
        feedback = self.mdf.collect_feedback(outcome)

        # Module 13: Framework Adaptation
        self.mdf.adapt_framework(feedback)

        results = {
            'status': 'success',
            'decision': decision,
            'execution_results': execution_results,
            'outcome': outcome,
            'feedback': feedback,
            'empowerment_metrics': {
                'individual': individual_empowerment,
                'group': group_empowerment,
                'reward': reward
            }
        }

        logger.info("Completed integrated decision process")
        return results
    def run_simulation(self, num_steps: int = 10, inputs: Any = None) -> Dict[str, Any]:
        """
        Run a multi-step simulation of the integrated system.

        Args:
            num_steps: Number of steps to run
            inputs: Initial inputs for the simulation

        Returns:
            Results of the simulation
        """
        logger.info(f"Starting integrated simulation for {num_steps} steps")

        if not self.eo.environment.agents:
            logger.error("No agents available for simulation")
            return {'status': 'failure', 'reason': 'No agents available'}

        results = []
        current_inputs = inputs

        for i in range(num_steps):
            logger.info(f"Simulation step {i+1}/{num_steps}")

            for agent_id in self.eo.environment.agents:
                # Run decision process for each agent
                result = self.integrated_decision_process(current_inputs, agent_id)
                results.append({
                    'step': i+1,
                    'agent_id': agent_id,
                    'result': result
                })

                # Update inputs based on new state
                if result['status'] == 'success':
                    next_state = result['execution_results']['next_state']
                    current_inputs = self._state_to_data(next_state)

        simulation_results = {
            'status': 'success',
            'num_steps': num_steps,
            'step_results': results,
            'final_agent_states': self.eo.environment.get_all_agent_states(),
            'summary': self._generate_simulation_summary(results)
        }

        logger.info("Completed integrated simulation")
        return simulation_results

    def _generate_simulation_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of simulation results.

        Args:
            results: List of step results

        Returns:
            Summary dictionary
        """
        # Extract relevant metrics
        empowerment_values = []
        group_empowerment_values = []
        quality_scores = []

        for step_result in results:
            if step_result['result']['status'] == 'success':
                empowerment_values.append(step_result['result']['empowerment_metrics']['individual'])
                group_empowerment_values.append(step_result['result']['empowerment_metrics']['group'])

                # Extract quality score if available
                decision = step_result['result'].get('decision', {})
                if 'thought' in decision and 'quality_score' in decision:
                    quality_scores.append(decision['quality_score'])

        # Calculate summary statistics
        summary = {
            'avg_empowerment': np.mean(empowerment_values) if empowerment_values else 0,
            'max_empowerment': np.max(empowerment_values) if empowerment_values else 0,
            'avg_group_empowerment': np.mean(group_empowerment_values) if group_empowerment_values else 0,
            'max_group_empowerment': np.max(group_empowerment_values) if group_empowerment_values else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'empowerment_trend': 'increasing' if len(empowerment_values) > 1 and
                                 empowerment_values[-1] > empowerment_values[0] else 'stable_or_decreasing'
        }

        return summary
