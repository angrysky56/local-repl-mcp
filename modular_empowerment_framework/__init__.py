"""
Modular Empowerment Framework (MEF)

This package provides a framework for creating and managing intelligent agent systems.
"""

__version__ = "0.1.0"

# Import key classes from the framework
from modular_empowerment_framework.src import *

# Define the main agent system components
class AgentSystem:
    """Agent system that manages multiple agents"""
    
    def __init__(self, data_dir=None):
        self.agents = {}
        self.data_dir = data_dir
    
    def add_agent(self, agent):
        """Add an agent to the system"""
        self.agents[agent.agent_id] = agent
        return True
    
    def get_agent(self, agent_id):
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def save_all_agents(self):
        """Save all agents to disk"""
        for agent in self.agents.values():
            if hasattr(agent, 'save'):
                agent.save()
        return True
    
    def get_agent_summary(self):
        """Get a summary of all agents in the system"""
        summaries = []
        for agent_id, agent in self.agents.items():
            summary = {
                "id": agent_id,
                "type": agent.__class__.__name__,
                "energy": getattr(agent, 'energy', 1.0),
                "empowerment": getattr(agent, 'empowerment', 1.0)
            }
            summaries.append(summary)
        return summaries

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, agent_id, name=None, description=None):
        self.agent_id = agent_id
        self.name = name or agent_id
        self.description = description or ""
        self.tasks = []
        self.energy = 1.0
        self.empowerment = 1.0
    
    def add_task(self, task):
        """Add a task to this agent"""
        self.tasks.append(task)
        return True
    
    def get_tasks(self, status=None):
        """Get tasks with an optional status filter"""
        if status is None:
            return self.tasks
        return [task for task in self.tasks if task.get('status') == status]
    
    def save(self):
        """Save agent state (to be implemented by subclasses)"""
        pass

class MemoryAgent(BaseAgent):
    """Agent that stores and retrieves information"""
    
    def __init__(self, agent_id, name=None, description=None):
        super().__init__(agent_id, name, description)
        self.memory = {}
        self.categories = {}
        self.tags = {}
        self.notes = ""
    
    def store(self, key, value, category=None, tags=None):
        """Store information in memory"""
        self.memory[key] = value
        
        if category:
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(key)
        
        if tags:
            for tag in tags:
                if tag not in self.tags:
                    self.tags[tag] = []
                self.tags[tag].append(key)
        
        return True

class WorkflowAgent(BaseAgent):
    """Agent that manages workflows"""
    
    def __init__(self, agent_id, name=None, description=None):
        super().__init__(agent_id, name, description)
        self.workflows = {}
    
    def execute_workflow(self, workflow_request):
        """Execute a workflow"""
        workflow_type = workflow_request.get('type')
        params = workflow_request.get('params', {})
        
        # Store the workflow request
        if workflow_type not in self.workflows:
            self.workflows[workflow_type] = []
        self.workflows[workflow_type].append(workflow_request)
        
        # For now, just return a placeholder result
        return {
            "status": "completed",
            "workflow_type": workflow_type,
            "results": f"Executed {workflow_type} workflow with parameters {params}"
        }

class FunctionAgent(BaseAgent):
    """Agent that tests and evaluates code functions"""
    
    def __init__(self, agent_id, name=None, description=None):
        super().__init__(agent_id, name, description)
        self.functions = {}

class ResearchAgent(BaseAgent):
    """Agent that conducts research on topics"""
    
    def __init__(self, agent_id, name=None, description=None):
        super().__init__(agent_id, name, description)
        self.research_topics = {}

# Define the key classes as exports
__all__ = [
    'AgentSystem',
    'BaseAgent',
    'MemoryAgent',
    'WorkflowAgent',
    'FunctionAgent',
    'ResearchAgent'
]
