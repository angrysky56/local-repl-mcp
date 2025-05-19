
# Permanent Agent System Utility Library
# This file contains common functions for working with the agent system

import os
import json
import random
from datetime import datetime, timedelta

def get_memory_contents(system):
    """Get a summary of what's in the memory agent"""
    memory = system.get_agent("memory_adv")
    if not memory:
        return {"error": "Memory agent not found"}
    
    return {
        "memory_count": len(memory.memory),
        "categories": list(memory.categories.keys()) if hasattr(memory, 'categories') else [],
        "tags": list(memory.tags.keys()) if hasattr(memory, 'tags') else [],
        "task_count": len(memory.get_tasks()),
        "notes_length": len(memory.notes) if memory.notes else 0
    }

def store_information(system, key, value, category=None, tags=None):
    """Store information in the memory agent"""
    memory = system.get_agent("memory_adv")
    if not memory:
        return {"error": "Memory agent not found"}
    
    memory.store(key, value, category, tags)
    system.save_all_agents()
    
    return {"status": "success", "key": key}

def add_task(system, agent_id, task_description, priority="medium"):
    """Add a task to an agent"""
    agent = system.get_agent(agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    task = {
        "description": task_description,
        "priority": priority,
        "status": "pending",
        "created": datetime.now().isoformat()
    }
    
    agent.add_task(task)
    system.save_all_agents()
    
    return {"status": "success", "task": task}

def get_agent_tasks(system, agent_id, status=None):
    """Get tasks for a specific agent"""
    agent = system.get_agent(agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}
    
    tasks = agent.get_tasks(status)
    
    return {"agent_id": agent_id, "tasks": tasks}

def run_agent_workflow(system, workflow_type, params):
    """Run an agent workflow with the specified parameters
    
    Args:
        system: The agent system
        workflow_type: Type of workflow to run (e.g., "research_and_document")
        params: Dictionary of parameters for the workflow
    
    Returns:
        Dictionary with workflow results
    """
    # Get the workflow agent
    workflow_agent = system.get_agent("workflow")
    if not workflow_agent:
        return {"error": "Workflow agent not found", "status": "failed"}
    
    # Prepare the workflow request
    workflow_request = {
        "type": workflow_type,
        "params": params,
        "created": datetime.now().isoformat()
    }
    
    # Execute the workflow
    try:
        result = workflow_agent.execute_workflow(workflow_request)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def run_research(system, topic):
    """Run a research workflow"""
    return run_agent_workflow(system, "research_and_document", {"topic": topic})

def test_code(system, code, test_cases=None):
    """Test code using the function tester agent"""
    return run_agent_workflow(system, "code_test_document", {"code": code, "test_cases": test_cases or []})

def save_system_state(system):
    """Save the entire system state"""
    system.save_all_agents()
    return {"status": "success", "agents_saved": len(system.agents)}
