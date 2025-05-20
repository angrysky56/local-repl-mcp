"""
Modular Empowerment Framework prompt template.

This module provides the template for the Modular Empowerment Framework workflow prompt.
"""

def full_empowerment_workflow() -> str:
    """
    A prompt template showing how to use the Modular Empowerment Framework with Persistent Agents.
    """
    return """
    # Enhanced Modular Empowerment Framework with Persistent Agents

    This workflow demonstrates how to use the Modular Empowerment Framework (MEF) with enhanced agents
    that can handle complex data types and persist across sessions. This creates a powerful multi-agent
    system that maintains state and knowledge over time, ideal for long-term projects.

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

    ## Step 3: Create enhanced agents with complex data support

    Create a set of enhanced agents that can handle complex data while remaining compatible with the MEF framework:

    ```python
    import random
    import json
    import os
    import numpy as np
    from datetime import datetime, timedelta

    class AdvancedAgent:
        \"\"\"
        Enhanced agent class that can handle complex data types while still
        interfacing with the MEF framework.
        \"\"\"
        def __init__(self, agent_id, agent_type, capabilities, initial_energy=1.0):
            self.agent_id = agent_id
            self.agent_type = agent_type
            self.capabilities = capabilities
            self.energy = initial_energy
            self.empowerment = 0.5

            # Complex data containers
            self.memory = {}            # For storing arbitrary key-value pairs
            self.knowledge_base = []    # For storing learned information
            self.history = []           # For tracking task execution history
            self.relationships = {}     # For tracking interactions with other agents
            self.notes = ""             # For storing text notes
            self.tasks_todo = []        # For tracking upcoming tasks

            # Framework integration timestamp
            self.last_framework_sync = datetime.now()

        def __str__(self):
            return f"{self.agent_type} Agent ({self.agent_id})"

        def get_state_dict(self):
            \"\"\"
            Convert the agent state to a dictionary suitable for MEF framework.
            Only includes numeric values to avoid array conversion issues.
            \"\"\"
            # Base numeric state
            state_dict = {
                'energy': self.energy,
                'empowerment': self.empowerment,
                **{k: v for k, v in self.capabilities.items() if isinstance(v, (int, float))}
            }

            # Add some numeric metadata from complex data
            state_dict['memory_size'] = len(self.memory)
            state_dict['knowledge_size'] = len(self.knowledge_base)
            state_dict['history_size'] = len(self.history)
            state_dict['relationship_count'] = len(self.relationships)
            state_dict['notes_length'] = len(self.notes) if self.notes else 0
            state_dict['task_count'] = len(self.tasks_todo)

            # Store last update timestamp as a number (seconds since epoch)
            self.last_framework_sync = datetime.now()
            state_dict['last_sync'] = self.last_framework_sync.timestamp()

            return state_dict

        def add_memory(self, key, value):
            \"\"\"Add or update an item in the agent's memory\"\"\"
            self.memory[key] = value
            return True

        def get_memory(self, key, default=None):
            \"\"\"Retrieve an item from the agent's memory\"\"\"
            return self.memory.get(key, default)

        def add_knowledge(self, knowledge_item):
            \"\"\"Add a new knowledge item to the knowledge base\"\"\"
            if knowledge_item not in self.knowledge_base:
                self.knowledge_base.append(knowledge_item)
                return True
            return False

        def search_knowledge(self, query):
            \"\"\"Search the knowledge base for items matching the query\"\"\"
            if not query:
                return []

            # Simple text-based search
            return [item for item in self.knowledge_base
                    if str(query).lower() in str(item).lower()]

        def add_note(self, note):
            \"\"\"Append a note to the agent's notes\"\"\"
            separator = "\\n" if self.notes else ""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.notes += f"{separator}[{timestamp}] {note}"
            return True

        def get_notes(self):
            \"\"\"Get the agent's notes\"\"\"
            return self.notes

        def add_task(self, task):
            \"\"\"Add a task to the agent's todo list\"\"\"
            if isinstance(task, dict) and 'description' in task:
                if task not in self.tasks_todo:
                    self.tasks_todo.append(task)
                    return True
            elif isinstance(task, str):
                # Convert string task to a task object
                task_obj = {
                    'description': task,
                    'created': datetime.now().isoformat(),
                    'priority': 'medium',
                    'status': 'pending'
                }
                self.tasks_todo.append(task_obj)
                return True
            return False

        def complete_task(self, task_index):
            \"\"\"Mark a task as completed\"\"\"
            if 0 <= task_index < len(self.tasks_todo):
                if isinstance(self.tasks_todo[task_index], dict):
                    self.tasks_todo[task_index]['status'] = 'completed'
                    self.tasks_todo[task_index]['completed'] = datetime.now().isoformat()
                else:
                    # Convert to dict if it wasn't already
                    task = self.tasks_todo[task_index]
                    self.tasks_todo[task_index] = {
                        'description': task,
                        'created': datetime.now().isoformat(),
                        'completed': datetime.now().isoformat(),
                        'status': 'completed'
                    }
                return True
            return False

        def get_tasks(self, status=None):
            \"\"\"Get tasks, optionally filtered by status\"\"\"
            if status:
                return [t for t in self.tasks_todo
                        if isinstance(t, dict) and t.get('status') == status]
            return self.tasks_todo

        def add_relationship(self, agent_id, relationship_type, strength=0.5, notes=""):
            \"\"\"Add or update a relationship with another agent\"\"\"
            self.relationships[agent_id] = {
                'type': relationship_type,
                'strength': strength,
                'notes': notes,
                'last_updated': datetime.now().isoformat()
            }
            return True

        def execute_task(self, task, **kwargs):
            \"\"\"
            Execute a task and store the results in history.
            This is a high-level method that can be overridden by specific agent types.
            \"\"\"
            # Record the task start
            task_record = {
                'task': task,
                'start_time': datetime.now().isoformat(),
                'parameters': kwargs,
                'status': 'in_progress'
            }

            # Simulate task execution
            success_chance = min(0.7, self.energy * self.empowerment)
            success = random.random() < success_chance
            quality = random.uniform(0.3, 0.9) * self.energy * self.empowerment

            # Update the task record
            task_record['end_time'] = datetime.now().isoformat()
            task_record['success'] = success
            task_record['quality'] = quality
            task_record['status'] = 'completed'

            # Update agent state
            self.energy = max(0.1, self.energy - random.uniform(0.1, 0.3))
            if success:
                self.empowerment = min(1.0, self.empowerment + random.uniform(0.01, 0.05))
            else:
                self.empowerment = max(0.1, self.empowerment - random.uniform(0.01, 0.03))

            # Store in history
            self.history.append(task_record)

            return {
                'success': success,
                'quality': quality,
                'energy_remaining': self.energy,
                'empowerment': self.empowerment
            }

        def save_to_disk(self, directory="agent_data"):
            \"\"\"Save the agent's state to disk for persistence\"\"\"
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Create a file path based on agent ID
            file_path = os.path.join(directory, f"{self.agent_id}.json")

            # Prepare data for saving (convert to JSON serializable format)
            data = {
                'agent_id': self.agent_id,
                'agent_type': self.agent_type,
                'capabilities': self.capabilities,
                'energy': self.energy,
                'empowerment': self.empowerment,
                'memory': self.memory,
                'knowledge_base': self.knowledge_base,
                'history': self.history,
                'relationships': self.relationships,
                'notes': self.notes,
                'tasks_todo': self.tasks_todo,
                'last_saved': datetime.now().isoformat()
            }

            # Save to disk
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            return file_path

    # Create different specialized agent types
    class ResearcherAgent(AdvancedAgent):
        \"\"\"Agent specialized in gathering and analyzing information\"\"\"

        def __init__(self, agent_id, capabilities=None, initial_energy=1.0):
            if capabilities is None:
                capabilities = {
                    'information_gathering': 0.9,
                    'analysis': 0.8,
                    'critical_thinking': 0.9,
                    'communication': 0.7,
                    'knowledge': 0.8
                }
            super().__init__(agent_id, "researcher", capabilities, initial_energy)

            # Researcher-specific properties
            self.research_topics = []
            self.findings = []

        def add_research_topic(self, topic, importance=0.5):
            \"\"\"Add a research topic\"\"\"
            self.research_topics.append({
                'topic': topic,
                'importance': importance,
                'status': 'new',
                'created': datetime.now().isoformat()
            })
            return True

        def add_finding(self, topic, finding):
            \"\"\"Add a research finding\"\"\"
            self.findings.append({
                'topic': topic,
                'finding': finding,
                'timestamp': datetime.now().isoformat()
            })
            # Also add to knowledge base
            self.add_knowledge(f"Finding about {topic}: {finding}")
            return True

        def execute_task(self, task, **kwargs):
            \"\"\"Override to add researcher-specific behavior\"\"\"
            result = super().execute_task(task, **kwargs)

            # Add researcher-specific logic
            if task == "research" and "topic" in kwargs:
                topic = kwargs["topic"]
                if result['success']:
                    # Generate a simulated finding
                    finding = f"Research finding on {topic} with quality {result['quality']:.2f}"
                    self.add_finding(topic, finding)
                    result['finding'] = finding

            return result

    class MemoryAgent(AdvancedAgent):
        \"\"\"Specialized agent focused on storing and retrieving information\"\"\"

        def __init__(self, agent_id, capabilities=None, initial_energy=1.0):
            if capabilities is None:
                capabilities = {
                    'memory': 0.95,
                    'organization': 0.9,
                    'retrieval': 0.9,
                    'storage': 0.95,
                    'context_awareness': 0.8
                }
            super().__init__(agent_id, "memory", capabilities, initial_energy)

            # Memory-specific properties
            self.categories = {}  # Categorized memories
            self.tags = {}        # Tagged memories
            self.associations = {}  # Associative memory links

        def store(self, key, value, category=None, tags=None):
            \"\"\"Store data with optional category and tags\"\"\"
            # Store in main memory
            self.add_memory(key, value)

            # Add to category if specified
            if category:
                if category not in self.categories:
                    self.categories[category] = {}
                self.categories[category][key] = value

            # Add tags if specified
            if tags:
                for tag in tags:
                    if tag not in self.tags:
                        self.tags[tag] = {}
                    self.tags[tag][key] = value

            return True

        def retrieve_by_category(self, category):
            \"\"\"Retrieve all items in a category\"\"\"
            return self.categories.get(category, {})

        def retrieve_by_tag(self, tag):
            \"\"\"Retrieve all items with a specific tag\"\"\"
            return self.tags.get(tag, {})

    class ProjectManagerAgent(AdvancedAgent):
        \"\"\"Agent specialized in managing projects and coordinating tasks\"\"\"

        def __init__(self, agent_id, capabilities=None, initial_energy=1.0):
            if capabilities is None:
                capabilities = {
                    'organization': 0.9,
                    'planning': 0.9,
                    'delegation': 0.8,
                    'estimation': 0.7,
                    'tracking': 0.8
                }
            super().__init__(agent_id, "project_manager", capabilities, initial_energy)

            # Project manager specific properties
            self.projects = []
            self.assigned_tasks = {}  # agent_id -> [tasks]
            self.project_timelines = {}  # project_id -> timeline

        def create_project(self, name, description, deadline=None):
            \"\"\"Create a new project\"\"\"
            project_id = f"proj_{len(self.projects) + 1}"

            project = {
                'id': project_id,
                'name': name,
                'description': description,
                'status': 'new',
                'created': datetime.now().isoformat(),
                'deadline': deadline,
                'tasks': [],
                'team': []
            }

            self.projects.append(project)
            return project_id

        def add_project_task(self, project_id, task_description, assigned_to=None, due_date=None):
            \"\"\"Add a task to a project\"\"\"
            # Find the project
            project = next((p for p in self.projects if p['id'] == project_id), None)

            if not project:
                return False

            task_id = f"task_{len(project['tasks']) + 1}"

            task = {
                'id': task_id,
                'description': task_description,
                'status': 'pending',
                'created': datetime.now().isoformat(),
                'assigned_to': assigned_to,
                'due_date': due_date
            }

            project['tasks'].append(task)

            # Track the assignment
            if assigned_to:
                if assigned_to not in self.assigned_tasks:
                    self.assigned_tasks[assigned_to] = []
                self.assigned_tasks[assigned_to].append({
                    'project_id': project_id,
                    'task_id': task_id,
                    'description': task_description
                })

            return task_id

        def update_task_status(self, project_id, task_id, status):
            \"\"\"Update a task's status\"\"\"
            # Find the project
            project = next((p for p in self.projects if p['id'] == project_id), None)

            if not project:
                return False

            # Find the task
            task = next((t for t in project['tasks'] if t['id'] == task_id), None)

            if not task:
                return False

            # Update status
            task['status'] = status
            task['updated'] = datetime.now().isoformat()

            return True
    ```

    ## Step 4: Create a Permanent Agent System for persistence

    Create a system to manage agents that persist across sessions:

    ```python
    class PermanentAgentSystem:
        \"\"\"
        A system for managing permanent agents that persist across sessions.
        \"\"\"
        def __init__(self, storage_dir="agent_data"):
            self.storage_dir = storage_dir
            self.agents = {}
            self.mef_integrated = False
            os.makedirs(storage_dir, exist_ok=True)

        def initialize(self, mef=None):
            \"\"\"Initialize the system and load any saved agents\"\"\"
            print(f"Initializing Permanent Agent System from {self.storage_dir}")

            # Load all saved agents
            self._load_all_agents()

            # Integrate with MEF if provided
            if mef:
                self.integrate_with_mef(mef)

            return self

        def _load_all_agents(self):
            \"\"\"Load all agents from the storage directory\"\"\"
            self.agents = {}

            # List all JSON files in the storage directory
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    agent_id = filename[:-5]  # Remove .json extension
                    agent = self.load_agent_from_disk(agent_id)
                    if agent:
                        self.agents[agent_id] = agent
                        print(f"Loaded agent: {agent}")

        def load_agent_from_disk(self, agent_id):
            \"\"\"Load an agent from disk by ID\"\"\"
            file_path = os.path.join(self.storage_dir, f"{agent_id}.json")

            if not os.path.exists(file_path):
                return None

            try:
                # Load data from disk
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Determine which class to instantiate based on agent_type
                agent_type = data['agent_type']
                if agent_type == 'researcher':
                    agent = ResearcherAgent(
                        agent_id=data['agent_id'],
                        capabilities=data['capabilities'],
                        initial_energy=data['energy']
                    )
                    # Restore researcher-specific properties
                    if 'research_topics' in data:
                        agent.research_topics = data.get('research_topics', [])
                    if 'findings' in data:
                        agent.findings = data.get('findings', [])
                elif agent_type == 'memory':
                    agent = MemoryAgent(
                        agent_id=data['agent_id'],
                        capabilities=data['capabilities'],
                        initial_energy=data['energy']
                    )
                    # Restore memory-specific properties
                    if 'categories' in data:
                        agent.categories = data.get('categories', {})
                    if 'tags' in data:
                        agent.tags = data.get('tags', {})
                    if 'associations' in data:
                        agent.associations = data.get('associations', {})
                elif agent_type == 'project_manager':
                    agent = ProjectManagerAgent(
                        agent_id=data['agent_id'],
                        capabilities=data['capabilities'],
                        initial_energy=data['energy']
                    )
                    # Restore project manager-specific properties
                    if 'projects' in data:
                        agent.projects = data.get('projects', [])
                    if 'assigned_tasks' in data:
                        agent.assigned_tasks = data.get('assigned_tasks', {})
                    if 'project_timelines' in data:
                        agent.project_timelines = data.get('project_timelines', {})
                else:
                    # Generic agent
                    agent = AdvancedAgent(
                        agent_id=data['agent_id'],
                        agent_type=data['agent_type'],
                        capabilities=data['capabilities'],
                        initial_energy=data['energy']
                    )

                # Restore common properties
                agent.empowerment = data['empowerment']
                agent.memory = data['memory']
                agent.knowledge_base = data['knowledge_base']
                agent.history = data['history']
                agent.relationships = data['relationships']
                agent.notes = data['notes']
                agent.tasks_todo = data['tasks_todo']

                # Type-specific properties
                if agent_type == 'researcher' and hasattr(agent, 'research_topics') and 'research_topics' in data:
                    agent.research_topics = data.get('research_topics', [])
                if agent_type == 'researcher' and hasattr(agent, 'findings') and 'findings' in data:
                    agent.findings = data.get('findings', [])
                if agent_type == 'memory' and hasattr(agent, 'categories') and 'categories' in data:
                    agent.categories = data.get('categories', {})
                if agent_type == 'memory' and hasattr(agent, 'tags') and 'tags' in data:
                    agent.tags = data.get('tags', {})
                if agent_type == 'memory' and hasattr(agent, 'associations') and 'associations' in data:
                    agent.associations = data.get('associations', {})
                if agent_type == 'project_manager' and hasattr(agent, 'projects') and 'projects' in data:
                    agent.projects = data.get('projects', [])
                if agent_type == 'project_manager' and hasattr(agent, 'assigned_tasks') and 'assigned_tasks' in data:
                    agent.assigned_tasks = data.get('assigned_tasks', {})
                if agent_type == 'project_manager' and hasattr(agent, 'project_timelines') and 'project_timelines' in data:
                    agent.project_timelines = data.get('project_timelines', {})

                return agent
            except Exception as e:
                print(f"Error loading agent from disk: {e}")
                return None

        def integrate_with_mef(self, mef):
            \"\"\"Integrate all agents with the MEF framework\"\"\"
            self.mef_integrated = True

            for agent_id, agent in self.agents.items():
                # Register the agent with MEF
                mef.add_agent(agent_id, agent.get_state_dict())
                print(f"Registered agent {agent} with MEF")

        def get_agent(self, agent_id):
            \"\"\"Get an agent by ID\"\"\"
            return self.agents.get(agent_id)

        def create_agent(self, agent_type, agent_id=None, capabilities=None, **kwargs):
            \"\"\"Create a new agent of the specified type\"\"\"
            if agent_id is None:
                # Generate a unique ID if not provided
                agent_id = f"{agent_type}_{len(self.agents) + 1}"

            # Check if an agent with this ID already exists
            if agent_id in self.agents:
                print(f"Agent with ID {agent_id} already exists")
                return self.agents[agent_id]

            # Create the agent based on type
            if agent_type == "researcher":
                agent = ResearcherAgent(agent_id, capabilities, **kwargs)
            elif agent_type == "memory":
                agent = MemoryAgent(agent_id, capabilities, **kwargs)
            elif agent_type == "project_manager":
                agent = ProjectManagerAgent(agent_id, capabilities, **kwargs)
            else:
                # Generic agent
                agent = AdvancedAgent(agent_id, agent_type, capabilities or {}, **kwargs)

            # Add to our collection
            self.agents[agent_id] = agent

            # Save to disk
            agent.save_to_disk(self.storage_dir)

            print(f"Created new {agent_type} agent: {agent}")
            return agent

        def save_all_agents(self):
            \"\"\"Save all agents to disk\"\"\"
            for agent_id, agent in self.agents.items():
                agent.save_to_disk(self.storage_dir)
            print(f"Saved {len(self.agents)} agents to disk")

        def delete_agent(self, agent_id):
            \"\"\"Delete an agent\"\"\"
            if agent_id in self.agents:
                # Remove from our collection
                del self.agents[agent_id]

                # Remove from disk
                file_path = os.path.join(self.storage_dir, f"{agent_id}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)

                print(f"Deleted agent {agent_id}")
                return True
            return False

        def get_agent_summary(self):
            \"\"\"Get a summary of all agents\"\"\"
            summary = []

            for agent_id, agent in self.agents.items():
                agent_summary = {
                    'id': agent_id,
                    'type': agent.agent_type,
                    'energy': agent.energy,
                    'empowerment': agent.empowerment,
                    'memory_count': len(agent.memory),
                    'task_count': len(agent.get_tasks()),
                    'knowledge_count': len(agent.knowledge_base)
                }
                summary.append(agent_summary)

            return summary

    # Initialize the permanent agent system
    permanent_agents = PermanentAgentSystem().initialize(mef)

    # Create core agents if they don't exist yet
    if "memory_adv" not in permanent_agents.agents:
        memory_agent = permanent_agents.create_agent(
            agent_type="memory",
            agent_id="memory_adv",
            capabilities={
                'memory': 0.95,
                'organization': 0.9,
                'retrieval': 0.9,
                'storage': 0.95,
                'context_awareness': 0.8
            }
        )
    else:
        memory_agent = permanent_agents.get_agent("memory_adv")

    if "researcher_adv" not in permanent_agents.agents:
        researcher = permanent_agents.create_agent(
            agent_type="researcher",
            agent_id="researcher_adv",
            capabilities={
                'information_gathering': 0.9,
                'analysis': 0.8,
                'critical_thinking': 0.9,
                'communication': 0.7,
                'knowledge': 0.8
            }
        )
    else:
        researcher = permanent_agents.get_agent("researcher_adv")

    print(f"Permanent Agent System active with {len(permanent_agents.agents)} agents")
    ```

    ## Step 5: Store and retrieve complex data with the Memory Agent

    The Memory Agent can store any type of data, including structured information:

    ```python
    # Let's test the memory agent's capabilities
    memory_agent = permanent_agents.get_agent("memory_adv")

    # Store some information in different categories and with tags
    memory_agent.store(
        key="project_idea_1",
        value="Create an AI-powered personal assistant for task management",
        category="project_ideas",
        tags=["ai", "productivity", "future"]
    )

    memory_agent.store(
        key="important_code_snippet",
        value=\"\"\"def preprocess_data(data):
        # Clean and normalize data
        cleaned = [x.strip().lower() for x in data if x]
        return cleaned\"\"\",
        category="code_snippets",
        tags=["python", "data", "utility"]
    )

    memory_agent.store(
        key="meeting_note_1",
        value="Meeting with AI team: Discussed using transformer models for text generation",
        category="meeting_notes",
        tags=["ai", "meeting", "transformers"]
    )

    # Add some tasks
    memory_agent.add_task("Implement persistent storage for all agents")
    memory_agent.add_task("Create visualization for agent relationships")
    memory_agent.add_task({
        "description": "Test interagent communication framework",
        "priority": "high",
        "due_date": "2025-05-25"
    })

    # Add some notes
    memory_agent.add_note("The memory agent is working as expected. I should extend it to handle larger datasets.")
    memory_agent.add_note("Consider implementing fuzzy search for memory retrieval to handle approximate matching.")

    # Now retrieve the stored information
    print("=== Memory Agent Contents ===")
    print(f"Total memories: {len(memory_agent.memory)}")
    print(f"Categories: {list(memory_agent.categories.keys())}")
    print(f"Tags: {list(memory_agent.tags.keys())}")
    print(f"Tasks: {len(memory_agent.get_tasks())} pending")
    print(f"Notes length: {len(memory_agent.notes)} characters")

    # Retrieve information by category
    project_ideas = memory_agent.retrieve_by_category("project_ideas")
    print("Project Ideas:")
    for key, value in project_ideas.items():
        print(f"- {key}: {value}")

    # Retrieve by tag
    ai_related = memory_agent.retrieve_by_tag("ai")
    print("AI-related items:")
    for key, value in ai_related.items():
        print(f"- {key}: {value[:50]}..." if len(value) > 50 else f"- {key}: {value}")

    # Save to disk for persistence
    memory_agent.save_to_disk()
    ```

    ## Step 6: Use the Researcher Agent

    The Researcher Agent can perform research tasks and store findings:

    ```python
    # Get the researcher agent
    researcher = permanent_agents.get_agent("researcher_adv")

    # Add research topics
    researcher.add_research_topic("Neural networks for time series prediction", 0.8)
    researcher.add_research_topic("Quantum computing algorithms", 0.9)
    researcher.add_research_topic("Empowerment optimization in multi-agent systems", 0.7)

    # Execute research tasks
    result1 = researcher.execute_task("research", topic="Neural networks for time series prediction")
    result2 = researcher.execute_task("research", topic="Quantum computing algorithms")

    # Add a direct finding
    researcher.add_finding(
        topic="Empowerment optimization",
        finding="Recent papers show that using a transformer-based architecture improves agent cooperation in multi-agent systems."
    )

    # View researcher state
    print("=== Researcher Agent Status ===")
    print(f"Energy level: {researcher.energy:.2f}")
    print(f"Empowerment: {researcher.empowerment:.2f}")
    print(f"Research topics: {len(researcher.research_topics)}")
    print(f"Findings: {len(researcher.findings)}")
    print(f"Knowledge base entries: {len(researcher.knowledge_base)}")

    # Check the findings
    print("Research Findings:")
    for finding in researcher.findings:
        print(f"- Topic: {finding['topic']}")
        print(f"  Finding: {finding['finding']}")

    # Save to disk for persistence
    researcher.save_to_disk()
    ```

    ## Step 7: Implement Multi-Agent Workflows

    Create workflows that involve multiple agents working together:

    ```python
    # Define a workflow function
    def run_research_and_document(permanent_agents, topic):
        \"\"\"
        Run a workflow where a researcher investigates a topic and a memory agent stores the findings.

        Args:
            permanent_agents: The permanent agent system
            topic: Topic to research

        Returns:
            Dict with workflow results
        \"\"\"
        print(f"=== Running Research Workflow: {topic} ===")

        # Get the agents
        researcher = permanent_agents.get_agent("researcher_adv")
        memory = permanent_agents.get_agent("memory_adv")

        if not researcher or not memory:
            return {"error": "Required agents not found"}

        # Step 1: Researcher studies the topic
        research_result = researcher.execute_task("research", topic=topic)

        # Add a specific finding (simulated)
        finding = f"Research finding on {topic}: LSTM networks show promising results for time series with long-term dependencies."

        researcher.add_finding(topic, finding)

        # Step 2: Store the finding in memory
        memory.store(
            key=f"research_{len(memory.memory)}",
            value=finding,
            category="research_findings",
            tags=["research", topic.split()[0].lower()]
        )

        # Save all agents to persist changes
        permanent_agents.save_all_agents()

        return {
            "workflow": "research_and_document",
            "topic": topic,
            "research_success": research_result['success'],
            "finding": finding,
            "stored_in_memory": True
        }

    # Run the workflow
    result = run_research_and_document(
        permanent_agents,
        "Neural networks for time series forecasting"
    )

    print("Workflow Result:")
    for key, value in result.items():
        print(f"- {key}: {value}")
    ```

    ## Step 8: Test Agent Persistence

    Verify that agents maintain their state across sessions:

    ```python
    # First save all agents
    permanent_agents.save_all_agents()

    # Simulate starting a new session by clearing variables and reloading
    # In a real environment, you would exit the REPL and start a new one
    print("=== Simulating a new session ===")
    print("Clearing variables and reloading agents from disk...")

    # Reload the agent system
    reloaded_agents = PermanentAgentSystem().initialize(mef)

    # Check what was loaded
    print(f"Reloaded {len(reloaded_agents.agents)} agents from disk")
    for summary in reloaded_agents.get_agent_summary():
        print(f"- {summary['id']} ({summary['type']}): Memory={summary['memory_count']}, Tasks={summary['task_count']}")

    # Verify memory agent data
    memory_agent = reloaded_agents.get_agent("memory_adv")
    if memory_agent:
        research_findings = memory_agent.retrieve_by_category("research_findings")
        print(f"Retrieved {len(research_findings)} research findings from memory agent")

        for key, value in research_findings.items():
            print(f"- {key}: {value}")
    ```

    ## Step 9: Add a New Agent Type

    Create a custom agent type for specific tasks:

    ```python
    # Create a function tester agent
    function_tester = permanent_agents.create_agent(
        agent_type="function_tester",
        agent_id="function_tester_1",
        capabilities={
            'code_execution': 0.9,
            'debugging': 0.8,
            'error_detection': 0.9,
            'test_generation': 0.8,
            'code_analysis': 0.7
        }
    )

    # Add a method for testing functions
    def test_function(self, func_code, test_cases=None):
        \"\"\"Test a function with provided test cases or generate some\"\"\"
        self.add_memory(f"func_test_{len(self.memory)}", func_code)

        # Record the testing activity
        test_record = {
            'function_code': func_code,
            'test_cases': test_cases or [],
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }

        # Simulate function testing
        success_chance = self.capabilities.get('code_execution', 0.5)
        success = random.random() < success_chance

        # Generate a result
        if success:
            result = "Function tests completed successfully"
            test_record['status'] = 'passed'
        else:
            result = "Function tests failed"
            test_record['status'] = 'failed'

        # Add test record to history
        self.history.append(test_record)

        # Update energy and empowerment
        self.energy = max(0.1, self.energy - 0.2)
        self.empowerment += 0.02 if success else -0.01

        return {
            'success': success,
            'result': result
        }

    # Add the method to the agent
    function_tester.test_function = test_function.__get__(function_tester)

    # Test it out
    test_result = function_tester.test_function(\"\"\"
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n-1)
    \"\"\",
    test_cases=[
        {'input': 5, 'expected': 120},
        {'input': 0, 'expected': 1}
    ])

    print("=== Function Tester Results ===")
    print(f"Test success: {test_result['success']}")
    print(f"Result: {test_result['result']}")

    # Save the agent
    function_tester.save_to_disk()
    ```

    ## Step 10: Clean Up and Persist State

    ```python
    # Save all agents to ensure persistence
    permanent_agents.save_all_agents()
    print("All agents saved to disk for persistence")

    # Create a startup script for future sessions
    startup_script = \"\"\"
    # Permanent Agent System Startup Script
    # This script loads any existing agents or creates them if they don't exist

    import os
    import json
    import random
    from datetime import datetime, timedelta

    # Create or load the permanent agent system
    permanent_agents = PermanentAgentSystem().initialize(mef)

    print(f"Permanent Agent System initialized with {len(permanent_agents.agents)} agents")
    for summary in permanent_agents.get_agent_summary():
        print(f"- {summary['id']} ({summary['type']}): Energy={summary['energy']:.2f}, Empowerment={summary['empowerment']:.2f}")
    \"\"\"

    # Print information about using the system
    print("=== Permanent Agent System ===")
    print("This system allows you to maintain agents across sessions with these features:")
    print("1. Agents can store and process complex data (not just numbers)")
    print("2. Agents persist their state to disk and can be loaded in future sessions")
    print("3. Each agent type has specialized capabilities for different tasks")
    print("4. The system integrates with the MEF for empowerment optimization")
    print("5. Workflows can involve multiple agents working together")

    # Clean up the REPL
    delete_repl(repl_id)
    print("REPL cleaned up. The permanent agent system will be available in future sessions.")
    ```

    ## Using the Permanent Agent System in Future Sessions

    In future sessions, you can reuse your persistent agents like this:

    ```python
    # Set up the framework
    setup_result = setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/modular_empowerment_framework")

    # Create a REPL and initialize MEF
    repl_id = create_python_repl()
    init_result = initialize_modular_empowerment(repl_id=repl_id)

    # Run this code to load your previous agents
    run_python_in_repl(code='''
    import json
    import os
    import random
    from datetime import datetime, timedelta

    # Import the PermanentAgentSystem and agent classes (defined in previous session)
    # ... (copy the class definitions from above) ...

    # Initialize the system with the MEF
    permanent_agents = PermanentAgentSystem().initialize(mef)

    # Now you can use your persistent agents
    memory_agent = permanent_agents.get_agent("memory_adv")
    researcher = permanent_agents.get_agent("researcher_adv")

    print(f"Loaded {len(permanent_agents.agents)} agents")
    for summary in permanent_agents.get_agent_summary():
        print(f"- {summary['id']} ({summary['type']}): Energy={summary['energy']:.2f}, Empowerment={summary['empowerment']:.2f}")
    ''', repl_id=repl_id)
    ```

    ## Extension Ideas

    1. **Create Additional Agent Types**:
       - A Planner agent that creates and manages project plans
       - A Creative agent specialized in generating ideas and content
       - A Data Analysis agent that processes and visualizes data
       - A Communication agent that handles interactions with external systems

    2. **Implement Learning and Growth Mechanisms**:
       - Add capabilities for agents to learn from experience
       - Implement skill improvement based on task success
       - Create agent mentorship where agents can learn from each other

    3. **Develop Complex Workflows**:
       - Multi-stage research and development projects
       - Competitive agent scenarios to optimize solutions
       - Collaborative problem-solving with specialized agent teams

    4. **Integrate with External Systems**:
       - Connect agents to data sources for real-time information
       - Implement APIs for agents to control external services
       - Create web dashboards to monitor and interact with agents

    Note: This enhanced workflow supports complex data types while maintaining compatibility with the MEF framework. The agent persistence system ensures that your agents' knowledge and capabilities are preserved across sessions, making them ideal for long-term projects.
    """
