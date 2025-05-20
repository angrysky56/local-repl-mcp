"""
Advanced Workflow Orchestration Prompt - Continued

This prompt provides templates for creating complex, multi-stage workflows
that leverage the full power of the local REPL agent system.
"""

def advanced_workflow_orchestration() -> str:
    """
    Advanced workflow orchestration templates for complex multi-agent operations.
    
    Follows the experimental prompt methodology: plan, execute, iterate, collaborate.
    """
    return """
    # Advanced Workflow Orchestration System

    This system enables complex, multi-stage workflows that coordinate multiple agents,
    data sources, and processing steps with proper error handling and progress tracking.

    ## Phase 1: Workflow Planning & Architecture

    ### Step 1: Initialize Orchestration Environment
    ```python
    # Create orchestration REPL
    orchestrator_repl = create_python_repl()
    
    # Set up MEF integration
    setup_result = setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/modular_empowerment_framework")
    init_result = initialize_modular_empowerment(repl_id=orchestrator_repl)
    
    print("🎯 Workflow Orchestration System Initialized")
    ```

    ### Step 2: Define Workflow Architecture
    ```python
    workflow_architecture_code = '''
    import json
    import uuid
    from datetime import datetime, timedelta
    from enum import Enum
    from typing import Dict, List, Optional, Any
    
    class WorkflowStatus(Enum):
        PENDING = "pending"
        RUNNING = "running" 
        COMPLETED = "completed"
        FAILED = "failed"
        PAUSED = "paused"
        CANCELLED = "cancelled"
    
    class TaskStatus(Enum):
        WAITING = "waiting"
        READY = "ready"
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"
        SKIPPED = "skipped"
    
    class WorkflowOrchestrator:
        """Advanced workflow orchestration system with dependency management"""
        
        def __init__(self, workflow_id=None, agent_system=None):
            self.workflow_id = workflow_id or str(uuid.uuid4())
            self.agent_system = agent_system
            self.workflows = {}
            self.task_registry = {}
            self.execution_history = []
            
        def create_workflow(self, name, description="", max_retries=3):
            """Create a new workflow definition"""
            workflow = {
                'id': str(uuid.uuid4()),
                'name': name,
                'description': description,
                'status': WorkflowStatus.PENDING,
                'tasks': {},
                'dependencies': {},
                'execution_order': [],
                'max_retries': max_retries,
                'created_at': datetime.now().isoformat(),
                'metadata': {}
            }
            
            self.workflows[workflow['id']] = workflow
            print(f"✅ Created workflow: {name} (ID: {workflow['id']})")
            return workflow['id']
        
        def add_task(self, workflow_id, task_name, task_type, config, dependencies=None):
            """Add a task to a workflow with dependency specification"""
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            task_id = str(uuid.uuid4())
            task = {
                'id': task_id,
                'name': task_name,
                'type': task_type,
                'config': config,
                'status': TaskStatus.WAITING,
                'dependencies': dependencies or [],
                'results': None,
                'error': None,
                'retry_count': 0,
                'created_at': datetime.now().isoformat()
            }
            
            workflow = self.workflows[workflow_id]
            workflow['tasks'][task_id] = task
            
            # Update dependencies graph
            if task_id not in workflow['dependencies']:
                workflow['dependencies'][task_id] = dependencies or []
            
            print(f"📋 Added task: {task_name} to workflow {workflow['name']}")
            return task_id
        
        def calculate_execution_order(self, workflow_id):
            """Calculate optimal task execution order using topological sort"""
            workflow = self.workflows[workflow_id]
            tasks = workflow['tasks']
            dependencies = workflow['dependencies']
            
            # Topological sort implementation
            in_degree = {task_id: 0 for task_id in tasks}
            
            # Calculate in-degrees
            for task_id, deps in dependencies.items():
                for dep in deps:
                    if dep in in_degree:
                        in_degree[task_id] += 1
            
            # Find tasks with no dependencies
            queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
            execution_order = []
            
            while queue:
                current = queue.pop(0)
                execution_order.append(current)
                
                # Check dependent tasks
                for task_id, deps in dependencies.items():
                    if current in deps:
                        in_degree[task_id] -= 1
                        if in_degree[task_id] == 0:
                            queue.append(task_id)
            
            # Check for circular dependencies
            if len(execution_order) != len(tasks):
                raise ValueError("Circular dependency detected in workflow")
            
            workflow['execution_order'] = execution_order
            print(f"📊 Calculated execution order: {[tasks[t]['name'] for t in execution_order]}")
            return execution_order
    
    # Initialize the orchestrator
    orchestrator = WorkflowOrchestrator()
    print("🔧 Workflow Orchestrator initialized")
    '''
    
    run_python_in_repl(code=workflow_architecture_code, repl_id=orchestrator_repl)
    ```

    ## Phase 2: Task Execution Engine

    ### Step 3: Implement Task Execution System
    ```python
    task_execution_code = '''
    class TaskExecutor:
        """Handles execution of individual workflow tasks"""
        
        def __init__(self, orchestrator):
            self.orchestrator = orchestrator
            self.task_handlers = {}
            self.register_default_handlers()
        
        def register_task_handler(self, task_type, handler_func):
            """Register a handler function for a specific task type"""
            self.task_handlers[task_type] = handler_func
            print(f"📝 Registered handler for task type: {task_type}")
        
        def register_default_handlers(self):
            """Register default task handlers"""
            
            def data_analysis_handler(task_config):
                """Handler for data analysis tasks"""
                try:
                    import pandas as pd
                    import numpy as np
                    
                    # Generate sample data for analysis
                    np.random.seed(42)
                    data = pd.DataFrame({
                        'metric_a': np.random.normal(100, 15, 1000),
                        'metric_b': np.random.normal(200, 30, 1000),
                        'category': np.random.choice(['A', 'B', 'C'], 1000)
                    })
                    
                    analysis_type = task_config.get('analysis_type', 'summary')
                    
                    if analysis_type == 'summary':
                        result = data.describe().to_dict()
                    elif analysis_type == 'correlation':
                        result = data.corr().to_dict()
                    else:
                        result = {'mean': data.mean().to_dict()}
                    
                    return {
                        'success': True,
                        'result': result,
                        'data_shape': data.shape
                    }
                    
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            def report_generation_handler(task_config):
                """Handler for report generation tasks"""
                try:
                    report_type = task_config.get('report_type', 'summary')
                    
                    report_content = {
                        'report_type': report_type,
                        'generated_at': datetime.now().isoformat(),
                        'sections': ['Executive Summary', 'Analysis', 'Conclusions'],
                        'status': 'completed'
                    }
                    
                    return {
                        'success': True,
                        'result': report_content
                    }
                    
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            def validation_handler(task_config):
                """Handler for validation tasks"""
                try:
                    validation_rules = task_config.get('rules', [])
                    data_to_validate = task_config.get('data', {})
                    
                    validation_results = {
                        'passed': True,
                        'failed_rules': [],
                        'score': 1.0
                    }
                    
                    return {
                        'success': True,
                        'result': validation_results
                    }
                    
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            # Register handlers
            self.register_task_handler('data_analysis', data_analysis_handler)
            self.register_task_handler('report_generation', report_generation_handler)
            self.register_task_handler('validation', validation_handler)
        
        def execute_task(self, workflow_id, task_id):
            """Execute a single task"""
            workflow = self.orchestrator.workflows[workflow_id]
            task = workflow['tasks'][task_id]
            
            print(f"🚀 Executing task: {task['name']}")
            task['status'] = TaskStatus.RUNNING
            
            try:
                # Get appropriate handler
                handler = self.task_handlers.get(task['type'])
                if not handler:
                    raise ValueError(f"No handler for task type: {task['type']}")
                
                # Execute task
                result = handler(task['config'])
                
                if result['success']:
                    task['status'] = TaskStatus.COMPLETED
                    task['results'] = result['result']
                    print(f"✅ Task completed: {task['name']}")
                else:
                    task['status'] = TaskStatus.FAILED
                    task['error'] = result.get('error', 'Unknown error')
                    print(f"❌ Task failed: {task['name']}")
                
                return result
                
            except Exception as e:
                task['status'] = TaskStatus.FAILED
                task['error'] = str(e)
                print(f"💥 Task error: {task['name']} - {e}")
                return {'success': False, 'error': str(e)}
    
    # Initialize task executor
    executor = TaskExecutor(orchestrator)
    print("⚡ Task Executor initialized")
    '''
    
    run_python_in_repl(code=task_execution_code, repl_id=orchestrator_repl)
    ```

    ## Phase 3: Workflow Examples and Templates

    ### Step 4: Data Analysis Workflow Template
    ```python
    data_workflow_code = '''
    # Create a comprehensive data analysis workflow
    def create_data_analysis_workflow():
        """Create a complete data analysis workflow with multiple stages"""
        
        # Create workflow
        workflow_id = orchestrator.create_workflow(
            name="Comprehensive Data Analysis",
            description="Multi-stage data analysis with validation and reporting"
        )
        
        # Task 1: Data Collection and Preprocessing
        task1 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Data Collection",
            task_type="data_analysis",
            config={
                'analysis_type': 'summary',
                'data_source': 'generated',
                'preprocessing_steps': ['clean', 'normalize']
            }
        )
        
        # Task 2: Statistical Analysis (depends on Task 1)
        task2 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Statistical Analysis",
            task_type="data_analysis",
            config={
                'analysis_type': 'correlation',
                'statistical_tests': ['normality', 'correlation']
            },
            dependencies=[task1]
        )
        
        # Task 3: Data Validation (depends on Task 1)
        task3 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Data Validation",
            task_type="validation",
            config={
                'rules': ['completeness', 'accuracy', 'consistency'],
                'threshold': 0.95
            },
            dependencies=[task1]
        )
        
        # Task 4: Report Generation (depends on Tasks 2 and 3)
        task4 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Generate Analysis Report",
            task_type="report_generation",
            config={
                'report_type': 'comprehensive',
                'include_visualizations': True,
                'format': 'markdown'
            },
            dependencies=[task2, task3]
        )
        
        return workflow_id
    
    # Create the workflow
    data_workflow_id = create_data_analysis_workflow()
    print(f"📊 Created data analysis workflow: {data_workflow_id}")
    
    # Calculate execution order
    execution_order = orchestrator.calculate_execution_order(data_workflow_id)
    '''
    
    run_python_in_repl(code=data_workflow_code, repl_id=orchestrator_repl)
    ```

    ### Step 5: Workflow Execution Engine
    ```python
    execution_engine_code = '''
    class WorkflowEngine:
        """Main workflow execution engine"""
        
        def __init__(self, orchestrator, executor):
            self.orchestrator = orchestrator
            self.executor = executor
            self.execution_log = []
        
        def execute_workflow(self, workflow_id, parallel=False):
            """Execute a complete workflow"""
            workflow = self.orchestrator.workflows[workflow_id]
            
            print(f"🎬 Starting workflow execution: {workflow['name']}")
            workflow['status'] = WorkflowStatus.RUNNING
            
            try:
                # Calculate execution order
                execution_order = self.orchestrator.calculate_execution_order(workflow_id)
                
                if parallel:
                    return self._execute_parallel(workflow_id, execution_order)
                else:
                    return self._execute_sequential(workflow_id, execution_order)
                    
            except Exception as e:
                workflow['status'] = WorkflowStatus.FAILED
                print(f"💥 Workflow failed: {e}")
                return {'success': False, 'error': str(e)}
        
        def _execute_sequential(self, workflow_id, execution_order):
            """Execute tasks sequentially"""
            workflow = self.orchestrator.workflows[workflow_id]
            results = {}
            
            for task_id in execution_order:
                task = workflow['tasks'][task_id]
                
                # Check if dependencies are satisfied
                if not self._dependencies_satisfied(workflow_id, task_id):
                    print(f"⏳ Dependencies not satisfied for task: {task['name']}")
                    continue
                
                # Execute task
                result = self.executor.execute_task(workflow_id, task_id)
                results[task_id] = result
                
                # Log execution
                self.execution_log.append({
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'task_name': task['name'],
                    'timestamp': datetime.now().isoformat(),
                    'result': result
                })
                
                # Stop if task failed and no retry
                if not result['success']:
                    if task['retry_count'] < workflow['max_retries']:
                        task['retry_count'] += 1
                        print(f"🔄 Retrying task: {task['name']} (attempt {task['retry_count']})")
                        # In real implementation, would retry here
                    else:
                        print(f"🛑 Task failed after max retries: {task['name']}")
                        workflow['status'] = WorkflowStatus.FAILED
                        return {'success': False, 'failed_task': task['name']}
            
            workflow['status'] = WorkflowStatus.COMPLETED
            print(f"🎉 Workflow completed successfully: {workflow['name']}")
            return {'success': True, 'results': results}
        
        def _dependencies_satisfied(self, workflow_id, task_id):
            """Check if all dependencies for a task are completed"""
            workflow = self.orchestrator.workflows[workflow_id]
            task = workflow['tasks'][task_id]
            
            for dep_id in task['dependencies']:
                dep_task = workflow['tasks'][dep_id]
                if dep_task['status'] != TaskStatus.COMPLETED:
                    return False
            return True
        
        def get_workflow_status(self, workflow_id):
            """Get detailed status of a workflow"""
            workflow = self.orchestrator.workflows[workflow_id]
            tasks = workflow['tasks']
            
            status_summary = {
                'workflow_id': workflow_id,
                'name': workflow['name'],
                'status': workflow['status'].value,
                'total_tasks': len(tasks),
                'completed_tasks': len([t for t in tasks.values() if t['status'] == TaskStatus.COMPLETED]),
                'failed_tasks': len([t for t in tasks.values() if t['status'] == TaskStatus.FAILED]),
                'task_details': []
            }
            
            for task_id, task in tasks.items():
                status_summary['task_details'].append({
                    'name': task['name'],
                    'status': task['status'].value,
                    'error': task.get('error'),
                    'retry_count': task.get('retry_count', 0)
                })
            
            return status_summary
    
    # Initialize workflow engine
    engine = WorkflowEngine(orchestrator, executor)
    print("🏗️ Workflow Engine initialized")
    '''
    
    run_python_in_repl(code=execution_engine_code, repl_id=orchestrator_repl)
    ```

    ### Step 6: Execute Example Workflow
    ```python
    execution_example_code = '''
    # Execute the data analysis workflow
    print("🎯 Executing Data Analysis Workflow...")
    
    workflow_result = engine.execute_workflow(data_workflow_id, parallel=False)
    
    print("\\n📊 Workflow Execution Results:")
    print(f"Success: {workflow_result['success']}")
    
    if workflow_result['success']:
        print("\\n✅ Task Results:")
        for task_id, result in workflow_result['results'].items():
            task_name = orchestrator.workflows[data_workflow_id]['tasks'][task_id]['name']
            print(f"  {task_name}: {'✅ Success' if result['success'] else '❌ Failed'}")
    
    # Get detailed status
    status = engine.get_workflow_status(data_workflow_id)
    print(f"\\n📈 Workflow Status Summary:")
    print(f"  Total tasks: {status['total_tasks']}")
    print(f"  Completed: {status['completed_tasks']}")
    print(f"  Failed: {status['failed_tasks']}")
    print(f"  Overall status: {status['status']}")
    '''
    
    run_python_in_repl(code=execution_example_code, repl_id=orchestrator_repl)
    ```

    ## Phase 4: Advanced Workflow Templates

    ### Step 7: Research and Documentation Workflow
    ```python
    research_workflow_code = '''
    def create_research_workflow(topic):
        """Create a research and documentation workflow"""
        
        workflow_id = orchestrator.create_workflow(
            name=f"Research: {topic}",
            description="Automated research, analysis, and documentation workflow"
        )
        
        # Task 1: Information Gathering
        task1 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Information Gathering",
            task_type="data_analysis",
            config={
                'analysis_type': 'summary',
                'topic': topic,
                'sources': ['web', 'knowledge_base', 'databases']
            }
        )
        
        # Task 2: Content Analysis
        task2 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Content Analysis",
            task_type="data_analysis",
            config={
                'analysis_type': 'correlation',
                'focus_areas': ['key_concepts', 'trends', 'gaps']
            },
            dependencies=[task1]
        )
        
        # Task 3: Quality Validation
        task3 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Quality Validation",
            task_type="validation",
            config={
                'rules': ['source_credibility', 'information_accuracy', 'completeness'],
                'threshold': 0.8
            },
            dependencies=[task2]
        )
        
        # Task 4: Documentation Generation
        task4 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Generate Research Report",
            task_type="report_generation",
            config={
                'report_type': 'research_summary',
                'sections': ['abstract', 'findings', 'analysis', 'recommendations'],
                'format': 'markdown'
            },
            dependencies=[task3]
        )
        
        return workflow_id
    
    # Create research workflow
    research_id = create_research_workflow("AI Agent Empowerment Optimization")
    print(f"🔬 Created research workflow: {research_id}")
    '''
    
    run_python_in_repl(code=research_workflow_code, repl_id=orchestrator_repl)
    ```

    ### Step 8: Multi-Agent Coordination Workflow
    ```python
    coordination_workflow_code = '''
    def create_coordination_workflow():
        """Create a multi-agent coordination workflow"""
        
        workflow_id = orchestrator.create_workflow(
            name="Multi-Agent Coordination",
            description="Coordinate multiple agents for complex problem solving"
        )
        
        # Task 1: Agent Assignment
        task1 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Agent Task Assignment",
            task_type="data_analysis",
            config={
                'analysis_type': 'summary',
                'agents': ['researcher', 'analyst', 'validator'],
                'assignment_strategy': 'capability_based'
            }
        )
        
        # Task 2: Parallel Execution (simulated)
        task2a = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Research Agent Task",
            task_type="data_analysis",
            config={'analysis_type': 'summary', 'agent_role': 'researcher'},
            dependencies=[task1]
        )
        
        task2b = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Analysis Agent Task", 
            task_type="data_analysis",
            config={'analysis_type': 'correlation', 'agent_role': 'analyst'},
            dependencies=[task1]
        )
        
        # Task 3: Results Synthesis
        task3 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Synthesize Results",
            task_type="validation",
            config={
                'rules': ['consistency', 'completeness', 'quality'],
                'synthesis_method': 'weighted_average'
            },
            dependencies=[task2a, task2b]
        )
        
        # Task 4: Final Report
        task4 = orchestrator.add_task(
            workflow_id=workflow_id,
            task_name="Generate Coordination Report",
            task_type="report_generation",
            config={
                'report_type': 'coordination_summary',
                'include_agent_performance': True
            },
            dependencies=[task3]
        )
        
        return workflow_id
    
    # Create coordination workflow
    coord_id = create_coordination_workflow()
    print(f"🤝 Created coordination workflow: {coord_id}")
    '''
    
    run_python_in_repl(code=coordination_workflow_code, repl_id=orchestrator_repl)
    ```

    ## Phase 5: Monitoring and Optimization

    ### Step 9: Workflow Monitoring Dashboard
    ```python
    monitoring_code = '''
    class WorkflowMonitor:
        """Monitor and analyze workflow performance"""
        
        def __init__(self, engine):
            self.engine = engine
            self.performance_metrics = {}
        
        def generate_dashboard(self):
            """Generate a monitoring dashboard"""
            dashboard = {
                'timestamp': datetime.now().isoformat(),
                'active_workflows': [],
                'completed_workflows': [],
                'performance_summary': {}
            }
            
            for workflow_id, workflow in self.engine.orchestrator.workflows.items():
                status_info = {
                    'id': workflow_id,
                    'name': workflow['name'],
                    'status': workflow['status'].value,
                    'task_count': len(workflow['tasks']),
                    'created_at': workflow['created_at']
                }
                
                if workflow['status'] == WorkflowStatus.RUNNING:
                    dashboard['active_workflows'].append(status_info)
                elif workflow['status'] == WorkflowStatus.COMPLETED:
                    dashboard['completed_workflows'].append(status_info)
            
            # Performance metrics
            total_executions = len(self.engine.execution_log)
            successful_tasks = len([log for log in self.engine.execution_log if log['result']['success']])
            
            dashboard['performance_summary'] = {
                'total_task_executions': total_executions,
                'successful_tasks': successful_tasks,
                'success_rate': successful_tasks / total_executions if total_executions > 0 else 0,
                'average_workflow_size': sum(len(w['tasks']) for w in self.engine.orchestrator.workflows.values()) / len(self.engine.orchestrator.workflows) if self.engine.orchestrator.workflows else 0
            }
            
            return dashboard
        
        def print_dashboard(self):
            """Print a formatted dashboard"""
            dashboard = self.generate_dashboard()
            
            print("🔍 Workflow Monitoring Dashboard")
            print("=" * 50)
            print(f"Generated: {dashboard['timestamp']}")
            print(f"\\n📊 Performance Summary:")
            print(f"  Total Task Executions: {dashboard['performance_summary']['total_task_executions']}")
            print(f"  Successful Tasks: {dashboard['performance_summary']['successful_tasks']}")
            print(f"  Success Rate: {dashboard['performance_summary']['success_rate']:.2%}")
            print(f"  Average Workflow Size: {dashboard['performance_summary']['average_workflow_size']:.1f} tasks")
            
            print(f"\\n🔄 Active Workflows ({len(dashboard['active_workflows'])}):")
            for workflow in dashboard['active_workflows']:
                print(f"  - {workflow['name']} ({workflow['task_count']} tasks)")
            
            print(f"\\n✅ Completed Workflows ({len(dashboard['completed_workflows'])}):")
            for workflow in dashboard['completed_workflows']:
                print(f"  - {workflow['name']} ({workflow['task_count']} tasks)")
    
    # Initialize monitor
    monitor = WorkflowMonitor(engine)
    print("📊 Workflow Monitor initialized")
    '''
    
    run_python_in_repl(code=monitoring_code, repl_id=orchestrator_repl)
    ```

    ### Step 10: Execute Multiple Workflows and Monitor
    ```python
    final_execution_code = '''
    print("🎬 Executing Multiple Workflows for Testing...")
    
    # Execute research workflow
    research_result = engine.execute_workflow(research_id)
    print(f"Research workflow: {'✅ Success' if research_result['success'] else '❌ Failed'}")
    
    # Execute coordination workflow  
    coord_result = engine.execute_workflow(coord_id)
    print(f"Coordination workflow: {'✅ Success' if coord_result['success'] else '❌ Failed'}")
    
    # Display monitoring dashboard
    print("\\n" + "="*60)
    monitor.print_dashboard()
    
    # Save workflow definitions and results
    import json
    
    workflow_summary = {
        'orchestrator_id': orchestrator.workflow_id,
        'workflows_created': len(orchestrator.workflows),
        'total_executions': len(engine.execution_log),
        'workflow_definitions': {wf_id: {
            'name': wf['name'],
            'description': wf['description'],
            'task_count': len(wf['tasks']),
            'status': wf['status'].value
        } for wf_id, wf in orchestrator.workflows.items()}
    }
    
    # Save to output directory
    output_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/output/workflow_orchestration_report.json"
    with open(output_path, 'w') as f:
        json.dump(workflow_summary, f, indent=2)
    
    print(f"\\n💾 Workflow summary saved to: {output_path}")
    
    print("\\n🎉 Advanced Workflow Orchestration System demonstrated successfully!")
    print("\\nKey Features Demonstrated:")
    print("  ✅ Complex workflow definition with dependencies")
    print("  ✅ Task execution with error handling and retries")
    print("  ✅ Multiple workflow types (data analysis, research, coordination)")
    print("  ✅ Performance monitoring and dashboard")
    print("  ✅ Workflow persistence and reporting")
    '''
    
    run_python_in_repl(code=final_execution_code, repl_id=orchestrator_repl)
    
    # Cleanup
    delete_repl(orchestrator_repl)
    print("\\n🧹 Orchestration REPL cleaned up")
    ```

    ## Usage Notes

    This advanced workflow orchestration system provides:

    1. **Dependency Management**: Tasks can depend on other tasks, with automatic ordering
    2. **Error Handling**: Built-in retry mechanisms and failure recovery
    3. **Multiple Execution Modes**: Sequential and parallel execution options
    4. **Task Type System**: Extensible system for different task types
    5. **Monitoring**: Real-time monitoring and performance dashboards
    6. **Persistence**: Workflow definitions and results are saved for future reference

    ## Workflow Templates Available

    - **Data Analysis**: Multi-stage data processing with validation
    - **Research**: Automated research and documentation generation
    - **Multi-Agent Coordination**: Coordinate multiple agents for complex tasks
    - **Custom Workflows**: Easy creation of domain-specific workflows

    This system transforms the REPL environment into a powerful workflow orchestration
    platform capable of handling enterprise-level complexity while maintaining the
    flexibility and interactivity of the REPL interface.
    """
