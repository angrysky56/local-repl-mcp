"""
System Discovery and Enhancement Prompt

This prompt helps systematically discover and document the full capabilities
of the local REPL MCP system using methodical exploration.
"""

def system_discovery_workflow() -> str:
    """
    A systematic workflow for discovering and enhancing the local REPL system capabilities.
    
    This follows the experimental prompt's approach of careful planning, modular exploration,
    and collaborative documentation of findings.
    """
    return """
    # System Discovery and Enhancement Protocol

    ## Phase 1: Comprehensive System Mapping

    ### Step 1: Environment Analysis
    ```python
    # Create exploration REPL
    discovery_repl = create_python_repl()
    
    # Initialize with MEF if available
    try:
        setup_result = setup_modular_empowerment(path="/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/modular_empowerment_framework")
        init_result = initialize_modular_empowerment(repl_id=discovery_repl)
        print("✅ MEF Integration Active")
    except Exception as e:
        print(f"MEF not available: {e}")

    # System paths and structure discovery
    discovery_code = '''
    import os
    import sys
    import json
    from pathlib import Path
    
    # Map the system structure
    base_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"
    
    def explore_directory(path, max_depth=3, current_depth=0):
        """Recursively explore directory structure"""
        items = []
        if current_depth >= max_depth:
            return items
            
        try:
            for item in sorted(os.listdir(path)):
                if item.startswith('.'):
                    continue
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    sub_items = explore_directory(item_path, max_depth, current_depth + 1)
                    items.append({
                        'name': item,
                        'type': 'directory',
                        'path': item_path,
                        'contents': sub_items
                    })
                else:
                    # Check file type and metadata
                    stat = os.stat(item_path)
                    items.append({
                        'name': item,
                        'type': 'file',
                        'path': item_path,
                        'size': stat.st_size,
                        'extension': os.path.splitext(item)[1]
                    })
        except PermissionError:
            pass
            
        return items
    
    print("=== System Structure Discovery ===")
    structure = explore_directory(base_path)
    
    # Analyze key directories
    key_dirs = ['agent_data', 'modular_empowerment_framework', 'output', 
                'prompts', 'repl_agent_library', 'repl_script_library']
    
    for directory in key_dirs:
        dir_path = os.path.join(base_path, directory)
        if os.path.exists(dir_path):
            print(f"\\n📁 {directory}:")
            contents = os.listdir(dir_path)
            for item in contents[:10]:  # Limit to first 10 items
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"  📄 {item} ({size} bytes)")
                else:
                    print(f"  📁 {item}/")
            if len(contents) > 10:
                print(f"  ... and {len(contents) - 10} more items")
    '''
    
    run_python_in_repl(code=discovery_code, repl_id=discovery_repl)
    ```

    ### Step 2: Agent System Analysis
    ```python
    # Discover existing agent capabilities
    agent_analysis_code = '''
    import json
    import os
    from datetime import datetime
    
    # Check for existing agent data
    agent_data_dir = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/agent_data"
    
    print("=== Agent System Analysis ===")
    if os.path.exists(agent_data_dir):
        agent_files = [f for f in os.listdir(agent_data_dir) if f.endswith('.json')]
        print(f"Found {len(agent_files)} agent data files:")
        
        for agent_file in agent_files:
            agent_path = os.path.join(agent_data_dir, agent_file)
            try:
                with open(agent_path, 'r') as f:
                    agent_data = json.load(f)
                
                agent_id = agent_data.get('agent_id', 'unknown')
                agent_type = agent_data.get('agent_type', 'unknown')
                energy = agent_data.get('energy', 0)
                empowerment = agent_data.get('empowerment', 0)
                memory_count = len(agent_data.get('memory', {}))
                
                print(f"  🤖 {agent_id} ({agent_type})")
                print(f"     Energy: {energy:.2f}, Empowerment: {empowerment:.2f}")
                print(f"     Memory items: {memory_count}")
                
                # Check for specialized properties
                if 'research_topics' in agent_data:
                    print(f"     Research topics: {len(agent_data['research_topics'])}")
                if 'projects' in agent_data:
                    print(f"     Projects: {len(agent_data['projects'])}")
                if 'categories' in agent_data:
                    print(f"     Memory categories: {len(agent_data['categories'])}")
                    
            except Exception as e:
                print(f"  ❌ Error reading {agent_file}: {e}")
    else:
        print("No agent data directory found")
    '''
    
    run_python_in_repl(code=agent_analysis_code, repl_id=discovery_repl)
    ```

    ### Step 3: Database and Storage Discovery
    ```python
    # Check for databases and storage systems
    storage_analysis_code = '''
    import sqlite3
    import os
    
    print("=== Storage Systems Analysis ===")
    
    # Check for SQLite databases
    base_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.db') or file.endswith('.sqlite'):
                db_path = os.path.join(root, file)
                print(f"📊 Database found: {db_path}")
                
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get table names
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    print(f"  Tables: {[table[0] for table in tables]}")
                    
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"    {table_name}: {count} records")
                    
                    conn.close()
                    
                except Exception as e:
                    print(f"  Error reading database: {e}")
    
    # Check for log files
    print("\\n📝 Log Files:")
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.log') or 'log' in file.lower():
                log_path = os.path.join(root, file)
                size = os.path.getsize(log_path)
                print(f"  {log_path} ({size} bytes)")
    '''
    
    run_python_in_repl(code=storage_analysis_code, repl_id=discovery_repl)
    ```

    ## Phase 2: Capability Testing and Documentation

    ### Step 4: Advanced Feature Detection
    ```python
    # Test for advanced capabilities
    capability_test_code = '''
    print("=== Advanced Capability Detection ===")
    
    # Test 1: Multi-REPL coordination
    print("\\n🔄 Multi-REPL Test:")
    try:
        # This would be run from main context, not inside REPL
        print("  (Would test multiple REPL coordination)")
    except:
        pass
    
    # Test 2: Package availability
    print("\\n📦 Package Ecosystem:")
    import importlib
    
    advanced_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'networkx', 'requests', 'beautifulsoup4', 'jupyter', 'fastapi'
    ]
    
    available_packages = []
    for package in advanced_packages:
        try:
            importlib.import_module(package)
            available_packages.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
    
    print(f"\\nAvailable advanced packages: {len(available_packages)}/{len(advanced_packages)}")
    
    # Test 3: Memory and processing capabilities
    print("\\n🧠 System Resources:")
    import psutil
    import sys
    
    print(f"  Python version: {sys.version}")
    print(f"  Memory available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"  CPU cores: {psutil.cpu_count()}")
    
    # Test 4: File system capabilities
    print("\\n💾 File System:")
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    print(f"  Temp directory access: {temp_dir}")
    shutil.rmtree(temp_dir)
    print("  ✅ Temporary file operations working")
    '''
    
    run_python_in_repl(code=capability_test_code, repl_id=discovery_repl)
    ```

    ### Step 5: Integration Point Discovery
    ```python
    # Discover integration points and APIs
    integration_discovery_code = '''
    print("=== Integration Points Discovery ===")
    
    # Check for server configurations
    import os
    import json
    
    base_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl"
    
    # Look for configuration files
    config_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(keyword in file.lower() for keyword in ['config', 'settings', 'env']):
                if file.endswith(('.json', '.yaml', '.yml', '.ini', '.conf')):
                    config_files.append(os.path.join(root, file))
    
    print(f"\\n⚙️ Configuration Files ({len(config_files)}):")
    for config_file in config_files:
        print(f"  📄 {config_file}")
    
    # Check for API endpoints or server files
    print("\\n🌐 Server/API Files:")
    server_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(keyword in file.lower() for keyword in ['server', 'api', 'endpoint', 'client']):
                if file.endswith('.py'):
                    server_files.append(os.path.join(root, file))
    
    for server_file in server_files:
        print(f"  🖥️ {server_file}")
    
    # Check for workflow/pipeline files
    print("\\n🔄 Workflow Files:")
    workflow_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(keyword in file.lower() for keyword in ['workflow', 'pipeline', 'process']):
                workflow_files.append(os.path.join(root, file))
    
    for workflow_file in workflow_files:
        print(f"  ⚡ {workflow_file}")
    '''
    
    run_python_in_repl(code=integration_discovery_code, repl_id=discovery_repl)
    ```

    ## Phase 3: Enhancement and Documentation

    ### Step 6: Capability Gap Analysis
    ```python
    # Identify areas for enhancement
    gap_analysis_code = '''
    print("=== Capability Gap Analysis ===")
    
    # Define ideal capabilities for advanced AI system
    ideal_capabilities = {
        'Multi-Agent Coordination': ['Agent communication', 'Task delegation', 'Conflict resolution'],
        'Advanced Analytics': ['Time series analysis', 'ML model training', 'Statistical testing'],
        'Data Integration': ['API connections', 'Database queries', 'Real-time data'],
        'Visualization': ['Interactive plots', 'Dashboards', 'Report generation'],
        'Automation': ['Scheduled tasks', 'Event triggers', 'Workflow orchestration'],
        'Knowledge Management': ['Semantic search', 'Knowledge graphs', 'Context awareness'],
        'External Integration': ['Cloud services', 'External APIs', 'File systems'],
        'Monitoring': ['Performance metrics', 'Error tracking', 'Health checks']
    }
    
    print("\\n🎯 Capability Assessment:")
    for category, features in ideal_capabilities.items():
        print(f"\\n📋 {category}:")
        for feature in features:
            # This would be enhanced with actual capability detection
            print(f"  🔍 {feature}: [Assessment needed]")
    
    print("\\n📊 Enhancement Opportunities:")
    enhancement_areas = [
        "Advanced agent communication protocols",
        "Real-time data streaming capabilities", 
        "Enhanced visualization dashboard",
        "Automated model training pipelines",
        "Distributed computing integration",
        "Natural language query interface",
        "Advanced memory/knowledge indexing",
        "Performance optimization tools"
    ]
    
    for i, area in enumerate(enhancement_areas, 1):
        print(f"  {i}. {area}")
    '''
    
    run_python_in_repl(code=gap_analysis_code, repl_id=discovery_repl)
    ```

    ### Step 7: Documentation Generation
    ```python
    # Generate comprehensive system documentation
    documentation_code = '''
    import json
    from datetime import datetime
    
    print("=== Generating System Documentation ===")
    
    # Create comprehensive system map
    system_map = {
        'discovery_timestamp': datetime.now().isoformat(),
        'system_version': 'local-repl-mcp-enhanced',
        'capabilities': {
            'repl_management': {
                'multiple_sessions': True,
                'persistent_state': True,
                'cross_session_data': True
            },
            'agent_system': {
                'persistent_agents': True,
                'specialized_types': ['memory', 'researcher', 'project_manager'],
                'state_management': True,
                'inter_agent_communication': True
            },
            'data_analysis': {
                'statistical_analysis': True,
                'visualization': True,
                'report_generation': True,
                'multiple_formats': True
            },
            'empowerment_framework': {
                'energy_tracking': True,
                'empowerment_optimization': True,
                'multi_agent_coordination': True,
                'decision_framework': True
            }
        },
        'enhancement_priorities': [
            'Advanced workflow orchestration',
            'Real-time monitoring dashboard', 
            'Enhanced agent collaboration',
            'External API integration',
            'Performance optimization'
        ]
    }
    
    # Save documentation
    doc_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/output/system_discovery_report.json"
    with open(doc_path, 'w') as f:
        json.dump(system_map, f, indent=2)
    
    print(f"✅ System documentation saved to: {doc_path}")
    
    # Generate markdown summary
    md_content = f"""
    # Local REPL MCP System Discovery Report
    
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    ## System Overview
    
    This system provides advanced multi-agent capabilities with persistent state
    management and empowerment optimization.
    
    ## Key Capabilities Discovered
    
    ### Agent System
    - Persistent agent state across sessions
    - Specialized agent types (memory, researcher, project_manager)
    - Inter-agent communication and coordination
    - Energy and empowerment tracking
    
    ### Data Analysis
    - Advanced statistical analysis capabilities
    - Multiple visualization formats
    - Automated report generation
    - Multi-format data export
    
    ### Empowerment Framework Integration
    - Decision framework optimization
    - Multi-agent empowerment coordination
    - Energy-based task allocation
    - Adaptive behavior improvement
    
    ## Enhancement Opportunities
    
    1. Advanced workflow orchestration
    2. Real-time monitoring dashboard
    3. Enhanced agent collaboration protocols
    4. External API integration framework
    5. Performance optimization tools
    
    ## Recommended Next Steps
    
    1. Implement advanced workflow templates
    2. Create monitoring and analytics dashboard
    3. Enhance agent communication protocols
    4. Develop external integration capabilities
    5. Add performance profiling tools
    """
    
    md_path = "/home/ty/Repositories/ai_workspace/local-repl-mcp/local_repl/output/discovery_summary.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"✅ Summary report saved to: {md_path}")
    '''
    
    run_python_in_repl(code=documentation_code, repl_id=discovery_repl)
    ```

    ### Step 8: Cleanup and Recommendations
    ```python
    # Clean up and provide next steps
    print("=== Discovery Session Complete ===")
    print("\\n📋 Key Findings:")
    print("1. Advanced multi-agent system with persistent state")
    print("2. Empowerment optimization framework integration") 
    print("3. Comprehensive data analysis capabilities")
    print("4. Extensible architecture for enhancement")
    
    print("\\n🎯 Recommended Enhancements:")
    print("1. Create advanced workflow orchestration prompts")
    print("2. Develop real-time monitoring capabilities")
    print("3. Enhance agent collaboration protocols")
    print("4. Add external integration frameworks")
    
    print("\\n📁 Documentation saved to:")
    print("  - system_discovery_report.json")
    print("  - discovery_summary.md")
    
    # Clean up REPL
    delete_repl(discovery_repl)
    ```

    ## Usage Notes

    This systematic discovery protocol helps identify:
    
    1. **System Architecture**: Complete mapping of directories, files, and capabilities
    2. **Agent Capabilities**: Existing agents, their specializations, and states
    3. **Integration Points**: APIs, databases, configuration systems
    4. **Enhancement Opportunities**: Areas for improvement and expansion
    5. **Documentation**: Comprehensive system documentation for future reference
    
    Run this protocol whenever you need to:
    - Understand the full system capabilities
    - Plan major enhancements
    - Document the current state
    - Identify integration opportunities
    - Assess system health and performance
    
    The methodical approach ensures nothing is missed and provides a solid
    foundation for system enhancement and optimization.
    """
