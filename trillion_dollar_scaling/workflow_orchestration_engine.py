"""
Advanced Workflow Orchestration - Cross-Agent Automation Engine
$15B+ Value Potential - Intelligent Process Automation for Enterprise
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import asyncio
import aiohttp
from openai import OpenAI
from enum import Enum
import networkx as nx
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "workflow-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///workflows.db")

db.init_app(app)

# Workflow Enums
class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    DATA_CHANGE = "data_change"
    API_CALL = "api_call"

# Workflow Data Models
class WorkflowDefinition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    workflow_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    version = db.Column(db.String(20), default='1.0.0')
    
    # Workflow Configuration
    workflow_config = db.Column(db.JSON)  # Complete workflow definition
    agent_dependencies = db.Column(db.JSON)  # Required agents
    trigger_config = db.Column(db.JSON)  # Trigger configuration
    
    # Execution Settings
    max_execution_time = db.Column(db.Integer, default=3600)  # seconds
    retry_policy = db.Column(db.JSON)
    error_handling = db.Column(db.JSON)
    
    # Metadata
    created_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.Enum(WorkflowStatus), default=WorkflowStatus.DRAFT)
    
    # Analytics
    execution_count = db.Column(db.Integer, default=0)
    success_rate = db.Column(db.Float, default=0.0)
    avg_execution_time = db.Column(db.Float, default=0.0)

class WorkflowExecution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    execution_id = db.Column(db.String(100), unique=True, nullable=False)
    workflow_id = db.Column(db.String(100), db.ForeignKey('workflow_definition.workflow_id'), nullable=False)
    
    # Execution Details
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.Enum(WorkflowStatus), default=WorkflowStatus.ACTIVE)
    
    # Execution Context
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    execution_context = db.Column(db.JSON)
    
    # Results
    tasks_total = db.Column(db.Integer, default=0)
    tasks_completed = db.Column(db.Integer, default=0)
    tasks_failed = db.Column(db.Integer, default=0)
    
    # Error Handling
    error_details = db.Column(db.JSON)
    retry_count = db.Column(db.Integer, default=0)
    
    # Triggered by
    triggered_by = db.Column(db.String(100))
    trigger_type = db.Column(db.Enum(TriggerType))

class TaskExecution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(100), nullable=False)
    execution_id = db.Column(db.String(100), db.ForeignKey('workflow_execution.execution_id'), nullable=False)
    
    # Task Details
    task_name = db.Column(db.String(200), nullable=False)
    task_type = db.Column(db.String(100), nullable=False)  # agent_call, data_transform, decision, etc.
    agent_id = db.Column(db.String(100))
    
    # Execution
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    status = db.Column(db.Enum(TaskStatus), default=TaskStatus.PENDING)
    
    # Data
    input_data = db.Column(db.JSON)
    output_data = db.Column(db.JSON)
    task_config = db.Column(db.JSON)
    
    # Results
    execution_time = db.Column(db.Float, default=0.0)
    error_details = db.Column(db.JSON)
    retry_count = db.Column(db.Integer, default=0)

class AgentRegistry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Agent Details
    agent_type = db.Column(db.String(100), nullable=False)
    capabilities = db.Column(db.JSON)
    api_endpoints = db.Column(db.JSON)
    
    # Configuration
    base_url = db.Column(db.String(500))
    authentication = db.Column(db.JSON)
    rate_limits = db.Column(db.JSON)
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    last_health_check = db.Column(db.DateTime)
    health_status = db.Column(db.String(50), default='unknown')
    
    # Performance Metrics
    avg_response_time = db.Column(db.Float, default=0.0)
    success_rate = db.Column(db.Float, default=100.0)
    error_rate = db.Column(db.Float, default=0.0)

# Workflow Orchestration Engine
class WorkflowOrchestrator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
        # Active executions
        self.active_executions = {}
        
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any], triggered_by: str = 'manual') -> Dict[str, Any]:
        """Execute a workflow with given input data"""
        
        # Get workflow definition
        workflow = WorkflowDefinition.query.filter_by(workflow_id=workflow_id).first()
        if not workflow:
            return {'error': 'Workflow not found'}
        
        if workflow.status != WorkflowStatus.ACTIVE:
            return {'error': 'Workflow is not active'}
        
        # Create execution record
        execution = WorkflowExecution(
            execution_id=f"EXEC_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            workflow_id=workflow_id,
            input_data=input_data,
            triggered_by=triggered_by,
            trigger_type=TriggerType.MANUAL if triggered_by == 'manual' else TriggerType.API_CALL
        )
        
        db.session.add(execution)
        db.session.commit()
        
        # Start workflow execution
        self.active_executions[execution.execution_id] = execution
        
        try:
            result = await self._execute_workflow_tasks(execution, workflow)
            
            # Update execution status
            execution.status = WorkflowStatus.COMPLETED if result['success'] else WorkflowStatus.FAILED
            execution.completed_at = datetime.utcnow()
            execution.output_data = result.get('output_data', {})
            execution.tasks_completed = result.get('tasks_completed', 0)
            execution.tasks_failed = result.get('tasks_failed', 0)
            
            db.session.commit()
            
            # Update workflow statistics
            await self._update_workflow_stats(workflow, execution)
            
            return {
                'execution_id': execution.execution_id,
                'success': result['success'],
                'output_data': result.get('output_data', {}),
                'execution_time': result.get('execution_time', 0),
                'tasks_completed': result.get('tasks_completed', 0)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_details = {'error': str(e)}
            db.session.commit()
            
            return {'error': 'Workflow execution failed', 'execution_id': execution.execution_id}
        
        finally:
            # Clean up
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
    
    async def _execute_workflow_tasks(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Execute workflow tasks in proper order"""
        
        workflow_config = workflow.workflow_config
        tasks = workflow_config.get('tasks', [])
        
        if not tasks:
            return {'success': False, 'error': 'No tasks defined'}
        
        # Build task dependency graph
        task_graph = self._build_task_graph(tasks)
        
        # Execute tasks in topological order
        execution_context = {'input_data': execution.input_data}
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = datetime.utcnow()
        
        try:
            # Get execution order
            execution_order = list(nx.topological_sort(task_graph))
            
            for task_id in execution_order:
                task_config = next(t for t in tasks if t['id'] == task_id)
                
                # Execute task
                task_result = await self._execute_task(execution, task_config, execution_context)
                
                if task_result['success']:
                    completed_tasks += 1
                    # Add task output to context
                    execution_context[f"task_{task_id}_output"] = task_result.get('output_data', {})
                else:
                    failed_tasks += 1
                    
                    # Check if task failure should stop workflow
                    if task_config.get('critical', True):
                        break
                    
            # Calculate results
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            success = failed_tasks == 0
            
            return {
                'success': success,
                'execution_time': execution_time,
                'tasks_completed': completed_tasks,
                'tasks_failed': failed_tasks,
                'output_data': execution_context.get('workflow_output', {})
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_task_graph(self, tasks: List[Dict[str, Any]]) -> nx.DiGraph:
        """Build directed graph of task dependencies"""
        
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in tasks:
            graph.add_node(task['id'], **task)
        
        # Add dependency edges
        for task in tasks:
            dependencies = task.get('depends_on', [])
            for dep in dependencies:
                graph.add_edge(dep, task['id'])
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains circular dependencies")
        
        return graph
    
    async def _execute_task(self, execution: WorkflowExecution, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual task"""
        
        task_execution = TaskExecution(
            task_id=task_config['id'],
            execution_id=execution.execution_id,
            task_name=task_config['name'],
            task_type=task_config['type'],
            agent_id=task_config.get('agent_id'),
            input_data=context,
            task_config=task_config
        )
        
        db.session.add(task_execution)
        db.session.commit()
        
        start_time = datetime.utcnow()
        
        try:
            # Execute based on task type
            if task_config['type'] == 'agent_call':
                result = await self._execute_agent_task(task_config, context)
            elif task_config['type'] == 'data_transform':
                result = await self._execute_data_transform(task_config, context)
            elif task_config['type'] == 'decision':
                result = await self._execute_decision_task(task_config, context)
            elif task_config['type'] == 'parallel':
                result = await self._execute_parallel_tasks(task_config, context)
            else:
                result = {'success': False, 'error': f"Unknown task type: {task_config['type']}"}
            
            # Update task execution
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            task_execution.completed_at = datetime.utcnow()
            task_execution.execution_time = execution_time
            task_execution.status = TaskStatus.COMPLETED if result['success'] else TaskStatus.FAILED
            task_execution.output_data = result.get('output_data', {})
            task_execution.error_details = result.get('error_details', {}) if not result['success'] else None
            
            db.session.commit()
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            task_execution.status = TaskStatus.FAILED
            task_execution.error_details = {'error': str(e)}
            db.session.commit()
            
            return {'success': False, 'error': str(e)}
    
    async def _execute_agent_task(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task that calls an AI agent"""
        
        agent_id = task_config.get('agent_id')
        if not agent_id:
            return {'success': False, 'error': 'No agent_id specified'}
        
        # Get agent details
        agent = AgentRegistry.query.filter_by(agent_id=agent_id).first()
        if not agent:
            return {'success': False, 'error': f'Agent {agent_id} not found'}
        
        if not agent.is_active:
            return {'success': False, 'error': f'Agent {agent_id} is inactive'}
        
        # Prepare request data
        request_data = self._prepare_agent_request(task_config, context)
        
        # Make API call to agent
        async with aiohttp.ClientSession() as session:
            try:
                endpoint = f"{agent.base_url}/process"
                timeout = aiohttp.ClientTimeout(total=task_config.get('timeout', 300))
                
                async with session.post(endpoint, json=request_data, timeout=timeout) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        return {
                            'success': True,
                            'output_data': result_data,
                            'agent_id': agent_id
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f'Agent call failed: {response.status}',
                            'error_details': {'status': response.status, 'response': error_text}
                        }
                        
            except asyncio.TimeoutError:
                return {'success': False, 'error': 'Agent call timeout'}
            except Exception as e:
                return {'success': False, 'error': f'Agent call failed: {str(e)}'}
    
    def _prepare_agent_request(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request data for agent call"""
        
        # Extract input mapping
        input_mapping = task_config.get('input_mapping', {})
        request_data = {}
        
        for output_key, input_key in input_mapping.items():
            if input_key in context:
                request_data[output_key] = context[input_key]
        
        # Add task-specific parameters
        if 'parameters' in task_config:
            request_data.update(task_config['parameters'])
        
        return request_data
    
    async def _execute_data_transform(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformation task"""
        
        transformation = task_config.get('transformation', {})
        transform_type = transformation.get('type', 'simple')
        
        try:
            if transform_type == 'simple':
                # Simple field mapping and transformation
                result = self._simple_transform(transformation, context)
            elif transform_type == 'ai_transform':
                # AI-powered data transformation
                result = await self._ai_transform(transformation, context)
            else:
                return {'success': False, 'error': f'Unknown transform type: {transform_type}'}
            
            return {'success': True, 'output_data': result}
            
        except Exception as e:
            return {'success': False, 'error': f'Data transformation failed: {str(e)}'}
    
    def _simple_transform(self, transformation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple data transformation"""
        
        mapping = transformation.get('mapping', {})
        result = {}
        
        for output_key, input_key in mapping.items():
            if input_key in context:
                result[output_key] = context[input_key]
        
        return result
    
    async def _ai_transform(self, transformation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered data transformation"""
        
        prompt = f"""
        Transform the following data according to the specified requirements:
        
        Input Data: {json.dumps(context, indent=2)}
        
        Transformation Requirements: {transformation.get('requirements', '')}
        Output Schema: {transformation.get('output_schema', {})}
        
        Perform the data transformation and return the result in the specified format.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a data transformation specialist. Transform input data according to requirements and return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"AI transformation failed: {e}")
            return {'error': 'AI transformation failed'}
    
    async def _execute_decision_task(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision/branching logic task"""
        
        decision_logic = task_config.get('decision_logic', {})
        decision_type = decision_logic.get('type', 'simple')
        
        try:
            if decision_type == 'simple':
                result = self._simple_decision(decision_logic, context)
            elif decision_type == 'ai_decision':
                result = await self._ai_decision(decision_logic, context)
            else:
                return {'success': False, 'error': f'Unknown decision type: {decision_type}'}
            
            return {'success': True, 'output_data': result}
            
        except Exception as e:
            return {'success': False, 'error': f'Decision task failed: {str(e)}'}
    
    def _simple_decision(self, decision_logic: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simple rule-based decision"""
        
        conditions = decision_logic.get('conditions', [])
        
        for condition in conditions:
            field = condition.get('field')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if field in context:
                field_value = context[field]
                
                if operator == 'equals' and field_value == value:
                    return condition.get('result', {})
                elif operator == 'greater_than' and field_value > value:
                    return condition.get('result', {})
                elif operator == 'less_than' and field_value < value:
                    return condition.get('result', {})
        
        # Default result if no conditions match
        return decision_logic.get('default_result', {})
    
    async def _ai_decision(self, decision_logic: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-powered decision making"""
        
        prompt = f"""
        Make a decision based on the following context and requirements:
        
        Context Data: {json.dumps(context, indent=2)}
        
        Decision Requirements: {decision_logic.get('requirements', '')}
        Available Options: {decision_logic.get('options', [])}
        
        Analyze the context and make the best decision based on the requirements.
        Return the decision with reasoning.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a decision support system. Analyze context and make informed decisions based on requirements."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"AI decision failed: {e}")
            return {'decision': 'default', 'error': 'AI decision failed'}
    
    async def _execute_parallel_tasks(self, task_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        
        parallel_tasks = task_config.get('parallel_tasks', [])
        
        if not parallel_tasks:
            return {'success': False, 'error': 'No parallel tasks specified'}
        
        # Execute all tasks concurrently
        tasks = []
        for ptask in parallel_tasks:
            task = self._execute_task(None, ptask, context)  # Note: execution is None for parallel tasks
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        all_outputs = {}
        success_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Parallel task {i} failed: {result}")
            elif result.get('success', False):
                success_count += 1
                all_outputs[f"parallel_task_{i}"] = result.get('output_data', {})
        
        # Determine overall success
        min_success = task_config.get('min_success_count', len(parallel_tasks))
        success = success_count >= min_success
        
        return {
            'success': success,
            'output_data': all_outputs,
            'tasks_completed': success_count,
            'tasks_total': len(parallel_tasks)
        }
    
    async def _update_workflow_stats(self, workflow: WorkflowDefinition, execution: WorkflowExecution):
        """Update workflow execution statistics"""
        
        workflow.execution_count += 1
        
        # Calculate success rate
        total_executions = WorkflowExecution.query.filter_by(workflow_id=workflow.workflow_id).count()
        successful_executions = WorkflowExecution.query.filter_by(
            workflow_id=workflow.workflow_id,
            status=WorkflowStatus.COMPLETED
        ).count()
        
        workflow.success_rate = (successful_executions / total_executions) * 100 if total_executions > 0 else 0
        
        # Calculate average execution time
        if execution.completed_at and execution.started_at:
            execution_time = (execution.completed_at - execution.started_at).total_seconds()
            
            if workflow.avg_execution_time == 0:
                workflow.avg_execution_time = execution_time
            else:
                # Moving average
                workflow.avg_execution_time = (workflow.avg_execution_time * 0.9) + (execution_time * 0.1)
        
        db.session.commit()

# Workflow Builder
class WorkflowBuilder:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate_workflow_from_description(self, description: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate workflow definition from natural language description"""
        
        prompt = f"""
        Create a workflow definition based on the following description and requirements:
        
        Description: {description}
        
        Requirements:
        - Available Agents: {requirements.get('available_agents', [])}
        - Input Data: {requirements.get('input_schema', {})}
        - Output Requirements: {requirements.get('output_requirements', '')}
        - Performance Requirements: {requirements.get('performance', {})}
        
        Generate a complete workflow definition including:
        1. Task breakdown with dependencies
        2. Agent assignments for each task
        3. Data flow between tasks
        4. Error handling strategies
        5. Performance optimizations
        
        Format as JSON workflow configuration.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a workflow automation expert. Design efficient workflows that leverage AI agents for maximum automation and performance."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            workflow_def = json.loads(response.choices[0].message.content)
            workflow_def['generated_at'] = datetime.utcnow().isoformat()
            workflow_def['ai_generated'] = True
            
            return workflow_def
            
        except Exception as e:
            logger.error(f"Workflow generation failed: {e}")
            return {'error': 'Workflow generation failed'}

# Initialize components
orchestrator = WorkflowOrchestrator()
workflow_builder = WorkflowBuilder()

# Workflow Routes
@app.route('/workflows')
def workflow_dashboard():
    """Workflow orchestration dashboard"""
    
    # Get workflow definitions
    workflows = WorkflowDefinition.query.order_by(WorkflowDefinition.updated_at.desc()).limit(10).all()
    
    # Get recent executions
    executions = WorkflowExecution.query.order_by(WorkflowExecution.started_at.desc()).limit(10).all()
    
    return render_template('workflows/dashboard.html',
                         workflows=workflows,
                         executions=executions)

@app.route('/workflows/api/execute', methods=['POST'])
async def execute_workflow():
    """API endpoint for workflow execution"""
    
    data = request.get_json()
    
    workflow_id = data.get('workflow_id')
    input_data = data.get('input_data', {})
    triggered_by = data.get('triggered_by', 'api')
    
    if not workflow_id:
        return jsonify({'error': 'Workflow ID required'}), 400
    
    # Execute workflow
    result = await orchestrator.execute_workflow(workflow_id, input_data, triggered_by)
    
    return jsonify(result)

@app.route('/workflows/api/create', methods=['POST'])
def create_workflow():
    """API endpoint for workflow creation"""
    
    data = request.get_json()
    
    workflow = WorkflowDefinition(
        workflow_id=f"WF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name=data.get('name', 'New Workflow'),
        description=data.get('description', ''),
        workflow_config=data.get('config', {}),
        created_by=data.get('created_by', 'api')
    )
    
    db.session.add(workflow)
    db.session.commit()
    
    return jsonify({
        'workflow_id': workflow.workflow_id,
        'status': 'created'
    })

@app.route('/workflows/api/generate', methods=['POST'])
def generate_workflow():
    """API endpoint for AI-powered workflow generation"""
    
    data = request.get_json()
    
    description = data.get('description', '')
    requirements = data.get('requirements', {})
    
    if not description:
        return jsonify({'error': 'Workflow description required'}), 400
    
    # Generate workflow
    workflow_def = workflow_builder.generate_workflow_from_description(description, requirements)
    
    return jsonify(workflow_def)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample agents
    if AgentRegistry.query.count() == 0:
        sample_agents = [
            {
                'agent_id': 'financial_analyzer',
                'name': 'Financial Analysis Agent',
                'description': 'AI agent for financial data analysis',
                'agent_type': 'finance',
                'capabilities': ['financial_analysis', 'risk_assessment'],
                'base_url': 'http://localhost:5003',
                'health_status': 'healthy'
            },
            {
                'agent_id': 'healthcare_assistant',
                'name': 'Healthcare AI Assistant',
                'description': 'HIPAA-compliant healthcare AI agent',
                'agent_type': 'healthcare',
                'capabilities': ['clinical_analysis', 'drug_interactions'],
                'base_url': 'http://localhost:5002',
                'health_status': 'healthy'
            }
        ]
        
        for agent_data in sample_agents:
            agent = AgentRegistry(**agent_data)
            db.session.add(agent)
        
        db.session.commit()
        logger.info("Sample workflow data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, debug=True)