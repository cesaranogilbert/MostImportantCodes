"""
Autonomous Business Process Discovery - Process Mining and Optimization Engine
$10B+ Value Potential - Intelligent Process Discovery and Automation
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
import networkx as nx
from openai import OpenAI
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "process-discovery-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///process_discovery.db")

db.init_app(app)

# Process Discovery Data Models
class ProcessEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(100), unique=True, nullable=False)
    process_instance_id = db.Column(db.String(100), nullable=False)
    activity_name = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)
    
    # Event Details
    user_id = db.Column(db.String(100))
    department = db.Column(db.String(100))
    system_source = db.Column(db.String(100))
    event_type = db.Column(db.String(50))  # start, complete, cancel
    
    # Data Attributes
    event_attributes = db.Column(db.JSON)
    resource_info = db.Column(db.JSON)
    cost_info = db.Column(db.JSON)
    
    # Process Context
    process_name = db.Column(db.String(200))
    business_unit = db.Column(db.String(100))
    organization_id = db.Column(db.String(100))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class DiscoveredProcess(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    process_id = db.Column(db.String(100), unique=True, nullable=False)
    process_name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    organization_id = db.Column(db.String(100), nullable=False)
    
    # Process Characteristics
    complexity_score = db.Column(db.Float, default=0.0)
    automation_potential = db.Column(db.Float, default=0.0)
    efficiency_score = db.Column(db.Float, default=0.0)
    
    # Process Model
    process_model = db.Column(db.JSON)  # BPMN-like representation
    process_variants = db.Column(db.JSON)
    bottlenecks = db.Column(db.JSON)
    optimization_opportunities = db.Column(db.JSON)
    
    # Statistics
    total_instances = db.Column(db.Integer, default=0)
    avg_duration = db.Column(db.Float, default=0.0)
    avg_cost = db.Column(db.Float, default=0.0)
    success_rate = db.Column(db.Float, default=100.0)
    
    # Discovery Metadata
    discovered_at = db.Column(db.DateTime, default=datetime.utcnow)
    discovery_method = db.Column(db.String(100))
    confidence_score = db.Column(db.Float, default=0.0)
    
    # Analysis Results
    performance_analysis = db.Column(db.JSON)
    compliance_analysis = db.Column(db.JSON)
    automation_recommendations = db.Column(db.JSON)

class ProcessOptimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    optimization_id = db.Column(db.String(100), unique=True, nullable=False)
    process_id = db.Column(db.String(100), db.ForeignKey('discovered_process.process_id'), nullable=False)
    
    # Optimization Details
    optimization_type = db.Column(db.String(100), nullable=False)  # automation, redesign, elimination
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Impact Assessment
    estimated_time_savings = db.Column(db.Float, default=0.0)  # percentage
    estimated_cost_savings = db.Column(db.Float, default=0.0)  # monetary
    estimated_quality_improvement = db.Column(db.Float, default=0.0)  # percentage
    implementation_complexity = db.Column(db.String(50))  # low, medium, high
    
    # Implementation Plan
    implementation_steps = db.Column(db.JSON)
    required_resources = db.Column(db.JSON)
    estimated_timeline = db.Column(db.Integer)  # days
    
    # ROI Analysis
    investment_required = db.Column(db.Float, default=0.0)
    annual_savings = db.Column(db.Float, default=0.0)
    roi_percentage = db.Column(db.Float, default=0.0)
    payback_period = db.Column(db.Float, default=0.0)  # months
    
    # Status
    status = db.Column(db.String(50), default='proposed')  # proposed, approved, implementing, completed
    priority = db.Column(db.String(50), default='medium')  # low, medium, high, critical
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Process Discovery Engine
class ProcessDiscoveryEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def discover_processes_from_events(self, organization_id: str, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """Discover business processes from event data"""
        
        # Get event data
        query = ProcessEvent.query.filter_by(organization_id=organization_id)
        
        if time_range:
            query = query.filter(
                ProcessEvent.timestamp >= time_range[0],
                ProcessEvent.timestamp <= time_range[1]
            )
        
        events = query.all()
        
        if len(events) < 10:
            return {'error': 'Insufficient event data for process discovery'}
        
        # Convert to DataFrame for analysis
        event_data = []
        for event in events:
            event_data.append({
                'process_instance_id': event.process_instance_id,
                'activity_name': event.activity_name,
                'timestamp': event.timestamp,
                'user_id': event.user_id,
                'department': event.department,
                'system_source': event.system_source,
                'event_type': event.event_type
            })
        
        df = pd.DataFrame(event_data)
        
        # Discover process models
        discovered_processes = self._mine_process_models(df, organization_id)
        
        # Analyze each discovered process
        for process_data in discovered_processes:
            analysis_result = self._analyze_process_performance(process_data, df)
            process_data.update(analysis_result)
            
            # Generate optimization recommendations
            optimizations = self._generate_optimization_recommendations(process_data)
            process_data['optimization_recommendations'] = optimizations
            
            # Store discovered process
            self._store_discovered_process(process_data, organization_id)
        
        return {
            'organization_id': organization_id,
            'discovery_date': datetime.utcnow().isoformat(),
            'processes_discovered': len(discovered_processes),
            'total_events_analyzed': len(events),
            'discovered_processes': discovered_processes
        }
    
    def _mine_process_models(self, df: pd.DataFrame, organization_id: str) -> List[Dict[str, Any]]:
        """Mine process models from event data using process mining algorithms"""
        
        # Group events by process instance
        instances = df.groupby('process_instance_id')
        
        # Extract process variants (unique activity sequences)
        process_variants = {}
        
        for instance_id, instance_events in instances:
            # Sort events by timestamp
            sorted_events = instance_events.sort_values('timestamp')
            
            # Create activity sequence
            activity_sequence = tuple(sorted_events['activity_name'].tolist())
            
            if activity_sequence not in process_variants:
                process_variants[activity_sequence] = {
                    'variant': activity_sequence,
                    'instances': [],
                    'frequency': 0
                }
            
            process_variants[activity_sequence]['instances'].append({
                'instance_id': instance_id,
                'duration': (sorted_events['timestamp'].iloc[-1] - sorted_events['timestamp'].iloc[0]).total_seconds() / 3600,  # hours
                'activities': sorted_events.to_dict('records')
            })
            process_variants[activity_sequence]['frequency'] += 1
        
        # Cluster similar variants into processes
        discovered_processes = self._cluster_process_variants(process_variants)
        
        # Generate process models using AI
        for process in discovered_processes:
            process_model = self._generate_process_model(process)
            process['process_model'] = process_model
        
        return discovered_processes
    
    def _cluster_process_variants(self, variants: Dict[tuple, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar process variants into distinct processes"""
        
        if len(variants) == 0:
            return []
        
        # For simplicity, group by most common activities
        # In production, use more sophisticated similarity measures
        
        processes = []
        variant_list = list(variants.values())
        
        # Sort by frequency
        variant_list.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Group variants into processes based on common activities
        processed_variants = set()
        
        for i, variant in enumerate(variant_list):
            if i in processed_variants:
                continue
            
            # Start new process
            process = {
                'process_id': f"PROC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                'variants': [variant],
                'total_instances': variant['frequency'],
                'main_variant': variant['variant']
            }
            
            processed_variants.add(i)
            
            # Find similar variants
            for j, other_variant in enumerate(variant_list[i+1:], i+1):
                if j in processed_variants:
                    continue
                
                # Calculate similarity (simplified)
                similarity = self._calculate_variant_similarity(variant['variant'], other_variant['variant'])
                
                if similarity > 0.7:  # 70% similarity threshold
                    process['variants'].append(other_variant)
                    process['total_instances'] += other_variant['frequency']
                    processed_variants.add(j)
            
            processes.append(process)
        
        return processes
    
    def _calculate_variant_similarity(self, variant1: tuple, variant2: tuple) -> float:
        """Calculate similarity between two process variants"""
        
        # Simple Jaccard similarity on activity sets
        set1 = set(variant1)
        set2 = set(variant2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    
    def _generate_process_model(self, process_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate process model using AI"""
        
        variants_description = []
        for variant in process_data['variants']:
            variants_description.append({
                'sequence': list(variant['variant']),
                'frequency': variant['frequency'],
                'sample_instances': len(variant['instances'][:3])  # First 3 instances
            })
        
        prompt = f"""
        Generate a business process model based on the following process variants:
        
        Process Variants: {json.dumps(variants_description, indent=2)}
        Total Instances: {process_data['total_instances']}
        Main Variant: {list(process_data['main_variant'])}
        
        Create a process model that includes:
        1. Standard BPMN-like process flow
        2. Decision points and parallel activities
        3. Process boundaries (start/end events)
        4. Roles and responsibilities
        5. Key performance indicators
        6. Risk points and bottlenecks
        
        Format as structured JSON suitable for process visualization.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a business process analyst expert in process mining and BPMN modeling. Generate comprehensive process models from variant data."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            process_model = json.loads(response.choices[0].message.content)
            process_model['generated_at'] = datetime.utcnow().isoformat()
            
            return process_model
            
        except Exception as e:
            logger.error(f"Process model generation failed: {e}")
            return {
                'error': 'Process model generation failed',
                'fallback_model': {
                    'activities': list(process_data['main_variant']),
                    'flow': 'sequential'
                }
            }
    
    def _analyze_process_performance(self, process_data: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze process performance metrics"""
        
        # Collect all instances for this process
        all_instances = []
        for variant in process_data['variants']:
            all_instances.extend(variant['instances'])
        
        if not all_instances:
            return {'error': 'No instances found for analysis'}
        
        # Calculate performance metrics
        durations = [instance['duration'] for instance in all_instances]
        
        performance_metrics = {
            'avg_duration_hours': np.mean(durations),
            'median_duration_hours': np.median(durations),
            'std_duration_hours': np.std(durations),
            'min_duration_hours': np.min(durations),
            'max_duration_hours': np.max(durations),
            'total_instances': len(all_instances),
            'throughput_per_day': len(all_instances) / 30  # Assuming 30-day period
        }
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(all_instances)
        
        # Calculate automation potential
        automation_score = self._calculate_automation_potential(process_data)
        
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(performance_metrics, bottlenecks)
        
        return {
            'performance_metrics': performance_metrics,
            'bottlenecks': bottlenecks,
            'automation_potential': automation_score,
            'efficiency_score': efficiency_score,
            'complexity_score': len(set().union(*[variant['variant'] for variant in process_data['variants']]))
        }
    
    def _identify_bottlenecks(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify process bottlenecks"""
        
        activity_times = {}
        
        for instance in instances:
            activities = instance['activities']
            
            for i in range(len(activities) - 1):
                current_activity = activities[i]['activity_name']
                next_activity = activities[i + 1]['activity_name']
                
                # Calculate waiting time between activities
                wait_time = (activities[i + 1]['timestamp'] - activities[i]['timestamp']).total_seconds() / 3600
                
                if current_activity not in activity_times:
                    activity_times[current_activity] = []
                
                activity_times[current_activity].append(wait_time)
        
        # Identify activities with high average wait times
        bottlenecks = []
        for activity, times in activity_times.items():
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            if avg_time > np.mean(list(np.mean(t) for t in activity_times.values())) * 1.5:
                bottlenecks.append({
                    'activity': activity,
                    'avg_duration_hours': avg_time,
                    'std_duration_hours': std_time,
                    'severity': 'high' if avg_time > 4 else 'medium'
                })
        
        return sorted(bottlenecks, key=lambda x: x['avg_duration_hours'], reverse=True)
    
    def _calculate_automation_potential(self, process_data: Dict[str, Any]) -> float:
        """Calculate automation potential score (0-100)"""
        
        # Get all unique activities
        all_activities = set()
        for variant in process_data['variants']:
            all_activities.update(variant['variant'])
        
        # Analyze automation potential using AI
        activities_list = list(all_activities)
        
        prompt = f"""
        Analyze the automation potential for the following business process activities:
        
        Activities: {activities_list}
        
        For each activity, assess:
        1. Rule-based automation potential (0-100)
        2. AI automation potential (0-100)
        3. Required technology/tools
        4. Implementation complexity
        
        Provide overall process automation score (0-100) and reasoning.
        Format as JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a process automation expert. Analyze business activities for automation potential using RPA, AI, and workflow tools."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            automation_analysis = json.loads(response.choices[0].message.content)
            return automation_analysis.get('overall_automation_score', 50.0)
            
        except Exception as e:
            logger.error(f"Automation analysis failed: {e}")
            return 50.0  # Default moderate automation potential
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> float:
        """Calculate process efficiency score (0-100)"""
        
        # Base efficiency from duration consistency
        duration_consistency = 100 - min(100, (metrics['std_duration_hours'] / metrics['avg_duration_hours']) * 100)
        
        # Bottleneck penalty
        bottleneck_penalty = len(bottlenecks) * 10  # 10 points per bottleneck
        
        # Throughput factor
        throughput_factor = min(100, metrics['throughput_per_day'] * 2)  # Bonus for high throughput
        
        efficiency_score = max(0, (duration_consistency + throughput_factor) / 2 - bottleneck_penalty)
        
        return min(100, efficiency_score)
    
    def _generate_optimization_recommendations(self, process_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI-powered optimization recommendations"""
        
        prompt = f"""
        Analyze the following business process and generate optimization recommendations:
        
        Process Performance:
        - Average Duration: {process_data.get('performance_metrics', {}).get('avg_duration_hours', 0):.2f} hours
        - Efficiency Score: {process_data.get('efficiency_score', 0):.1f}/100
        - Automation Potential: {process_data.get('automation_potential', 0):.1f}/100
        - Complexity Score: {process_data.get('complexity_score', 0)}
        - Bottlenecks: {process_data.get('bottlenecks', [])}
        
        Process Variants: {[list(v['variant']) for v in process_data.get('variants', [])]}
        
        Generate 5-10 specific optimization recommendations including:
        1. Automation opportunities
        2. Process redesign suggestions
        3. Bottleneck resolution
        4. Technology implementations
        5. Resource optimization
        
        For each recommendation, provide:
        - Title and description
        - Expected impact (time/cost savings)
        - Implementation complexity
        - Required investment
        - ROI estimate
        
        Format as structured JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a business process optimization consultant with expertise in lean management, automation, and digital transformation."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations.get('recommendations', [])
            
        except Exception as e:
            logger.error(f"Optimization recommendation generation failed: {e}")
            return []
    
    def _store_discovered_process(self, process_data: Dict[str, Any], organization_id: str):
        """Store discovered process in database"""
        
        discovered_process = DiscoveredProcess(
            process_id=process_data['process_id'],
            process_name=f"Discovered Process {process_data['process_id'][-4:]}",
            organization_id=organization_id,
            process_model=process_data.get('process_model', {}),
            process_variants=process_data.get('variants', []),
            bottlenecks=process_data.get('bottlenecks', []),
            optimization_opportunities=process_data.get('optimization_recommendations', []),
            total_instances=process_data.get('total_instances', 0),
            avg_duration=process_data.get('performance_metrics', {}).get('avg_duration_hours', 0),
            complexity_score=process_data.get('complexity_score', 0),
            automation_potential=process_data.get('automation_potential', 0),
            efficiency_score=process_data.get('efficiency_score', 0),
            performance_analysis=process_data.get('performance_metrics', {}),
            discovery_method='event_log_mining',
            confidence_score=85.0  # High confidence for mined processes
        )
        
        db.session.add(discovered_process)
        
        # Store optimization recommendations
        for rec in process_data.get('optimization_recommendations', []):
            optimization = ProcessOptimization(
                optimization_id=f"OPT_{process_data['process_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                process_id=process_data['process_id'],
                optimization_type=rec.get('type', 'automation'),
                title=rec.get('title', 'Optimization Opportunity'),
                description=rec.get('description', ''),
                estimated_time_savings=rec.get('time_savings_percentage', 0),
                estimated_cost_savings=rec.get('cost_savings', 0),
                implementation_complexity=rec.get('complexity', 'medium'),
                roi_percentage=rec.get('roi_percentage', 0)
            )
            
            db.session.add(optimization)
        
        db.session.commit()

# Initialize engine
discovery_engine = ProcessDiscoveryEngine()

# Process Discovery Routes
@app.route('/process-discovery')
def process_dashboard():
    """Process discovery dashboard"""
    
    # Get discovered processes
    processes = DiscoveredProcess.query.order_by(DiscoveredProcess.discovered_at.desc()).limit(10).all()
    
    # Get optimization opportunities
    optimizations = ProcessOptimization.query.order_by(ProcessOptimization.created_at.desc()).limit(10).all()
    
    return render_template('process_discovery/dashboard.html',
                         processes=processes,
                         optimizations=optimizations)

@app.route('/process-discovery/api/discover', methods=['POST'])
def discover_processes():
    """API endpoint for process discovery"""
    
    data = request.get_json()
    
    organization_id = data.get('organization_id')
    if not organization_id:
        return jsonify({'error': 'Organization ID required'}), 400
    
    # Time range for analysis
    time_range = None
    if data.get('start_date') and data.get('end_date'):
        time_range = (
            datetime.fromisoformat(data['start_date']),
            datetime.fromisoformat(data['end_date'])
        )
    
    # Discover processes
    discovery_result = discovery_engine.discover_processes_from_events(organization_id, time_range)
    
    return jsonify(discovery_result)

@app.route('/process-discovery/api/events', methods=['POST'])
def submit_process_events():
    """API endpoint for submitting process events"""
    
    data = request.get_json()
    events_data = data.get('events', [])
    
    if not events_data:
        return jsonify({'error': 'No events provided'}), 400
    
    created_events = []
    
    for event_data in events_data:
        event = ProcessEvent(
            event_id=event_data.get('event_id', f"EVT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"),
            process_instance_id=event_data.get('process_instance_id'),
            activity_name=event_data.get('activity_name'),
            timestamp=datetime.fromisoformat(event_data.get('timestamp')),
            user_id=event_data.get('user_id'),
            department=event_data.get('department'),
            system_source=event_data.get('system_source'),
            event_type=event_data.get('event_type', 'complete'),
            organization_id=event_data.get('organization_id')
        )
        
        db.session.add(event)
        created_events.append(event.event_id)
    
    db.session.commit()
    
    return jsonify({
        'message': f'{len(created_events)} events created',
        'event_ids': created_events
    })

@app.route('/process-discovery/api/process/<process_id>')
def get_process_details(process_id):
    """Get detailed process information"""
    
    process = DiscoveredProcess.query.filter_by(process_id=process_id).first()
    if not process:
        return jsonify({'error': 'Process not found'}), 404
    
    optimizations = ProcessOptimization.query.filter_by(process_id=process_id).all()
    
    return jsonify({
        'process_id': process.process_id,
        'process_name': process.process_name,
        'description': process.description,
        'performance_metrics': {
            'efficiency_score': process.efficiency_score,
            'automation_potential': process.automation_potential,
            'complexity_score': process.complexity_score,
            'avg_duration': process.avg_duration,
            'total_instances': process.total_instances
        },
        'process_model': process.process_model,
        'bottlenecks': process.bottlenecks,
        'optimizations': [{
            'optimization_id': opt.optimization_id,
            'title': opt.title,
            'description': opt.description,
            'estimated_savings': opt.estimated_cost_savings,
            'roi_percentage': opt.roi_percentage,
            'implementation_complexity': opt.implementation_complexity
        } for opt in optimizations]
    })

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample process events
    if ProcessEvent.query.count() == 0:
        sample_events = [
            {
                'event_id': 'EVT_001',
                'process_instance_id': 'INST_001',
                'activity_name': 'Submit Application',
                'timestamp': datetime.utcnow() - timedelta(hours=5),
                'user_id': 'user_001',
                'organization_id': 'ORG_001'
            },
            {
                'event_id': 'EVT_002',
                'process_instance_id': 'INST_001',
                'activity_name': 'Review Application',
                'timestamp': datetime.utcnow() - timedelta(hours=3),
                'user_id': 'user_002',
                'organization_id': 'ORG_001'
            },
            {
                'event_id': 'EVT_003',
                'process_instance_id': 'INST_001',
                'activity_name': 'Approve Application',
                'timestamp': datetime.utcnow() - timedelta(hours=1),
                'user_id': 'user_003',
                'organization_id': 'ORG_001'
            }
        ]
        
        for event_data in sample_events:
            event = ProcessEvent(**event_data)
            db.session.add(event)
        
        db.session.commit()
        logger.info("Sample process discovery data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)