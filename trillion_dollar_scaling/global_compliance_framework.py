"""
Global Compliance Framework - Multi-Jurisdiction Automation System
$25B+ Value Potential - Universal Regulatory Compliance for Enterprise
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
from openai import OpenAI
import requests
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "compliance-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///compliance.db")

db.init_app(app)

# Compliance Enums
class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    PENDING = "pending"

class RegulatoryFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    BASEL_III = "basel_iii"
    MIFID_II = "mifid_ii"
    CCPA = "ccpa"
    PIPEDA = "pipeda"
    LGPD = "lgpd"

# Compliance Data Models
class RegulatoryRequirement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    requirement_id = db.Column(db.String(100), unique=True, nullable=False)
    framework = db.Column(db.Enum(RegulatoryFramework), nullable=False)
    jurisdiction = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    
    # Requirement Details
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text, nullable=False)
    legal_reference = db.Column(db.String(200))
    effective_date = db.Column(db.Date)
    last_updated = db.Column(db.Date)
    
    # Implementation Details
    implementation_guidance = db.Column(db.Text)
    evidence_required = db.Column(db.JSON)
    automated_checks = db.Column(db.JSON)
    manual_review_required = db.Column(db.Boolean, default=False)
    
    # Risk Assessment
    compliance_risk = db.Column(db.String(50))  # low, medium, high, critical
    penalty_severity = db.Column(db.String(50))
    max_penalty_amount = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ComplianceAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), nullable=False)
    framework = db.Column(db.Enum(RegulatoryFramework), nullable=False)
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Assessment Results
    overall_status = db.Column(db.Enum(ComplianceStatus), default=ComplianceStatus.PENDING)
    compliance_score = db.Column(db.Float, default=0.0)
    requirements_total = db.Column(db.Integer, default=0)
    requirements_compliant = db.Column(db.Integer, default=0)
    requirements_non_compliant = db.Column(db.Integer, default=0)
    
    # Findings
    critical_findings = db.Column(db.JSON)
    high_findings = db.Column(db.JSON)
    medium_findings = db.Column(db.JSON)
    low_findings = db.Column(db.JSON)
    
    # Remediation
    remediation_plan = db.Column(db.JSON)
    estimated_cost = db.Column(db.Float)
    estimated_timeline = db.Column(db.Integer)  # days
    
    # Validation
    assessed_by = db.Column(db.String(100))
    reviewed_by = db.Column(db.String(100))
    approved_by = db.Column(db.String(100))
    
    expires_at = db.Column(db.DateTime)

class ComplianceEvidence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    evidence_id = db.Column(db.String(100), unique=True, nullable=False)
    requirement_id = db.Column(db.String(100), db.ForeignKey('regulatory_requirement.requirement_id'), nullable=False)
    organization_id = db.Column(db.String(100), nullable=False)
    
    # Evidence Details
    evidence_type = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500))
    
    # Validation
    validation_status = db.Column(db.Enum(ComplianceStatus), default=ComplianceStatus.PENDING)
    validation_notes = db.Column(db.Text)
    validated_by = db.Column(db.String(100))
    validated_at = db.Column(db.DateTime)
    
    # Metadata
    created_by = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

class AuditTrail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(100), unique=True, nullable=False)
    organization_id = db.Column(db.String(100), nullable=False)
    
    # Event Details
    event_type = db.Column(db.String(100), nullable=False)
    event_description = db.Column(db.Text, nullable=False)
    affected_systems = db.Column(db.JSON)
    affected_data = db.Column(db.JSON)
    
    # User Information
    user_id = db.Column(db.String(100))
    user_role = db.Column(db.String(100))
    ip_address = db.Column(db.String(50))
    user_agent = db.Column(db.String(500))
    
    # Compliance Context
    compliance_frameworks = db.Column(db.JSON)
    risk_level = db.Column(db.String(50))
    
    # Metadata
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    correlation_id = db.Column(db.String(100))

# Global Compliance Engine
class GlobalComplianceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Framework mapping
        self.framework_jurisdictions = {
            RegulatoryFramework.GDPR: ['EU', 'EEA'],
            RegulatoryFramework.HIPAA: ['USA'],
            RegulatoryFramework.SOX: ['USA'],
            RegulatoryFramework.PCI_DSS: ['Global'],
            RegulatoryFramework.ISO_27001: ['Global'],
            RegulatoryFramework.BASEL_III: ['Global'],
            RegulatoryFramework.MIFID_II: ['EU'],
            RegulatoryFramework.CCPA: ['California, USA'],
            RegulatoryFramework.PIPEDA: ['Canada'],
            RegulatoryFramework.LGPD: ['Brazil']
        }
        
    def assess_organization_compliance(self, organization_id: str, frameworks: List[RegulatoryFramework]) -> Dict[str, Any]:
        """Comprehensive compliance assessment for organization"""
        
        assessment_results = {}
        
        for framework in frameworks:
            result = self._assess_framework_compliance(organization_id, framework)
            assessment_results[framework.value] = result
        
        # Generate overall compliance summary
        overall_summary = self._generate_compliance_summary(assessment_results)
        
        return {
            'organization_id': organization_id,
            'assessment_date': datetime.utcnow().isoformat(),
            'frameworks_assessed': [f.value for f in frameworks],
            'detailed_results': assessment_results,
            'overall_summary': overall_summary
        }
    
    def _assess_framework_compliance(self, organization_id: str, framework: RegulatoryFramework) -> Dict[str, Any]:
        """Assess compliance for specific regulatory framework"""
        
        # Get all requirements for framework
        requirements = RegulatoryRequirement.query.filter_by(framework=framework).all()
        
        if not requirements:
            return {'error': f'No requirements found for {framework.value}'}
        
        compliance_results = []
        total_score = 0
        compliant_count = 0
        
        for requirement in requirements:
            result = self._check_requirement_compliance(organization_id, requirement)
            compliance_results.append(result)
            
            total_score += result['compliance_score']
            if result['status'] == ComplianceStatus.COMPLIANT.value:
                compliant_count += 1
        
        # Calculate overall framework compliance
        overall_score = total_score / len(requirements) if requirements else 0
        compliance_percentage = (compliant_count / len(requirements)) * 100 if requirements else 0
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(compliance_results, framework)
        
        # Store assessment
        assessment = ComplianceAssessment(
            assessment_id=f"ASSESS_{organization_id}_{framework.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            organization_id=organization_id,
            framework=framework,
            compliance_score=overall_score,
            requirements_total=len(requirements),
            requirements_compliant=compliant_count,
            requirements_non_compliant=len(requirements) - compliant_count,
            remediation_plan=remediation_plan
        )
        
        db.session.add(assessment)
        db.session.commit()
        
        return {
            'assessment_id': assessment.assessment_id,
            'framework': framework.value,
            'overall_score': overall_score,
            'compliance_percentage': compliance_percentage,
            'requirements_assessed': len(requirements),
            'compliant_requirements': compliant_count,
            'detailed_results': compliance_results,
            'remediation_plan': remediation_plan
        }
    
    def _check_requirement_compliance(self, organization_id: str, requirement: RegulatoryRequirement) -> Dict[str, Any]:
        """Check compliance for specific requirement"""
        
        # Get existing evidence
        evidence = ComplianceEvidence.query.filter_by(
            requirement_id=requirement.requirement_id,
            organization_id=organization_id
        ).all()
        
        # Automated compliance check using AI
        compliance_check = self._ai_compliance_analysis(requirement, evidence)
        
        return {
            'requirement_id': requirement.requirement_id,
            'title': requirement.title,
            'category': requirement.category,
            'status': compliance_check['status'],
            'compliance_score': compliance_check['score'],
            'evidence_count': len(evidence),
            'findings': compliance_check['findings'],
            'recommendations': compliance_check['recommendations'],
            'risk_level': requirement.compliance_risk
        }
    
    def _ai_compliance_analysis(self, requirement: RegulatoryRequirement, evidence: List[ComplianceEvidence]) -> Dict[str, Any]:
        """AI-powered compliance analysis"""
        
        evidence_summary = []
        for ev in evidence:
            evidence_summary.append({
                'type': ev.evidence_type,
                'title': ev.title,
                'status': ev.validation_status.value if ev.validation_status else 'pending'
            })
        
        prompt = f"""
        Analyze compliance status for the following regulatory requirement:
        
        Framework: {requirement.framework.value.upper()}
        Requirement: {requirement.title}
        Description: {requirement.description}
        Legal Reference: {requirement.legal_reference}
        
        Available Evidence: {evidence_summary}
        
        Assess:
        1. Compliance status (compliant/non_compliant/requires_review)
        2. Compliance score (0-100)
        3. Key findings and gaps
        4. Specific recommendations for improvement
        5. Evidence quality assessment
        
        Consider the legal requirements and industry best practices.
        Format as structured JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a regulatory compliance expert with deep knowledge of global regulatory frameworks. Provide accurate compliance assessments based on legal requirements."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            analysis['analysis_timestamp'] = datetime.utcnow().isoformat()
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI compliance analysis failed: {e}")
            return {
                'status': 'requires_review',
                'score': 50,
                'findings': ['AI analysis unavailable'],
                'recommendations': ['Manual review required']
            }
    
    def _generate_remediation_plan(self, compliance_results: List[Dict[str, Any]], framework: RegulatoryFramework) -> Dict[str, Any]:
        """Generate comprehensive remediation plan"""
        
        non_compliant = [r for r in compliance_results if r['status'] != 'compliant']
        
        if not non_compliant:
            return {'status': 'fully_compliant', 'actions': []}
        
        remediation_actions = []
        total_estimated_cost = 0
        max_timeline = 0
        
        for result in non_compliant:
            action = {
                'requirement_id': result['requirement_id'],
                'title': result['title'],
                'priority': self._calculate_priority(result),
                'actions': result.get('recommendations', []),
                'estimated_cost': self._estimate_remediation_cost(result),
                'estimated_timeline': self._estimate_timeline(result),
                'risk_level': result['risk_level']
            }
            
            remediation_actions.append(action)
            total_estimated_cost += action['estimated_cost']
            max_timeline = max(max_timeline, action['estimated_timeline'])
        
        return {
            'status': 'remediation_required',
            'total_actions': len(remediation_actions),
            'actions': remediation_actions,
            'estimated_total_cost': total_estimated_cost,
            'estimated_timeline': max_timeline,
            'framework': framework.value
        }
    
    def _calculate_priority(self, result: Dict[str, Any]) -> str:
        """Calculate remediation priority"""
        risk_level = result.get('risk_level', 'medium')
        compliance_score = result.get('compliance_score', 50)
        
        if risk_level in ['critical', 'high'] and compliance_score < 30:
            return 'critical'
        elif risk_level == 'high' or compliance_score < 50:
            return 'high'
        elif risk_level == 'medium' or compliance_score < 70:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_remediation_cost(self, result: Dict[str, Any]) -> float:
        """Estimate remediation cost"""
        base_costs = {
            'critical': 50000,
            'high': 25000,
            'medium': 10000,
            'low': 5000
        }
        
        risk_level = result.get('risk_level', 'medium')
        return base_costs.get(risk_level, 10000)
    
    def _estimate_timeline(self, result: Dict[str, Any]) -> int:
        """Estimate remediation timeline in days"""
        timelines = {
            'critical': 30,
            'high': 60,
            'medium': 90,
            'low': 120
        }
        
        risk_level = result.get('risk_level', 'medium')
        return timelines.get(risk_level, 90)
    
    def _generate_compliance_summary(self, assessment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall compliance summary"""
        
        if not assessment_results:
            return {'status': 'no_assessments'}
        
        total_frameworks = len(assessment_results)
        compliant_frameworks = sum(1 for r in assessment_results.values() 
                                 if r.get('compliance_percentage', 0) >= 90)
        
        overall_score = np.mean([r.get('overall_score', 0) for r in assessment_results.values()])
        
        # Identify highest priority items
        all_actions = []
        for result in assessment_results.values():
            if 'remediation_plan' in result and 'actions' in result['remediation_plan']:
                all_actions.extend(result['remediation_plan']['actions'])
        
        critical_actions = [a for a in all_actions if a.get('priority') == 'critical']
        
        return {
            'overall_compliance_score': overall_score,
            'frameworks_assessed': total_frameworks,
            'fully_compliant_frameworks': compliant_frameworks,
            'compliance_percentage': (compliant_frameworks / total_frameworks) * 100,
            'critical_actions_required': len(critical_actions),
            'total_remediation_actions': len(all_actions),
            'status': 'compliant' if overall_score >= 90 else 'requires_attention'
        }

# Audit Trail Manager
class AuditTrailManager:
    def __init__(self):
        pass
        
    def log_compliance_event(self, organization_id: str, event_type: str, event_data: Dict[str, Any]):
        """Log compliance-related event for audit trail"""
        
        audit_entry = AuditTrail(
            event_id=f"AUDIT_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            organization_id=organization_id,
            event_type=event_type,
            event_description=json.dumps(event_data),
            user_id=event_data.get('user_id', 'system'),
            timestamp=datetime.utcnow(),
            compliance_frameworks=event_data.get('frameworks', [])
        )
        
        db.session.add(audit_entry)
        db.session.commit()
        
        logger.info(f"Compliance event logged: {event_type} for {organization_id}")

# Initialize engines
compliance_engine = GlobalComplianceEngine()
audit_manager = AuditTrailManager()

# Compliance Routes
@app.route('/compliance')
def compliance_dashboard():
    """Global compliance dashboard"""
    
    # Get recent assessments
    recent_assessments = ComplianceAssessment.query.order_by(
        ComplianceAssessment.assessment_date.desc()
    ).limit(10).all()
    
    # Get framework statistics
    framework_stats = db.session.query(
        ComplianceAssessment.framework,
        db.func.count(ComplianceAssessment.id).label('count'),
        db.func.avg(ComplianceAssessment.compliance_score).label('avg_score')
    ).group_by(ComplianceAssessment.framework).all()
    
    return render_template('compliance/dashboard.html',
                         recent_assessments=recent_assessments,
                         framework_stats=framework_stats)

@app.route('/compliance/api/assess', methods=['POST'])
def assess_compliance():
    """API endpoint for compliance assessment"""
    
    data = request.get_json()
    
    organization_id = data.get('organization_id')
    frameworks = data.get('frameworks', [])
    
    if not organization_id or not frameworks:
        return jsonify({'error': 'Organization ID and frameworks required'}), 400
    
    # Convert framework strings to enums
    framework_enums = []
    for fw in frameworks:
        try:
            framework_enums.append(RegulatoryFramework(fw))
        except ValueError:
            return jsonify({'error': f'Invalid framework: {fw}'}), 400
    
    # Perform assessment
    assessment_result = compliance_engine.assess_organization_compliance(
        organization_id, framework_enums
    )
    
    # Log audit event
    audit_manager.log_compliance_event(
        organization_id,
        'compliance_assessment',
        {
            'frameworks': frameworks,
            'user_id': data.get('user_id', 'api'),
            'assessment_id': assessment_result.get('assessment_id')
        }
    )
    
    return jsonify(assessment_result)

@app.route('/compliance/api/evidence', methods=['POST'])
def submit_evidence():
    """API endpoint for submitting compliance evidence"""
    
    data = request.get_json()
    
    evidence = ComplianceEvidence(
        evidence_id=f"EVID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        requirement_id=data.get('requirement_id'),
        organization_id=data.get('organization_id'),
        evidence_type=data.get('type'),
        title=data.get('title'),
        description=data.get('description'),
        created_by=data.get('created_by', 'api')
    )
    
    db.session.add(evidence)
    db.session.commit()
    
    return jsonify({'evidence_id': evidence.evidence_id, 'status': 'submitted'})

@app.route('/compliance/api/requirements/<framework>')
def get_framework_requirements(framework):
    """Get all requirements for a regulatory framework"""
    
    try:
        framework_enum = RegulatoryFramework(framework)
    except ValueError:
        return jsonify({'error': 'Invalid framework'}), 400
    
    requirements = RegulatoryRequirement.query.filter_by(framework=framework_enum).all()
    
    requirements_data = []
    for req in requirements:
        requirements_data.append({
            'requirement_id': req.requirement_id,
            'title': req.title,
            'description': req.description,
            'category': req.category,
            'compliance_risk': req.compliance_risk,
            'legal_reference': req.legal_reference
        })
    
    return jsonify({
        'framework': framework,
        'requirements_count': len(requirements_data),
        'requirements': requirements_data
    })

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample regulatory requirements
    if RegulatoryRequirement.query.count() == 0:
        sample_requirements = [
            {
                'requirement_id': 'GDPR_ART_6',
                'framework': RegulatoryFramework.GDPR,
                'jurisdiction': 'EU',
                'category': 'data_processing',
                'title': 'Lawful basis for processing',
                'description': 'Processing shall be lawful only if at least one of the specified conditions applies',
                'legal_reference': 'GDPR Article 6',
                'compliance_risk': 'high',
                'max_penalty_amount': 20000000
            },
            {
                'requirement_id': 'HIPAA_164_308',
                'framework': RegulatoryFramework.HIPAA,
                'jurisdiction': 'USA',
                'category': 'administrative_safeguards',
                'title': 'Administrative safeguards',
                'description': 'Implement administrative safeguards to protect electronic PHI',
                'legal_reference': '45 CFR § 164.308',
                'compliance_risk': 'critical',
                'max_penalty_amount': 1500000
            },
            {
                'requirement_id': 'SOX_302',
                'framework': RegulatoryFramework.SOX,
                'jurisdiction': 'USA',
                'category': 'certification',
                'title': 'Corporate responsibility for financial reports',
                'description': 'CEO and CFO certification of financial reports',
                'legal_reference': 'SOX Section 302',
                'compliance_risk': 'critical',
                'max_penalty_amount': 5000000
            }
        ]
        
        for req_data in sample_requirements:
            requirement = RegulatoryRequirement(**req_data)
            db.session.add(requirement)
        
        db.session.commit()
        logger.info("Sample compliance requirements created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)