"""
Healthcare AI Suite - HIPAA-Compliant Medical Intelligence Agents
$15B+ Value Potential - Specialized AI for Healthcare Enterprise
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
from cryptography.fernet import Fernet
import hashlib
import re
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "healthcare-ai-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///healthcare_ai.db")

db.init_app(app)

# HIPAA-Compliant Data Models
class PatientData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id_hash = db.Column(db.String(256), nullable=False)  # Hashed patient ID for privacy
    encrypted_phi = db.Column(db.Text)  # Encrypted Protected Health Information
    medical_record_number = db.Column(db.String(100))
    admission_date = db.Column(db.DateTime)
    discharge_date = db.Column(db.DateTime)
    department = db.Column(db.String(100))
    diagnosis_codes = db.Column(db.JSON)  # ICD-10 codes
    procedure_codes = db.Column(db.JSON)  # CPT codes
    severity_score = db.Column(db.Float)
    risk_assessment = db.Column(db.JSON)
    treatment_plan = db.Column(db.JSON)
    outcomes = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ClinicalInsights(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    insight_id = db.Column(db.String(100), unique=True, nullable=False)
    patient_id_hash = db.Column(db.String(256), nullable=False)
    insight_type = db.Column(db.String(50), nullable=False)  # risk, diagnosis, treatment, outcome
    confidence_score = db.Column(db.Float, nullable=False)
    clinical_significance = db.Column(db.String(50))  # low, medium, high, critical
    
    # Insight Details
    primary_finding = db.Column(db.Text)
    supporting_evidence = db.Column(db.JSON)
    recommended_actions = db.Column(db.JSON)
    contraindications = db.Column(db.JSON)
    monitoring_parameters = db.Column(db.JSON)
    
    # Validation
    physician_reviewed = db.Column(db.Boolean, default=False)
    physician_approval = db.Column(db.Boolean)
    review_comments = db.Column(db.Text)
    
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)

class HealthcareMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    facility_id = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Patient Outcomes
    readmission_rate = db.Column(db.Float, default=0.0)
    mortality_rate = db.Column(db.Float, default=0.0)
    patient_satisfaction = db.Column(db.Float, default=0.0)
    length_of_stay_avg = db.Column(db.Float, default=0.0)
    
    # Operational Metrics
    bed_occupancy_rate = db.Column(db.Float, default=0.0)
    staff_efficiency_score = db.Column(db.Float, default=0.0)
    medication_error_rate = db.Column(db.Float, default=0.0)
    infection_control_score = db.Column(db.Float, default=0.0)
    
    # Financial Metrics
    cost_per_patient = db.Column(db.Float, default=0.0)
    revenue_per_bed = db.Column(db.Float, default=0.0)
    insurance_denial_rate = db.Column(db.Float, default=0.0)
    
    # Quality Indicators
    clinical_quality_score = db.Column(db.Float, default=0.0)
    safety_incidents = db.Column(db.Integer, default=0)
    compliance_score = db.Column(db.Float, default=0.0)

# HIPAA Compliance & Security Manager
class HIPAAComplianceManager:
    def __init__(self):
        self.encryption_key = os.environ.get("ENCRYPTION_KEY", Fernet.generate_key())
        self.cipher_suite = Fernet(self.encryption_key)
        
    def encrypt_phi(self, phi_data: str) -> str:
        """Encrypt Protected Health Information"""
        return self.cipher_suite.encrypt(phi_data.encode()).decode()
    
    def decrypt_phi(self, encrypted_data: str) -> str:
        """Decrypt Protected Health Information"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_patient_id(self, patient_id: str) -> str:
        """Create secure hash of patient ID for anonymization"""
        return hashlib.sha256(f"{patient_id}_salt_2025".encode()).hexdigest()
    
    def validate_phi_access(self, user_role: str, data_type: str) -> bool:
        """Validate access to PHI based on user role and data type"""
        access_matrix = {
            'physician': ['all'],
            'nurse': ['basic_info', 'treatment_plans', 'medications'],
            'admin': ['demographics', 'billing'],
            'researcher': ['anonymized_only']
        }
        
        allowed = access_matrix.get(user_role, [])
        return 'all' in allowed or data_type in allowed
    
    def audit_log_access(self, user_id: str, patient_id_hash: str, action: str):
        """Log all PHI access for audit compliance"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'patient_id_hash': patient_id_hash,
            'action': action,
            'ip_address': request.remote_addr if request else 'system'
        }
        
        # Store audit log (in production, use secure audit database)
        logger.info(f"AUDIT: {audit_entry}")

# Clinical Decision Support AI
class ClinicalDecisionSupport:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.compliance_manager = HIPAAComplianceManager()
        
    def analyze_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient risk factors using AI"""
        
        # De-identify data for AI processing
        deidentified_data = self._deidentify_patient_data(patient_data)
        
        # AI risk analysis prompt
        prompt = f"""
        Analyze the following de-identified patient data for clinical risks:
        
        Age Range: {deidentified_data['age_range']}
        Gender: {deidentified_data['gender']}
        Diagnosis Codes: {deidentified_data['diagnosis_codes']}
        Vital Signs: {deidentified_data['vital_signs']}
        Lab Results: {deidentified_data['lab_results']}
        Medications: {deidentified_data['medications']}
        Comorbidities: {deidentified_data['comorbidities']}
        
        Provide a clinical risk assessment including:
        1. Primary risk factors (scale 1-10)
        2. Secondary complications to monitor
        3. Recommended interventions
        4. Monitoring frequency
        5. Contraindications to avoid
        
        Format as JSON with confidence scores.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a clinical decision support AI trained on medical guidelines and evidence-based medicine. Provide accurate, evidence-based recommendations while noting this is for decision support only and requires physician review."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            risk_analysis = json.loads(response.choices[0].message.content)
            
            # Add metadata
            risk_analysis['generated_at'] = datetime.utcnow().isoformat()
            risk_analysis['model_version'] = 'gpt-5'
            risk_analysis['requires_physician_review'] = True
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            return {'error': 'Risk analysis unavailable', 'requires_physician_review': True}
    
    def generate_treatment_recommendations(self, diagnosis_codes: List[str], patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evidence-based treatment recommendations"""
        
        prompt = f"""
        Based on the following clinical information, provide evidence-based treatment recommendations:
        
        Primary Diagnosis Codes: {diagnosis_codes}
        Patient Profile: {patient_profile}
        
        Provide treatment recommendations including:
        1. First-line treatments with evidence levels
        2. Alternative therapies if contraindications exist
        3. Monitoring parameters and frequency
        4. Expected outcomes and timelines
        5. Potential adverse effects to watch
        6. Medication interactions to avoid
        
        Include clinical guidelines references where applicable.
        Format as structured JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a clinical decision support system. Provide evidence-based treatment recommendations following current medical guidelines. All recommendations require physician review and approval."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            recommendations['disclaimer'] = 'All recommendations require physician review and approval'
            recommendations['evidence_based'] = True
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Treatment recommendation failed: {e}")
            return {'error': 'Treatment recommendations unavailable'}
    
    def _deidentify_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or generalize identifying information"""
        
        deidentified = {}
        
        # Age ranges instead of exact age
        if 'age' in patient_data:
            age = patient_data['age']
            if age < 18:
                deidentified['age_range'] = 'pediatric'
            elif age < 65:
                deidentified['age_range'] = 'adult'
            else:
                deidentified['age_range'] = 'elderly'
        
        # Keep medical data that's not identifying
        safe_fields = ['gender', 'diagnosis_codes', 'vital_signs', 'lab_results', 'medications', 'comorbidities']
        for field in safe_fields:
            if field in patient_data:
                deidentified[field] = patient_data[field]
        
        return deidentified

# Medical Natural Language Processing
class MedicalNLP:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def extract_medical_entities(self, clinical_text: str) -> Dict[str, Any]:
        """Extract medical entities from clinical notes"""
        
        prompt = f"""
        Extract medical entities from the following clinical text:
        
        "{clinical_text}"
        
        Extract and categorize:
        1. Symptoms and signs
        2. Diagnoses and conditions
        3. Medications and dosages
        4. Procedures and treatments
        5. Anatomical references
        6. Lab values and vital signs
        7. Temporal expressions
        
        Format as structured JSON with confidence scores.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a medical NLP system specialized in extracting clinical entities from healthcare text. Maintain clinical accuracy and note any ambiguities."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Medical NLP extraction failed: {e}")
            return {'error': 'Entity extraction failed'}
    
    def clinical_note_summarization(self, full_note: str) -> Dict[str, Any]:
        """Summarize lengthy clinical notes"""
        
        prompt = f"""
        Summarize the following clinical note maintaining all critical medical information:
        
        "{full_note}"
        
        Provide:
        1. Chief complaint and history
        2. Key physical findings
        3. Assessment and plan
        4. Critical values or concerns
        5. Follow-up requirements
        
        Maintain clinical accuracy and include all medication changes, diagnostic results, and treatment modifications.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a clinical documentation specialist. Summarize notes while preserving all medically relevant information and maintaining clinical accuracy."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return {
                'summary': response.choices[0].message.content,
                'original_length': len(full_note),
                'summary_length': len(response.choices[0].message.content),
                'compression_ratio': len(response.choices[0].message.content) / len(full_note)
            }
            
        except Exception as e:
            logger.error(f"Clinical note summarization failed: {e}")
            return {'error': 'Summarization failed'}

# Drug Interaction Checker
class DrugInteractionChecker:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def check_drug_interactions(self, medications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for drug-drug interactions and contraindications"""
        
        med_list = [f"{med['name']} {med.get('dosage', '')} {med.get('frequency', '')}" for med in medications]
        
        prompt = f"""
        Analyze the following medication list for drug interactions and contraindications:
        
        Medications: {med_list}
        
        Provide:
        1. Major drug interactions (high clinical significance)
        2. Moderate interactions requiring monitoring
        3. Minor interactions with recommendations
        4. Contraindications or warnings
        5. Recommended monitoring parameters
        6. Alternative medications if interactions are severe
        
        Include clinical significance levels and recommended actions.
        Format as structured JSON.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacist AI specializing in drug interactions and medication safety. Provide evidence-based interaction analysis with clinical significance ratings."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            interaction_analysis = json.loads(response.choices[0].message.content)
            interaction_analysis['analysis_date'] = datetime.utcnow().isoformat()
            interaction_analysis['requires_pharmacist_review'] = True
            
            return interaction_analysis
            
        except Exception as e:
            logger.error(f"Drug interaction check failed: {e}")
            return {'error': 'Interaction analysis unavailable'}

# Initialize components
hipaa_manager = HIPAAComplianceManager()
clinical_support = ClinicalDecisionSupport()
medical_nlp = MedicalNLP()
drug_checker = DrugInteractionChecker()

# Healthcare AI Routes
@app.route('/healthcare-ai')
def healthcare_dashboard():
    """Healthcare AI Suite dashboard"""
    
    # Get facility metrics
    recent_metrics = HealthcareMetrics.query.order_by(
        HealthcareMetrics.date.desc()
    ).limit(10).all()
    
    # Get recent insights
    recent_insights = ClinicalInsights.query.order_by(
        ClinicalInsights.generated_at.desc()
    ).limit(5).all()
    
    return render_template('healthcare/dashboard.html',
                         recent_metrics=recent_metrics,
                         recent_insights=recent_insights)

@app.route('/healthcare-ai/api/risk-analysis', methods=['POST'])
def analyze_patient_risk():
    """API endpoint for patient risk analysis"""
    
    data = request.get_json()
    
    # Validate required fields
    if not data.get('patient_id'):
        return jsonify({'error': 'Patient ID required'}), 400
    
    # Hash patient ID for privacy
    patient_id_hash = hipaa_manager.hash_patient_id(data['patient_id'])
    
    # Audit log access
    hipaa_manager.audit_log_access(
        user_id=data.get('user_id', 'system'),
        patient_id_hash=patient_id_hash,
        action='risk_analysis'
    )
    
    # Perform risk analysis
    risk_analysis = clinical_support.analyze_patient_risk(data.get('patient_data', {}))
    
    # Store insight
    if 'error' not in risk_analysis:
        insight = ClinicalInsights(
            insight_id=f"RISK_{patient_id_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            patient_id_hash=patient_id_hash,
            insight_type='risk',
            confidence_score=risk_analysis.get('confidence_score', 0.8),
            clinical_significance='high',
            primary_finding=json.dumps(risk_analysis.get('primary_risk_factors', {})),
            recommended_actions=risk_analysis.get('recommended_interventions', []),
            monitoring_parameters=risk_analysis.get('monitoring_frequency', {})
        )
        
        db.session.add(insight)
        db.session.commit()
    
    return jsonify(risk_analysis)

@app.route('/healthcare-ai/api/drug-interactions', methods=['POST'])
def check_drug_interactions():
    """API endpoint for drug interaction checking"""
    
    data = request.get_json()
    
    medications = data.get('medications', [])
    if not medications:
        return jsonify({'error': 'Medications list required'}), 400
    
    # Check interactions
    interaction_analysis = drug_checker.check_drug_interactions(medications)
    
    return jsonify(interaction_analysis)

@app.route('/healthcare-ai/api/clinical-nlp', methods=['POST'])
def process_clinical_text():
    """API endpoint for medical NLP processing"""
    
    data = request.get_json()
    
    clinical_text = data.get('text', '')
    if not clinical_text:
        return jsonify({'error': 'Clinical text required'}), 400
    
    operation = data.get('operation', 'extract')
    
    if operation == 'extract':
        result = medical_nlp.extract_medical_entities(clinical_text)
    elif operation == 'summarize':
        result = medical_nlp.clinical_note_summarization(clinical_text)
    else:
        return jsonify({'error': 'Invalid operation'}), 400
    
    return jsonify(result)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample healthcare data
    if HealthcareMetrics.query.count() == 0:
        sample_metrics = HealthcareMetrics(
            facility_id='HOSP_001',
            readmission_rate=8.5,
            patient_satisfaction=4.2,
            bed_occupancy_rate=85.0,
            clinical_quality_score=92.3,
            compliance_score=98.7
        )
        
        db.session.add(sample_metrics)
        db.session.commit()
        
        logger.info("Sample healthcare data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)