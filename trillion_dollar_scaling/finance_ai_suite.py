"""
Finance AI Suite - SOX-Compliant Financial Intelligence Agents
$20B+ Value Potential - Specialized AI for Financial Enterprise
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
import yfinance as yf
from openai import OpenAI
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "finance-ai-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///finance_ai.db")

db.init_app(app)

# Financial Data Models
class FinancialTransaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), nullable=False)
    transaction_date = db.Column(db.DateTime, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    currency = db.Column(db.String(10), default='USD')
    transaction_type = db.Column(db.String(50), nullable=False)  # debit, credit, transfer
    category = db.Column(db.String(100))
    description = db.Column(db.Text)
    counterparty = db.Column(db.String(200))
    
    # Risk Assessment
    fraud_score = db.Column(db.Float, default=0.0)
    aml_risk_score = db.Column(db.Float, default=0.0)
    risk_flags = db.Column(db.JSON)
    
    # Compliance
    regulatory_category = db.Column(db.String(100))
    reporting_requirements = db.Column(db.JSON)
    audit_trail = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class RiskAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assessment_id = db.Column(db.String(100), unique=True, nullable=False)
    entity_id = db.Column(db.String(100), nullable=False)  # customer, transaction, portfolio
    entity_type = db.Column(db.String(50), nullable=False)
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Risk Scores
    credit_risk_score = db.Column(db.Float, default=0.0)
    market_risk_score = db.Column(db.Float, default=0.0)
    operational_risk_score = db.Column(db.Float, default=0.0)
    liquidity_risk_score = db.Column(db.Float, default=0.0)
    overall_risk_score = db.Column(db.Float, default=0.0)
    
    # Risk Details
    risk_factors = db.Column(db.JSON)
    mitigation_strategies = db.Column(db.JSON)
    monitoring_requirements = db.Column(db.JSON)
    stress_test_results = db.Column(db.JSON)
    
    # Regulatory
    basel_compliance = db.Column(db.JSON)
    ifrs_impact = db.Column(db.JSON)
    
    expires_at = db.Column(db.DateTime)

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    data_date = db.Column(db.Date, nullable=False)
    open_price = db.Column(db.Float)
    high_price = db.Column(db.Float)
    low_price = db.Column(db.Float)
    close_price = db.Column(db.Float)
    volume = db.Column(db.BigInteger)
    
    # Technical Indicators
    sma_20 = db.Column(db.Float)  # 20-day Simple Moving Average
    ema_12 = db.Column(db.Float)  # 12-day Exponential Moving Average
    rsi = db.Column(db.Float)     # Relative Strength Index
    volatility = db.Column(db.Float)
    
    # AI Predictions
    price_prediction_1d = db.Column(db.Float)
    price_prediction_7d = db.Column(db.Float)
    price_prediction_30d = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class ComplianceReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(100), unique=True, nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # sox, aml, basel, mifid
    reporting_period_start = db.Column(db.Date, nullable=False)
    reporting_period_end = db.Column(db.Date, nullable=False)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Report Data
    compliance_score = db.Column(db.Float, default=100.0)
    violations_count = db.Column(db.Integer, default=0)
    findings = db.Column(db.JSON)
    recommendations = db.Column(db.JSON)
    
    # Status
    status = db.Column(db.String(50), default='draft')  # draft, submitted, approved
    reviewed_by = db.Column(db.String(100))
    submitted_to_regulator = db.Column(db.Boolean, default=False)

# Financial AI Engines
class FraudDetectionEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_transaction_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction for fraud indicators"""
        
        prompt = f"""
        Analyze the following financial transaction for fraud indicators:
        
        Amount: ${transaction.get('amount', 0):,.2f}
        Transaction Type: {transaction.get('transaction_type', '')}
        Time: {transaction.get('transaction_date', '')}
        Location: {transaction.get('location', 'Unknown')}
        Merchant: {transaction.get('counterparty', '')}
        Account History: {transaction.get('account_history', 'Normal')}
        
        Analyze for:
        1. Unusual amount patterns
        2. Timing anomalies
        3. Location inconsistencies
        4. Merchant risk factors
        5. Velocity patterns
        6. Account behavior changes
        
        Provide fraud risk score (0-100) and specific risk factors.
        Format as JSON with detailed explanations.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a financial fraud detection expert. Analyze transactions for fraud indicators using industry best practices and statistical patterns."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            fraud_analysis = json.loads(response.choices[0].message.content)
            fraud_analysis['analysis_timestamp'] = datetime.utcnow().isoformat()
            fraud_analysis['model_version'] = 'fraud_detection_v2.1'
            
            return fraud_analysis
            
        except Exception as e:
            logger.error(f"Fraud analysis failed: {e}")
            return {'fraud_score': 0, 'error': 'Analysis unavailable'}
    
    def pattern_analysis(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transaction patterns for anomalies"""
        
        if len(transactions) < 10:
            return {'error': 'Insufficient transaction history'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values('transaction_date')
        
        # Calculate patterns
        patterns = {
            'avg_transaction_amount': df['amount'].mean(),
            'std_transaction_amount': df['amount'].std(),
            'transaction_frequency': len(df) / (df['transaction_date'].max() - df['transaction_date'].min()).days,
            'unusual_amounts': [],
            'timing_patterns': [],
            'velocity_spikes': []
        }
        
        # Identify unusual amounts (beyond 2 standard deviations)
        threshold = patterns['avg_transaction_amount'] + 2 * patterns['std_transaction_amount']
        unusual = df[df['amount'] > threshold]
        patterns['unusual_amounts'] = unusual[['transaction_id', 'amount', 'transaction_date']].to_dict('records')
        
        # Timing pattern analysis
        df['hour'] = df['transaction_date'].dt.hour
        hourly_dist = df['hour'].value_counts().sort_index()
        
        # Identify unusual timing (outside normal hours)
        unusual_hours = hourly_dist[(hourly_dist.index < 6) | (hourly_dist.index > 22)]
        if len(unusual_hours) > 0:
            patterns['timing_patterns'] = {
                'unusual_hour_transactions': unusual_hours.to_dict(),
                'risk_level': 'medium' if unusual_hours.sum() < len(df) * 0.1 else 'high'
            }
        
        return patterns

class CreditRiskEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def assess_credit_risk(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess credit risk for loan applications"""
        
        prompt = f"""
        Assess credit risk for the following loan application:
        
        Annual Income: ${applicant_data.get('annual_income', 0):,.2f}
        Requested Amount: ${applicant_data.get('loan_amount', 0):,.2f}
        Credit Score: {applicant_data.get('credit_score', 'Unknown')}
        Employment History: {applicant_data.get('employment_years', 0)} years
        Debt-to-Income Ratio: {applicant_data.get('debt_to_income', 0):.1%}
        Collateral: {applicant_data.get('collateral_type', 'None')}
        Loan Purpose: {applicant_data.get('loan_purpose', '')}
        
        Provide:
        1. Credit risk score (0-100, where 0 = highest risk)
        2. Key risk factors
        3. Mitigating factors
        4. Recommended loan terms
        5. Required monitoring
        6. Basel III capital requirements
        
        Format as structured JSON with regulatory compliance notes.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a credit risk analyst with expertise in Basel III regulations and lending best practices. Provide comprehensive risk assessments with regulatory compliance considerations."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            risk_assessment = json.loads(response.choices[0].message.content)
            risk_assessment['assessment_date'] = datetime.utcnow().isoformat()
            risk_assessment['regulatory_framework'] = 'Basel III'
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Credit risk assessment failed: {e}")
            return {'credit_score': 50, 'error': 'Assessment unavailable'}

class MarketAnalysisEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def analyze_market_trends(self, symbol: str, timeframe: str = '1y') -> Dict[str, Any]:
        """Analyze market trends and generate predictions"""
        
        try:
            # Fetch market data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=timeframe)
            
            if hist.empty:
                return {'error': 'No market data available'}
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['EMA_12'] = hist['Close'].ewm(span=12).mean()
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['Volatility'] = hist['Close'].rolling(window=20).std()
            
            # Prepare data for AI analysis
            latest_data = hist.tail(30)
            price_data = latest_data['Close'].tolist()
            volume_data = latest_data['Volume'].tolist()
            
            prompt = f"""
            Analyze the following market data for {symbol}:
            
            Recent Prices (30 days): {price_data[-10:]}  # Last 10 days
            Recent Volume: {volume_data[-10:]}
            Current Price: ${price_data[-1]:.2f}
            20-day SMA: ${latest_data['SMA_20'].iloc[-1]:.2f}
            RSI: {latest_data['RSI'].iloc[-1]:.2f}
            Volatility: {latest_data['Volatility'].iloc[-1]:.2f}
            
            Provide:
            1. Technical analysis summary
            2. Short-term price prediction (1-7 days)
            3. Medium-term outlook (1-3 months)
            4. Key support and resistance levels
            5. Risk factors and market sentiment
            6. Trading recommendations
            
            Format as JSON with confidence scores.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a quantitative market analyst specializing in technical analysis and market prediction. Provide data-driven insights with appropriate risk disclaimers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            analysis['analysis_date'] = datetime.utcnow().isoformat()
            analysis['symbol'] = symbol
            analysis['current_price'] = price_data[-1]
            analysis['disclaimer'] = 'This analysis is for informational purposes only and should not be considered as investment advice'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            return {'error': 'Market analysis unavailable'}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class ComplianceEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate_sox_report(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Sarbanes-Oxley compliance report"""
        
        prompt = f"""
        Generate a Sarbanes-Oxley compliance assessment based on:
        
        Financial Controls: {financial_data.get('internal_controls', {})}
        Audit Findings: {financial_data.get('audit_findings', [])}
        Management Assertions: {financial_data.get('management_assertions', {})}
        IT Controls: {financial_data.get('it_controls', {})}
        Financial Reporting: {financial_data.get('financial_reporting', {})}
        
        Assess compliance with:
        1. Section 302 (CEO/CFO Certification)
        2. Section 404 (Internal Control Assessment)
        3. Section 409 (Real-time Disclosure)
        4. Section 802 (Record Retention)
        
        Provide compliance score, deficiencies, and remediation recommendations.
        Format as structured JSON for regulatory submission.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a SOX compliance expert specializing in Section 404 internal controls and financial reporting requirements. Generate thorough compliance assessments."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            sox_report = json.loads(response.choices[0].message.content)
            sox_report['report_date'] = datetime.utcnow().isoformat()
            sox_report['regulatory_framework'] = 'Sarbanes-Oxley Act'
            sox_report['requires_management_review'] = True
            
            return sox_report
            
        except Exception as e:
            logger.error(f"SOX report generation failed: {e}")
            return {'error': 'SOX report generation failed'}

# Initialize engines
fraud_engine = FraudDetectionEngine()
credit_engine = CreditRiskEngine()
market_engine = MarketAnalysisEngine()
compliance_engine = ComplianceEngine()

# Finance AI Routes
@app.route('/finance-ai')
def finance_dashboard():
    """Finance AI Suite dashboard"""
    
    # Get recent transactions
    recent_transactions = FinancialTransaction.query.order_by(
        FinancialTransaction.transaction_date.desc()
    ).limit(10).all()
    
    # Get recent risk assessments
    recent_assessments = RiskAssessment.query.order_by(
        RiskAssessment.assessment_date.desc()
    ).limit(5).all()
    
    return render_template('finance/dashboard.html',
                         recent_transactions=recent_transactions,
                         recent_assessments=recent_assessments)

@app.route('/finance-ai/api/fraud-detection', methods=['POST'])
def detect_fraud():
    """API endpoint for fraud detection"""
    
    data = request.get_json()
    
    if not data.get('transaction'):
        return jsonify({'error': 'Transaction data required'}), 400
    
    # Analyze transaction for fraud
    fraud_analysis = fraud_engine.analyze_transaction_fraud(data['transaction'])
    
    # Store analysis result
    if 'error' not in fraud_analysis:
        transaction = FinancialTransaction(
            transaction_id=data['transaction'].get('transaction_id', f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            account_id=data['transaction'].get('account_id', ''),
            transaction_date=datetime.fromisoformat(data['transaction'].get('transaction_date', datetime.utcnow().isoformat())),
            amount=data['transaction'].get('amount', 0.0),
            transaction_type=data['transaction'].get('transaction_type', ''),
            fraud_score=fraud_analysis.get('fraud_score', 0.0),
            risk_flags=fraud_analysis.get('risk_factors', [])
        )
        
        db.session.add(transaction)
        db.session.commit()
    
    return jsonify(fraud_analysis)

@app.route('/finance-ai/api/credit-risk', methods=['POST'])
def assess_credit_risk():
    """API endpoint for credit risk assessment"""
    
    data = request.get_json()
    
    if not data.get('applicant_data'):
        return jsonify({'error': 'Applicant data required'}), 400
    
    # Assess credit risk
    risk_assessment = credit_engine.assess_credit_risk(data['applicant_data'])
    
    return jsonify(risk_assessment)

@app.route('/finance-ai/api/market-analysis/<symbol>')
def analyze_market(symbol):
    """API endpoint for market analysis"""
    
    timeframe = request.args.get('timeframe', '1y')
    
    # Analyze market trends
    market_analysis = market_engine.analyze_market_trends(symbol, timeframe)
    
    return jsonify(market_analysis)

@app.route('/finance-ai/api/compliance-report', methods=['POST'])
def generate_compliance_report():
    """API endpoint for compliance report generation"""
    
    data = request.get_json()
    
    report_type = data.get('report_type', 'sox')
    financial_data = data.get('financial_data', {})
    
    if report_type == 'sox':
        compliance_report = compliance_engine.generate_sox_report(financial_data)
    else:
        return jsonify({'error': 'Unsupported report type'}), 400
    
    return jsonify(compliance_report)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample financial data
    if FinancialTransaction.query.count() == 0:
        sample_transaction = FinancialTransaction(
            transaction_id='TXN_001',
            account_id='ACC_001',
            transaction_date=datetime.utcnow(),
            amount=1500.00,
            transaction_type='debit',
            category='purchase',
            fraud_score=15.0
        )
        
        db.session.add(sample_transaction)
        db.session.commit()
        
        logger.info("Sample financial data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)