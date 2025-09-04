"""
Data Intelligence Suite - Enterprise Analytics & Benchmarking Platform
$30B+ Value Potential - Real-time Business Intelligence for Fortune 500
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import func, and_, or_
import plotly.graph_objs as go
import plotly.utils
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
import requests
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "intelligence-suite-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///intelligence.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

db.init_app(app)

# Database Models for Intelligence Suite
class EnterpriseMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.String(100), nullable=False)  # Anonymized company identifier
    industry = db.Column(db.String(100), nullable=False)
    company_size = db.Column(db.String(50), nullable=False)  # small, medium, large, enterprise
    revenue_range = db.Column(db.String(50), nullable=False)  # Revenue bucket for anonymization
    date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Operational Metrics
    automation_percentage = db.Column(db.Float, default=0.0)
    ai_adoption_score = db.Column(db.Float, default=0.0)
    digital_transformation_stage = db.Column(db.String(50))  # early, developing, advanced, leader
    employee_productivity_index = db.Column(db.Float, default=100.0)
    
    # Financial Metrics
    operational_efficiency_ratio = db.Column(db.Float, default=1.0)
    cost_reduction_percentage = db.Column(db.Float, default=0.0)
    roi_improvement = db.Column(db.Float, default=0.0)
    revenue_growth_rate = db.Column(db.Float, default=0.0)
    
    # Technology Metrics
    system_uptime_percentage = db.Column(db.Float, default=99.0)
    data_quality_score = db.Column(db.Float, default=0.0)
    security_incidents_count = db.Column(db.Integer, default=0)
    compliance_score = db.Column(db.Float, default=0.0)
    
    # Customer Metrics
    customer_satisfaction_score = db.Column(db.Float, default=0.0)
    net_promoter_score = db.Column(db.Float, default=0.0)
    customer_retention_rate = db.Column(db.Float, default=0.0)
    churn_rate = db.Column(db.Float, default=0.0)
    
    # Innovation Metrics
    innovation_index = db.Column(db.Float, default=0.0)
    time_to_market_days = db.Column(db.Integer, default=0)
    patent_applications = db.Column(db.Integer, default=0)
    rd_investment_percentage = db.Column(db.Float, default=0.0)

class BenchmarkingReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), nullable=False)
    industry = db.Column(db.String(100), nullable=False)
    generated_date = db.Column(db.DateTime, default=datetime.utcnow)
    report_type = db.Column(db.String(50), nullable=False)  # operational, financial, innovation
    
    # Benchmark Results
    overall_score = db.Column(db.Float, default=0.0)
    industry_percentile = db.Column(db.Float, default=0.0)
    peer_comparison = db.Column(db.JSON)  # Comparison with similar companies
    improvement_areas = db.Column(db.JSON)  # Areas for improvement
    strengths = db.Column(db.JSON)  # Company strengths
    recommendations = db.Column(db.JSON)  # AI-generated recommendations
    
    # Trend Analysis
    historical_performance = db.Column(db.JSON)
    projected_performance = db.Column(db.JSON)
    risk_assessment = db.Column(db.JSON)

class MarketIntelligence(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    industry = db.Column(db.String(100), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    
    # Market Data
    market_size_billions = db.Column(db.Float, default=0.0)
    growth_rate_percentage = db.Column(db.Float, default=0.0)
    market_volatility_index = db.Column(db.Float, default=0.0)
    competitive_intensity = db.Column(db.Float, default=0.0)
    
    # Economic Indicators
    gdp_correlation = db.Column(db.Float, default=0.0)
    employment_impact = db.Column(db.Float, default=0.0)
    inflation_sensitivity = db.Column(db.Float, default=0.0)
    interest_rate_sensitivity = db.Column(db.Float, default=0.0)
    
    # Technology Trends
    ai_adoption_rate = db.Column(db.Float, default=0.0)
    automation_potential = db.Column(db.Float, default=0.0)
    digital_disruption_risk = db.Column(db.Float, default=0.0)
    innovation_velocity = db.Column(db.Float, default=0.0)
    
    # Regulatory Environment
    regulatory_complexity_score = db.Column(db.Float, default=0.0)
    compliance_cost_impact = db.Column(db.Float, default=0.0)
    regulatory_change_frequency = db.Column(db.Float, default=0.0)

class PredictiveInsights(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    insight_id = db.Column(db.String(100), unique=True, nullable=False)
    company_id = db.Column(db.String(100), nullable=False)
    insight_type = db.Column(db.String(50), nullable=False)  # forecast, risk, opportunity
    generated_date = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float, default=0.0)
    
    # Prediction Details
    prediction_horizon_days = db.Column(db.Integer, default=90)
    predicted_metric = db.Column(db.String(100), nullable=False)
    current_value = db.Column(db.Float, default=0.0)
    predicted_value = db.Column(db.Float, default=0.0)
    probability_distribution = db.Column(db.JSON)
    
    # Supporting Data
    key_drivers = db.Column(db.JSON)  # Factors influencing prediction
    scenario_analysis = db.Column(db.JSON)  # Best/worst/likely scenarios
    recommended_actions = db.Column(db.JSON)  # Suggested interventions
    
    # Validation
    is_validated = db.Column(db.Boolean, default=False)
    actual_outcome = db.Column(db.Float)
    accuracy_score = db.Column(db.Float)

# Business Intelligence Engine
class IntelligenceEngine:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        
    def generate_company_benchmark(self, company_id: str, industry: str) -> Dict[str, Any]:
        """Generate comprehensive benchmarking report for a company"""
        
        # Get company's latest metrics
        company_metrics = EnterpriseMetrics.query.filter_by(
            company_id=company_id
        ).order_by(EnterpriseMetrics.date.desc()).first()
        
        if not company_metrics:
            return {'error': 'No metrics found for company'}
        
        # Get industry peers
        industry_peers = EnterpriseMetrics.query.filter(
            EnterpriseMetrics.industry == industry,
            EnterpriseMetrics.company_id != company_id,
            EnterpriseMetrics.company_size == company_metrics.company_size
        ).all()
        
        if not industry_peers:
            return {'error': 'No industry peers found for comparison'}
        
        # Calculate benchmarks
        benchmark_data = self._calculate_benchmarks(company_metrics, industry_peers)
        
        # Generate insights and recommendations
        insights = self._generate_insights(benchmark_data)
        
        # Create benchmark report
        report = BenchmarkingReport(
            report_id=f"BR_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            company_id=company_id,
            industry=industry,
            report_type='comprehensive',
            overall_score=benchmark_data['overall_score'],
            industry_percentile=benchmark_data['industry_percentile'],
            peer_comparison=benchmark_data['peer_comparison'],
            improvement_areas=insights['improvement_areas'],
            strengths=insights['strengths'],
            recommendations=insights['recommendations']
        )
        
        db.session.add(report)
        db.session.commit()
        
        return {
            'report_id': report.report_id,
            'benchmark_data': benchmark_data,
            'insights': insights
        }
    
    def _calculate_benchmarks(self, company: EnterpriseMetrics, peers: List[EnterpriseMetrics]) -> Dict[str, Any]:
        """Calculate benchmark scores against industry peers"""
        
        metrics = [
            'automation_percentage', 'ai_adoption_score', 'employee_productivity_index',
            'operational_efficiency_ratio', 'cost_reduction_percentage', 'roi_improvement',
            'system_uptime_percentage', 'data_quality_score', 'compliance_score',
            'customer_satisfaction_score', 'net_promoter_score', 'customer_retention_rate',
            'innovation_index', 'rd_investment_percentage'
        ]
        
        peer_data = []
        for peer in peers:
            peer_values = [getattr(peer, metric) for metric in metrics]
            peer_data.append(peer_values)
        
        if not peer_data:
            return {'error': 'Insufficient peer data'}
        
        peer_df = pd.DataFrame(peer_data, columns=metrics)
        company_values = [getattr(company, metric) for metric in metrics]
        
        # Calculate percentiles
        percentiles = {}
        scores = {}
        
        for i, metric in enumerate(metrics):
            peer_values = peer_df[metric].values
            company_value = company_values[i]
            
            # Calculate percentile rank
            percentile = (peer_values < company_value).sum() / len(peer_values) * 100
            percentiles[metric] = percentile
            
            # Convert to score (0-100)
            scores[metric] = min(100, max(0, percentile))
        
        # Calculate overall score (weighted average)
        weights = {
            'automation_percentage': 0.15,
            'ai_adoption_score': 0.15,
            'operational_efficiency_ratio': 0.12,
            'roi_improvement': 0.12,
            'customer_satisfaction_score': 0.10,
            'innovation_index': 0.10,
            'compliance_score': 0.08,
            'system_uptime_percentage': 0.08,
            'data_quality_score': 0.05,
            'employee_productivity_index': 0.05
        }
        
        overall_score = sum(scores[metric] * weights.get(metric, 0.01) for metric in metrics)
        industry_percentile = (peer_df.mean().sum() < sum(company_values)) / len(peers) * 100
        
        return {
            'overall_score': round(overall_score, 2),
            'industry_percentile': round(industry_percentile, 2),
            'metric_scores': scores,
            'metric_percentiles': percentiles,
            'peer_comparison': {
                'peer_averages': peer_df.mean().to_dict(),
                'peer_medians': peer_df.median().to_dict(),
                'company_values': dict(zip(metrics, company_values))
            }
        }
    
    def _generate_insights(self, benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered insights and recommendations"""
        
        scores = benchmark_data['metric_scores']
        
        # Identify strengths (top 25% of metrics)
        strengths = []
        improvement_areas = []
        
        for metric, score in scores.items():
            if score >= 75:
                strengths.append({
                    'metric': metric,
                    'score': score,
                    'description': self._get_metric_description(metric)
                })
            elif score < 50:
                improvement_areas.append({
                    'metric': metric,
                    'score': score,
                    'description': self._get_metric_description(metric),
                    'priority': 'high' if score < 25 else 'medium'
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(improvement_areas, strengths)
        
        return {
            'strengths': strengths,
            'improvement_areas': improvement_areas,
            'recommendations': recommendations
        }
    
    def _get_metric_description(self, metric: str) -> str:
        """Get human-readable description for metrics"""
        descriptions = {
            'automation_percentage': 'Process Automation Level',
            'ai_adoption_score': 'AI Technology Adoption',
            'employee_productivity_index': 'Employee Productivity',
            'operational_efficiency_ratio': 'Operational Efficiency',
            'cost_reduction_percentage': 'Cost Optimization',
            'roi_improvement': 'Return on Investment',
            'system_uptime_percentage': 'System Reliability',
            'data_quality_score': 'Data Quality Management',
            'compliance_score': 'Regulatory Compliance',
            'customer_satisfaction_score': 'Customer Satisfaction',
            'net_promoter_score': 'Customer Loyalty',
            'customer_retention_rate': 'Customer Retention',
            'innovation_index': 'Innovation Capability',
            'rd_investment_percentage': 'R&D Investment Level'
        }
        return descriptions.get(metric, metric.replace('_', ' ').title())
    
    def _generate_recommendations(self, improvement_areas: List[Dict], strengths: List[Dict]) -> List[Dict]:
        """Generate AI-powered recommendations"""
        
        recommendations = []
        
        for area in improvement_areas:
            metric = area['metric']
            priority = area['priority']
            
            if metric == 'automation_percentage':
                recommendations.append({
                    'title': 'Accelerate Process Automation',
                    'description': 'Implement RPA and AI agents to automate repetitive tasks',
                    'impact': 'High',
                    'effort': 'Medium',
                    'timeline': '3-6 months',
                    'roi_estimate': '200-400%'
                })
            elif metric == 'ai_adoption_score':
                recommendations.append({
                    'title': 'Enhance AI Strategy',
                    'description': 'Deploy enterprise AI agents and machine learning models',
                    'impact': 'High',
                    'effort': 'High',
                    'timeline': '6-12 months',
                    'roi_estimate': '300-600%'
                })
            elif metric == 'customer_satisfaction_score':
                recommendations.append({
                    'title': 'Improve Customer Experience',
                    'description': 'Implement AI-powered customer service and personalization',
                    'impact': 'Medium',
                    'effort': 'Medium',
                    'timeline': '2-4 months',
                    'roi_estimate': '150-300%'
                })
        
        return recommendations
    
    def generate_market_forecast(self, industry: str, horizon_days: int = 90) -> Dict[str, Any]:
        """Generate market intelligence and forecasting"""
        
        # Get historical market data
        historical_data = MarketIntelligence.query.filter_by(industry=industry).order_by(
            MarketIntelligence.date.desc()
        ).limit(365).all()
        
        if len(historical_data) < 30:
            return {'error': 'Insufficient historical data for forecasting'}
        
        # Prepare data for analysis
        dates = [d.date for d in historical_data]
        market_sizes = [d.market_size_billions for d in historical_data]
        growth_rates = [d.growth_rate_percentage for d in historical_data]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'market_size': market_sizes,
            'growth_rate': growth_rates
        })
        df = df.sort_values('date')
        
        # Generate forecasts using multiple models
        forecast_results = self._generate_forecasts(df, horizon_days)
        
        # Generate insights
        insights = self._analyze_market_trends(df, forecast_results)
        
        return {
            'industry': industry,
            'forecast_horizon': horizon_days,
            'current_market_size': market_sizes[0] if market_sizes else 0,
            'forecasts': forecast_results,
            'insights': insights,
            'confidence_score': forecast_results.get('confidence', 0.7)
        }
    
    def _generate_forecasts(self, df: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
        """Generate forecasts using multiple algorithms"""
        
        # Linear regression forecast
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['market_size'].values
        
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Random forest forecast
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Generate predictions
        future_X = np.arange(len(df), len(df) + horizon_days).reshape(-1, 1)
        lr_predictions = lr_model.predict(future_X)
        rf_predictions = rf_model.predict(future_X)
        
        # Ensemble forecast (average of models)
        ensemble_predictions = (lr_predictions + rf_predictions) / 2
        
        # Calculate confidence intervals
        prediction_std = np.std([lr_predictions, rf_predictions], axis=0)
        lower_bound = ensemble_predictions - 1.96 * prediction_std
        upper_bound = ensemble_predictions + 1.96 * prediction_std
        
        return {
            'linear_regression': lr_predictions.tolist(),
            'random_forest': rf_predictions.tolist(),
            'ensemble': ensemble_predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'confidence': 0.95 - np.mean(prediction_std) / np.mean(ensemble_predictions)
        }
    
    def _analyze_market_trends(self, df: pd.DataFrame, forecasts: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze market trends and generate insights"""
        
        insights = []
        
        # Trend analysis
        recent_growth = df['growth_rate'].tail(30).mean()
        historical_growth = df['growth_rate'].mean()
        
        if recent_growth > historical_growth * 1.1:
            insights.append({
                'type': 'trend',
                'title': 'Accelerating Growth',
                'description': f'Market growth rate has increased by {((recent_growth/historical_growth - 1) * 100):.1f}% above historical average',
                'impact': 'positive'
            })
        elif recent_growth < historical_growth * 0.9:
            insights.append({
                'type': 'trend',
                'title': 'Slowing Growth',
                'description': f'Market growth rate has decreased by {((1 - recent_growth/historical_growth) * 100):.1f}% below historical average',
                'impact': 'negative'
            })
        
        # Volatility analysis
        volatility = df['growth_rate'].std()
        if volatility > 5:
            insights.append({
                'type': 'risk',
                'title': 'High Market Volatility',
                'description': f'Market shows high volatility with standard deviation of {volatility:.1f}%',
                'impact': 'neutral'
            })
        
        # Forecast insights
        forecast_trend = np.mean(np.diff(forecasts['ensemble']))
        if forecast_trend > 0:
            insights.append({
                'type': 'forecast',
                'title': 'Positive Outlook',
                'description': f'Market projected to grow at average rate of {forecast_trend:.2f}B per period',
                'impact': 'positive'
            })
        
        return insights

# Initialize Intelligence Engine
intelligence_engine = IntelligenceEngine()

# Routes for Data Intelligence Suite
@app.route('/intelligence')
def intelligence_dashboard():
    """Main intelligence dashboard"""
    
    # Get overview statistics
    total_companies = db.session.query(func.count(func.distinct(EnterpriseMetrics.company_id))).scalar()
    total_reports = BenchmarkingReport.query.count()
    industries_covered = db.session.query(func.count(func.distinct(EnterpriseMetrics.industry))).scalar()
    
    # Get recent insights
    recent_insights = PredictiveInsights.query.order_by(
        PredictiveInsights.generated_date.desc()
    ).limit(5).all()
    
    # Get industry distribution
    industry_stats = db.session.query(
        EnterpriseMetrics.industry,
        func.count(func.distinct(EnterpriseMetrics.company_id)).label('company_count'),
        func.avg(EnterpriseMetrics.ai_adoption_score).label('avg_ai_adoption')
    ).group_by(EnterpriseMetrics.industry).all()
    
    return render_template('intelligence/dashboard.html',
                         total_companies=total_companies,
                         total_reports=total_reports,
                         industries_covered=industries_covered,
                         recent_insights=recent_insights,
                         industry_stats=industry_stats)

@app.route('/intelligence/benchmark/<company_id>')
def company_benchmark(company_id):
    """Generate benchmarking report for specific company"""
    
    # Get company info
    company = EnterpriseMetrics.query.filter_by(company_id=company_id).first()
    if not company:
        return jsonify({'error': 'Company not found'}), 404
    
    # Generate benchmark report
    benchmark_result = intelligence_engine.generate_company_benchmark(
        company_id, company.industry
    )
    
    if 'error' in benchmark_result:
        return jsonify(benchmark_result), 400
    
    return jsonify(benchmark_result)

@app.route('/intelligence/market-forecast/<industry>')
def market_forecast(industry):
    """Generate market forecast for industry"""
    
    horizon_days = request.args.get('horizon', 90, type=int)
    
    forecast_result = intelligence_engine.generate_market_forecast(industry, horizon_days)
    
    if 'error' in forecast_result:
        return jsonify(forecast_result), 400
    
    return jsonify(forecast_result)

@app.route('/intelligence/api/metrics', methods=['POST'])
def submit_metrics():
    """API endpoint for submitting enterprise metrics"""
    
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['company_id', 'industry', 'company_size', 'revenue_range']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Create metrics record
    metrics = EnterpriseMetrics(
        company_id=data['company_id'],
        industry=data['industry'],
        company_size=data['company_size'],
        revenue_range=data['revenue_range'],
        automation_percentage=data.get('automation_percentage', 0.0),
        ai_adoption_score=data.get('ai_adoption_score', 0.0),
        employee_productivity_index=data.get('employee_productivity_index', 100.0),
        operational_efficiency_ratio=data.get('operational_efficiency_ratio', 1.0),
        customer_satisfaction_score=data.get('customer_satisfaction_score', 0.0)
    )
    
    db.session.add(metrics)
    db.session.commit()
    
    return jsonify({'message': 'Metrics submitted successfully', 'id': metrics.id})

# Initialize database with sample data
with app.app_context():
    db.create_all()
    
    # Create sample enterprise metrics if empty
    if EnterpriseMetrics.query.count() == 0:
        sample_companies = [
            {
                'company_id': 'ANON_001',
                'industry': 'finance',
                'company_size': 'enterprise',
                'revenue_range': '10B+',
                'automation_percentage': 65.0,
                'ai_adoption_score': 8.2,
                'operational_efficiency_ratio': 1.15,
                'customer_satisfaction_score': 8.5
            },
            {
                'company_id': 'ANON_002',
                'industry': 'healthcare',
                'company_size': 'large',
                'revenue_range': '1B-10B',
                'automation_percentage': 45.0,
                'ai_adoption_score': 6.8,
                'operational_efficiency_ratio': 1.05,
                'customer_satisfaction_score': 7.9
            },
            {
                'company_id': 'ANON_003',
                'industry': 'manufacturing',
                'company_size': 'enterprise',
                'revenue_range': '10B+',
                'automation_percentage': 78.0,
                'ai_adoption_score': 7.5,
                'operational_efficiency_ratio': 1.25,
                'customer_satisfaction_score': 8.1
            }
        ]
        
        for company_data in sample_companies:
            metrics = EnterpriseMetrics(**company_data)
            db.session.add(metrics)
        
        db.session.commit()
        logger.info("Sample intelligence data created successfully")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)