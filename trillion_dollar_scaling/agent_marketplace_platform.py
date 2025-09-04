"""
Agent Marketplace Platform - Complete Discovery, Deployment & Revenue System
$50B+ Value Potential - Enterprise AI Agent Ecosystem
"""

import os
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import stripe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "marketplace-secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///marketplace.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)

# Stripe configuration
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "sk_test_placeholder")

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    company = db.Column(db.String(200))
    user_type = db.Column(db.String(50), default='enterprise')  # enterprise, developer, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)
    subscription_tier = db.Column(db.String(50), default='basic')  # basic, professional, enterprise
    
    # Relationships
    agents = db.relationship('Agent', backref='developer', lazy=True)
    purchases = db.relationship('Purchase', backref='buyer', lazy=True)
    reviews = db.relationship('Review', backref='reviewer', lazy=True)

class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(100), nullable=False)  # finance, healthcare, manufacturing, etc.
    industry = db.Column(db.String(100), nullable=False)
    pricing_model = db.Column(db.String(50), nullable=False)  # one-time, subscription, usage-based
    price = db.Column(db.Float, nullable=False)
    version = db.Column(db.String(20), default='1.0.0')
    developer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    download_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)
    revenue = db.Column(db.Float, default=0.0)
    
    # Technical specifications
    tech_stack = db.Column(db.JSON)  # Languages, frameworks, dependencies
    api_endpoints = db.Column(db.JSON)  # Available API endpoints
    deployment_requirements = db.Column(db.JSON)  # System requirements
    integration_guides = db.Column(db.Text)
    
    # Compliance and security
    compliance_standards = db.Column(db.JSON)  # GDPR, HIPAA, SOX, etc.
    security_features = db.Column(db.JSON)
    audit_logs = db.Column(db.JSON)
    
    # Relationships
    purchases = db.relationship('Purchase', backref='agent', lazy=True)
    reviews = db.relationship('Review', backref='agent', lazy=True)
    analytics = db.relationship('AgentAnalytics', backref='agent', lazy=True)

class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    buyer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    purchase_date = db.Column(db.DateTime, default=datetime.utcnow)
    amount_paid = db.Column(db.Float, nullable=False)
    license_type = db.Column(db.String(50), nullable=False)  # single, enterprise, unlimited
    status = db.Column(db.String(50), default='active')  # active, expired, cancelled
    stripe_payment_id = db.Column(db.String(200))
    license_key = db.Column(db.String(200), unique=True)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reviewer_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    title = db.Column(db.String(200))
    comment = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified_purchase = db.Column(db.Boolean, default=False)

class AgentAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agent.id'), nullable=False)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    views = db.Column(db.Integer, default=0)
    downloads = db.Column(db.Integer, default=0)
    revenue = db.Column(db.Float, default=0.0)
    active_installations = db.Column(db.Integer, default=0)
    performance_metrics = db.Column(db.JSON)  # Response time, accuracy, etc.

class MarketplaceAnalytics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, default=datetime.utcnow().date)
    total_agents = db.Column(db.Integer, default=0)
    total_developers = db.Column(db.Integer, default=0)
    total_enterprises = db.Column(db.Integer, default=0)
    total_revenue = db.Column(db.Float, default=0.0)
    top_categories = db.Column(db.JSON)
    growth_metrics = db.Column(db.JSON)

# Business Logic Classes
class AgentMarketplace:
    def __init__(self):
        self.commission_rate = 0.3  # 30% platform commission
        
    def calculate_commission(self, sale_amount):
        """Calculate platform commission from agent sale"""
        platform_commission = sale_amount * self.commission_rate
        developer_earnings = sale_amount * (1 - self.commission_rate)
        return platform_commission, developer_earnings
    
    def search_agents(self, query="", category="", industry="", price_range="", rating_min=0):
        """Advanced agent search with filters"""
        agents_query = Agent.query.filter(Agent.is_active == True, Agent.is_verified == True)
        
        if query:
            agents_query = agents_query.filter(
                db.or_(
                    Agent.name.contains(query),
                    Agent.description.contains(query)
                )
            )
        
        if category:
            agents_query = agents_query.filter(Agent.category == category)
            
        if industry:
            agents_query = agents_query.filter(Agent.industry == industry)
            
        if rating_min:
            agents_query = agents_query.filter(Agent.rating >= rating_min)
            
        if price_range:
            if price_range == "free":
                agents_query = agents_query.filter(Agent.price == 0)
            elif price_range == "low":
                agents_query = agents_query.filter(Agent.price.between(0.01, 1000))
            elif price_range == "medium":
                agents_query = agents_query.filter(Agent.price.between(1000, 10000))
            elif price_range == "high":
                agents_query = agents_query.filter(Agent.price > 10000)
        
        return agents_query.order_by(Agent.rating.desc(), Agent.download_count.desc()).all()
    
    def get_trending_agents(self, limit=10):
        """Get trending agents based on recent activity"""
        last_week = datetime.utcnow() - timedelta(days=7)
        trending = db.session.query(Agent).join(AgentAnalytics).filter(
            AgentAnalytics.date >= last_week.date(),
            Agent.is_active == True,
            Agent.is_verified == True
        ).order_by(AgentAnalytics.downloads.desc()).limit(limit).all()
        return trending
    
    def get_agent_recommendations(self, user_id, limit=5):
        """Get personalized agent recommendations"""
        user = User.query.get(user_id)
        if not user:
            return []
            
        # Get user's purchase history
        purchased_categories = db.session.query(Agent.category).join(Purchase).filter(
            Purchase.buyer_id == user_id
        ).distinct().all()
        
        if purchased_categories:
            categories = [cat[0] for cat in purchased_categories]
            recommendations = Agent.query.filter(
                Agent.category.in_(categories),
                Agent.is_active == True,
                Agent.is_verified == True,
                ~Agent.id.in_(
                    db.session.query(Purchase.agent_id).filter(Purchase.buyer_id == user_id)
                )
            ).order_by(Agent.rating.desc()).limit(limit).all()
        else:
            # For new users, show top-rated agents
            recommendations = Agent.query.filter(
                Agent.is_active == True,
                Agent.is_verified == True
            ).order_by(Agent.rating.desc()).limit(limit).all()
            
        return recommendations

class PaymentProcessor:
    def __init__(self):
        self.stripe = stripe
        
    def create_payment_intent(self, amount, currency="usd", metadata=None):
        """Create Stripe payment intent"""
        try:
            intent = self.stripe.PaymentIntent.create(
                amount=int(amount * 100),  # Stripe uses cents
                currency=currency,
                metadata=metadata or {}
            )
            return intent
        except Exception as e:
            logger.error(f"Payment intent creation failed: {e}")
            return None
    
    def process_purchase(self, buyer_id, agent_id, payment_intent_id):
        """Process successful agent purchase"""
        try:
            agent = Agent.query.get(agent_id)
            buyer = User.query.get(buyer_id)
            
            if not agent or not buyer:
                return False
                
            # Create purchase record
            purchase = Purchase(
                buyer_id=buyer_id,
                agent_id=agent_id,
                amount_paid=agent.price,
                license_type='enterprise',
                stripe_payment_id=payment_intent_id,
                license_key=self.generate_license_key()
            )
            
            db.session.add(purchase)
            
            # Update agent statistics
            agent.download_count += 1
            agent.revenue += agent.price
            
            # Calculate commissions
            marketplace = AgentMarketplace()
            platform_commission, developer_earnings = marketplace.calculate_commission(agent.price)
            
            # Update analytics
            today = datetime.utcnow().date()
            analytics = AgentAnalytics.query.filter_by(agent_id=agent_id, date=today).first()
            if not analytics:
                analytics = AgentAnalytics(agent_id=agent_id, date=today)
                db.session.add(analytics)
            
            analytics.downloads += 1
            analytics.revenue += agent.price
            
            db.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Purchase processing failed: {e}")
            db.session.rollback()
            return False
    
    def generate_license_key(self):
        """Generate unique license key"""
        import uuid
        return f"AMP-{uuid.uuid4().hex[:16].upper()}"

# Initialize business logic
marketplace = AgentMarketplace()
payment_processor = PaymentProcessor()

# Routes
@app.route('/')
def index():
    """Marketplace homepage"""
    trending_agents = marketplace.get_trending_agents(6)
    categories = db.session.query(Agent.category).distinct().all()
    industries = db.session.query(Agent.industry).distinct().all()
    
    # Marketplace statistics
    stats = {
        'total_agents': Agent.query.filter_by(is_active=True, is_verified=True).count(),
        'total_developers': User.query.filter_by(user_type='developer').count(),
        'total_downloads': db.session.query(db.func.sum(Agent.download_count)).scalar() or 0,
        'total_revenue': db.session.query(db.func.sum(Agent.revenue)).scalar() or 0
    }
    
    return render_template('marketplace/index.html', 
                         trending_agents=trending_agents,
                         categories=[cat[0] for cat in categories],
                         industries=[ind[0] for ind in industries],
                         stats=stats)

@app.route('/agents')
def browse_agents():
    """Browse and search agents"""
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    industry = request.args.get('industry', '')
    price_range = request.args.get('price_range', '')
    rating_min = float(request.args.get('rating_min', 0))
    
    agents = marketplace.search_agents(query, category, industry, price_range, rating_min)
    
    return render_template('marketplace/browse.html',
                         agents=agents,
                         query=query,
                         category=category,
                         industry=industry,
                         price_range=price_range,
                         rating_min=rating_min)

@app.route('/agent/<int:agent_id>')
def agent_detail(agent_id):
    """Agent detail page"""
    agent = Agent.query.get_or_404(agent_id)
    reviews = Review.query.filter_by(agent_id=agent_id).order_by(Review.created_at.desc()).limit(10).all()
    
    # Update view count
    today = datetime.utcnow().date()
    analytics = AgentAnalytics.query.filter_by(agent_id=agent_id, date=today).first()
    if not analytics:
        analytics = AgentAnalytics(agent_id=agent_id, date=today)
        db.session.add(analytics)
    analytics.views += 1
    db.session.commit()
    
    # Check if user has purchased this agent
    has_purchased = False
    if 'user_id' in session:
        purchase = Purchase.query.filter_by(
            buyer_id=session['user_id'],
            agent_id=agent_id,
            status='active'
        ).first()
        has_purchased = purchase is not None
    
    return render_template('marketplace/agent_detail.html',
                         agent=agent,
                         reviews=reviews,
                         has_purchased=has_purchased)

@app.route('/purchase/<int:agent_id>', methods=['POST'])
def purchase_agent(agent_id):
    """Initiate agent purchase"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login to purchase agents'}), 401
    
    agent = Agent.query.get_or_404(agent_id)
    
    # Create payment intent
    payment_intent = payment_processor.create_payment_intent(
        amount=agent.price,
        metadata={
            'agent_id': agent_id,
            'buyer_id': session['user_id'],
            'agent_name': agent.name
        }
    )
    
    if not payment_intent:
        return jsonify({'error': 'Payment processing error'}), 500
    
    return jsonify({
        'client_secret': payment_intent.client_secret,
        'amount': agent.price
    })

@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.environ.get('STRIPE_WEBHOOK_SECRET', 'whsec_test')
        )
    except ValueError:
        return '', 400
    except stripe.error.SignatureVerificationError:
        return '', 400
    
    if event['type'] == 'payment_intent.succeeded':
        payment_intent = event['data']['object']
        metadata = payment_intent['metadata']
        
        success = payment_processor.process_purchase(
            buyer_id=int(metadata['buyer_id']),
            agent_id=int(metadata['agent_id']),
            payment_intent_id=payment_intent['id']
        )
        
        if success:
            logger.info(f"Purchase completed: Agent {metadata['agent_id']} by User {metadata['buyer_id']}")
    
    return '', 200

@app.route('/developer/dashboard')
def developer_dashboard():
    """Developer dashboard"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    if user.user_type != 'developer':
        flash('Access denied. Developer account required.', 'error')
        return redirect(url_for('index'))
    
    # Get developer's agents and analytics
    agents = Agent.query.filter_by(developer_id=user.id).all()
    
    total_revenue = sum(agent.revenue for agent in agents)
    total_downloads = sum(agent.download_count for agent in agents)
    
    # Recent analytics
    last_30_days = datetime.utcnow().date() - timedelta(days=30)
    recent_analytics = db.session.query(AgentAnalytics).join(Agent).filter(
        Agent.developer_id == user.id,
        AgentAnalytics.date >= last_30_days
    ).all()
    
    return render_template('developer/dashboard.html',
                         user=user,
                         agents=agents,
                         total_revenue=total_revenue,
                         total_downloads=total_downloads,
                         recent_analytics=recent_analytics)

@app.route('/api/analytics/<int:agent_id>')
def agent_analytics_api(agent_id):
    """API endpoint for agent analytics"""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    agent = Agent.query.get_or_404(agent_id)
    user = User.query.get(session['user_id'])
    
    # Check if user owns this agent or is admin
    if agent.developer_id != user.id and user.user_type != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    # Get analytics data
    analytics = AgentAnalytics.query.filter_by(agent_id=agent_id).order_by(AgentAnalytics.date.desc()).limit(30).all()
    
    data = {
        'dates': [a.date.isoformat() for a in reversed(analytics)],
        'views': [a.views for a in reversed(analytics)],
        'downloads': [a.downloads for a in reversed(analytics)],
        'revenue': [float(a.revenue) for a in reversed(analytics)]
    }
    
    return jsonify(data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['user_type'] = user.user_type
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        company = request.form.get('company', '')
        user_type = request.form.get('user_type', 'enterprise')
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('auth/register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            company=company,
            user_type=user_type
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data if empty
    if User.query.count() == 0:
        # Create admin user
        admin = User(
            username='admin',
            email='admin@agentmarketplace.com',
            password_hash=generate_password_hash('admin123'),
            user_type='admin',
            is_verified=True
        )
        db.session.add(admin)
        
        # Create sample developer
        developer = User(
            username='ai_developer',
            email='dev@company.com',
            password_hash=generate_password_hash('dev123'),
            user_type='developer',
            company='AI Innovations Inc',
            is_verified=True
        )
        db.session.add(developer)
        
        # Create sample enterprise user
        enterprise = User(
            username='fortune500',
            email='cto@fortune500.com',
            password_hash=generate_password_hash('enterprise123'),
            user_type='enterprise',
            company='Fortune 500 Corp',
            subscription_tier='enterprise',
            is_verified=True
        )
        db.session.add(enterprise)
        
        db.session.commit()
        
        # Create sample agents
        sample_agents = [
            {
                'name': 'Financial Risk Analyzer Pro',
                'description': 'Advanced AI agent for real-time financial risk assessment and regulatory compliance monitoring',
                'category': 'finance',
                'industry': 'banking',
                'pricing_model': 'subscription',
                'price': 5000.0,
                'developer_id': developer.id,
                'is_verified': True,
                'rating': 4.8,
                'download_count': 156,
                'tech_stack': ['Python', 'TensorFlow', 'FastAPI', 'PostgreSQL'],
                'compliance_standards': ['SOX', 'GDPR', 'PCI-DSS'],
                'security_features': ['End-to-end encryption', 'Multi-factor authentication', 'Audit logging']
            },
            {
                'name': 'Healthcare Data Intelligence',
                'description': 'HIPAA-compliant AI agent for medical record processing and patient outcome prediction',
                'category': 'healthcare',
                'industry': 'healthcare',
                'pricing_model': 'one-time',
                'price': 12000.0,
                'developer_id': developer.id,
                'is_verified': True,
                'rating': 4.9,
                'download_count': 89,
                'tech_stack': ['Python', 'PyTorch', 'Flask', 'MongoDB'],
                'compliance_standards': ['HIPAA', 'GDPR', 'FDA'],
                'security_features': ['PHI encryption', 'Access controls', 'Compliance monitoring']
            },
            {
                'name': 'Supply Chain Optimizer',
                'description': 'AI-powered supply chain optimization with predictive analytics and automated procurement',
                'category': 'operations',
                'industry': 'manufacturing',
                'pricing_model': 'usage-based',
                'price': 0.50,  # per transaction
                'developer_id': developer.id,
                'is_verified': True,
                'rating': 4.6,
                'download_count': 234,
                'tech_stack': ['Python', 'Scikit-learn', 'Django', 'Redis'],
                'compliance_standards': ['ISO 27001', 'GDPR'],
                'security_features': ['API authentication', 'Data encryption', 'Rate limiting']
            }
        ]
        
        for agent_data in sample_agents:
            agent = Agent(**agent_data)
            db.session.add(agent)
        
        db.session.commit()
        logger.info("Sample data created successfully")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)