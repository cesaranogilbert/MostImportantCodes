"""
Routes for company personalization and asset management
"""

from flask import Blueprint, request, jsonify, render_template, current_app
from services.intelligent_personalization_agent import IntelligentPersonalizationAgent
from models import db, CompanyProfile, CompanyAsset, ContentHistory
import json
import os
from werkzeug.utils import secure_filename
import base64

personalization_bp = Blueprint('personalization', __name__, url_prefix='/personalization')

# Initialize the personalization agent
personalization_agent = IntelligentPersonalizationAgent()

@personalization_bp.route('/')
def index():
    """Main personalization dashboard"""
    try:
        # Get all company profiles
        profiles = CompanyProfile.query.filter_by(is_active=True).all()
        
        return render_template('personalization_dashboard.html', profiles=profiles)
    except Exception as e:
        current_app.logger.error(f"Dashboard error: {str(e)}")
        return render_template('personalization_dashboard.html', profiles=[])

@personalization_bp.route('/profile')
def company_profiling():
    """Company profiling interface"""
    return render_template('company_profiling.html')

@personalization_bp.route('/api/start-profiling', methods=['POST'])
def start_profiling():
    """Start interactive company profiling"""
    try:
        initial_data = request.get_json() or {}
        
        # Generate profiling session
        profiling_session = personalization_agent.interactive_company_profiling(initial_data)
        
        return jsonify({
            "success": True,
            "profiling_session": profiling_session
        })
    except Exception as e:
        current_app.logger.error(f"Profiling start error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/process-company', methods=['POST'])
def process_company():
    """Process complete company information"""
    try:
        data = request.get_json()
        company_data = data.get('company_data', {})
        assets = data.get('assets', [])
        
        # Process company information
        result = personalization_agent.process_company_information(company_data, assets)
        
        if result.get('error'):
            return jsonify({
                "success": False,
                "error": result['error']
            }), 400
        
        return jsonify({
            "success": True,
            "processing_result": result
        })
    except Exception as e:
        current_app.logger.error(f"Company processing error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/upload-assets', methods=['POST'])
def upload_assets():
    """Upload company assets (logos, images, documents)"""
    try:
        company_profile_id = request.form.get('company_profile_id')
        if not company_profile_id:
            return jsonify({
                "success": False,
                "error": "Company profile ID required"
            }), 400
        
        # Process uploaded files
        files = []
        for file_key in request.files:
            file = request.files[file_key]
            if file and file.filename:
                # Read file data
                file_data = file.read()
                file_info = {
                    'filename': secure_filename(file.filename),
                    'content_type': file.content_type,
                    'size': len(file_data),
                    'data': base64.b64encode(file_data).decode('utf-8')
                }
                files.append(file_info)
        
        # Process assets through the agent
        result = personalization_agent.upload_and_process_assets(int(company_profile_id), files)
        
        return jsonify({
            "success": True,
            "upload_result": result
        })
    except Exception as e:
        current_app.logger.error(f"Asset upload error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/personalize-content', methods=['POST'])
def personalize_content():
    """Personalize content for specific company"""
    try:
        data = request.get_json()
        content = data.get('content', '')
        content_type = data.get('content_type', 'general')
        company_profile_id = data.get('company_profile_id')
        additional_context = data.get('additional_context', {})
        
        if not content or not company_profile_id:
            return jsonify({
                "success": False,
                "error": "Content and company profile ID required"
            }), 400
        
        # Personalize content
        result = personalization_agent.personalize_content(
            content, content_type, company_profile_id, additional_context
        )
        
        return jsonify({
            "success": True,
            "personalization_result": result
        })
    except Exception as e:
        current_app.logger.error(f"Content personalization error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/integrate-agents', methods=['POST'])
def integrate_agents():
    """Integrate personalization with AI agents"""
    try:
        data = request.get_json()
        company_profile_id = data.get('company_profile_id')
        target_agents = data.get('target_agents', [])
        
        if not company_profile_id:
            return jsonify({
                "success": False,
                "error": "Company profile ID required"
            }), 400
        
        # Integrate with AI agents
        result = personalization_agent.integrate_with_ai_agents(company_profile_id, target_agents)
        
        return jsonify({
            "success": True,
            "integration_result": result
        })
    except Exception as e:
        current_app.logger.error(f"Agent integration error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/companies', methods=['GET'])
def get_companies():
    """Get list of all company profiles"""
    try:
        profiles = CompanyProfile.query.filter_by(is_active=True).all()
        
        companies = []
        for profile in profiles:
            companies.append({
                "id": profile.id,
                "company_name": profile.company_name,
                "industry": profile.industry,
                "company_size": profile.company_size,
                "communication_tone": profile.communication_tone,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat()
            })
        
        return jsonify({
            "success": True,
            "companies": companies
        })
    except Exception as e:
        current_app.logger.error(f"Companies retrieval error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/company/<int:company_id>', methods=['GET'])
def get_company_details(company_id):
    """Get detailed company profile"""
    try:
        profile = CompanyProfile.query.get(company_id)
        if not profile:
            return jsonify({
                "success": False,
                "error": "Company profile not found"
            }), 404
        
        # Get company assets
        assets = CompanyAsset.query.filter_by(company_profile_id=company_id, is_active=True).all()
        
        # Get content history
        content_history = ContentHistory.query.filter_by(company_profile_id=company_id).order_by(ContentHistory.created_at.desc()).limit(10).all()
        
        company_details = {
            "profile": {
                "id": profile.id,
                "company_name": profile.company_name,
                "industry": profile.industry,
                "company_size": profile.company_size,
                "headquarters_location": profile.headquarters_location,
                "business_description": profile.business_description,
                "mission_statement": profile.mission_statement,
                "vision_statement": profile.vision_statement,
                "core_values": profile.core_values,
                "key_products_services": profile.key_products_services,
                "target_markets": profile.target_markets,
                "competitive_advantages": profile.competitive_advantages,
                "communication_tone": profile.communication_tone,
                "content_preferences": profile.content_preferences,
                "branding_guidelines": profile.branding_guidelines
            },
            "assets": [
                {
                    "id": asset.id,
                    "asset_name": asset.asset_name,
                    "asset_type": asset.asset_type,
                    "asset_category": asset.asset_category,
                    "description": asset.description,
                    "is_primary": asset.is_primary,
                    "usage_count": asset.usage_count,
                    "created_at": asset.created_at.isoformat()
                } for asset in assets
            ],
            "content_history": [
                {
                    "id": history.id,
                    "content_type": history.content_type,
                    "content_title": history.content_title,
                    "agent_used": history.agent_used,
                    "user_rating": history.user_rating,
                    "created_at": history.created_at.isoformat()
                } for history in content_history
            ]
        }
        
        return jsonify({
            "success": True,
            "company_details": company_details
        })
    except Exception as e:
        current_app.logger.error(f"Company details error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/agent-status', methods=['GET'])
def get_agent_status():
    """Get personalization agent status"""
    try:
        status = personalization_agent.get_agent_status()
        
        return jsonify({
            "success": True,
            "agent_status": status
        })
    except Exception as e:
        current_app.logger.error(f"Agent status error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@personalization_bp.route('/api/test-personalization', methods=['POST'])
def test_personalization():
    """Test personalization with sample content"""
    try:
        data = request.get_json()
        company_profile_id = data.get('company_profile_id')
        
        if not company_profile_id:
            return jsonify({
                "success": False,
                "error": "Company profile ID required"
            }), 400
        
        # Test with sample content
        sample_content = """
        Dear [Company],
        
        We are pleased to present this business proposal for your consideration. Our comprehensive analysis indicates significant opportunities for operational improvement and cost optimization.
        
        Our recommended approach includes:
        - Strategic assessment of current operations
        - Implementation of best practices
        - Performance monitoring and optimization
        
        We look forward to discussing this opportunity with your team.
        
        Best regards,
        [Your Company]
        """
        
        # Personalize the sample content
        result = personalization_agent.personalize_content(
            sample_content, 'business_proposal', company_profile_id
        )
        
        return jsonify({
            "success": True,
            "test_result": result
        })
    except Exception as e:
        current_app.logger.error(f"Personalization test error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Error handlers for the blueprint
@personalization_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Resource not found"
    }), 404

@personalization_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500