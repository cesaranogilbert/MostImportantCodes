"""
Gilbert's Sympathetic Writing Style AI Agent

This AI agent captures Gilbert Cesarano's unique writing style with enhanced
sympathy, relatability, and emotional depth while maintaining his authentic voice.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from dataclasses import dataclass


@dataclass
class StyleAnalysis:
    """Analysis results for writing style transformation"""
    vulnerability_score: float
    relatability_score: float
    authenticity_score: float
    cultural_intelligence: float
    emotional_depth: float
    flow_integration: float


class GilbertSympatheticWritingAgent:
    """
    AI Agent that transforms content to match Gilbert's sympathetic writing style.
    
    Key Features:
    - Vulnerability and failure story integration
    - Cultural intelligence and multicultural perspectives
    - Personal stakes and emotional cost awareness
    - Journey-focused narratives over destination-focused content
    - Self-deprecating humor and relatable struggles
    - Universal human experience connections
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        # Gilbert's core style characteristics
        self.style_framework = {
            "voice_characteristics": {
                "raw_honesty": "Direct, unfiltered truth-telling",
                "personal_vulnerability": "Admits challenges and struggles openly",
                "cultural_depth": "Blends German precision, Italian passion, Swiss discipline",
                "conversational_tone": "Uses ellipses, fragments, natural speech patterns",
                "spiritual_integration": "Combines business wisdom with spiritual insights"
            },
            "structural_patterns": {
                "metaphorical_framework": "Transforms activities into life laboratories",
                "training_grounds_concept": "Every experience becomes a learning opportunity",
                "bridge_building": "Connects physical activities to business wisdom",
                "circular_reasoning": "The way you do anything is the way you do everything",
                "question_cascades": "Ends sections with penetrating self-reflection questions"
            },
            "linguistic_patterns": {
                "ellipses_usage": "Creates dramatic pauses and emphasis",
                "capitalized_emphasis": "Strategic caps for key concepts like FLOW",
                "fragmented_sentences": "Creates urgency and conversational rhythm",
                "cultural_expressions": "AHO! and other multicultural gratitude expressions",
                "sacred_language": "Holy place, sanctuary, temple references"
            },
            "sympathy_enhancers": {
                "vulnerability_stories": "Share personal failures and embarrassing moments",
                "emotional_cost": "Show the price of growth and discipline",
                "journey_focus": "Process over destination, struggle over success",
                "universal_connections": "Connect to shared human experiences",
                "self_deprecating_humor": "Light humor about personal imperfections",
                "family_context": "Include personal relationships and family influence",
                "doubt_uncertainty": "Acknowledge moments of confusion and learning"
            }
        }
        
        # Enhanced sympathy transformation prompts
        self.sympathy_prompts = {
            "add_vulnerability": """
            Transform this content to include personal vulnerability and failure stories.
            Instead of presenting only success and wisdom, show:
            - Embarrassing moments and mistakes
            - Times when strategies failed
            - Personal struggles and doubts
            - Learning through humiliation or setbacks
            Keep Gilbert's authentic voice while making him more relatable and human.
            """,
            
            "emotional_depth": """
            Add emotional depth and personal stakes to this content.
            Show:
            - The emotional cost of decisions and growth
            - How challenges affected relationships and family
            - Moments of genuine fear, uncertainty, or loneliness
            - The human price of pursuing excellence
            Maintain the inspirational message while adding emotional reality.
            """,
            
            "journey_focus": """
            Reframe this content to focus on the journey rather than achievements.
            Emphasize:
            - The messy, imperfect process of learning
            - Multiple attempts and iterations
            - How failures led to insights
            - The ongoing nature of growth
            Transform achievement-focused content into process-focused wisdom.
            """,
            
            "universal_connection": """
            Connect this content to universal human experiences.
            Make readers think "I've felt this too" by including:
            - Common fears and insecurities
            - Shared struggles (parenting, relationships, self-doubt)
            - Universal moments of triumph and failure
            - Emotions everyone can relate to
            Keep Gilbert's unique perspective while making it universally relatable.
            """,
            
            "cultural_wisdom": """
            Integrate Gilbert's multicultural background with sympathetic storytelling.
            Show:
            - How different cultures taught him different lessons
            - Moments of cultural confusion or learning
            - Family traditions that shaped his thinking
            - The challenge of integrating different cultural values
            Make his cultural intelligence feel personal and earned through experience.
            """
        }
    
    def analyze_content_style(self, content: str) -> StyleAnalysis:
        """Analyze content for Gilbert's style characteristics"""
        
        analysis_prompt = f"""
        Analyze this content for Gilbert Cesarano's writing style characteristics.
        Rate each dimension from 0.0 to 1.0:
        
        1. Vulnerability Score: Does it include personal failures, mistakes, or struggles?
        2. Relatability Score: Can average readers connect with the experiences?
        3. Authenticity Score: Does it feel genuine and unfiltered?
        4. Cultural Intelligence: Does it integrate multicultural perspectives?
        5. Emotional Depth: Does it show emotional cost and human stakes?
        6. Flow Integration: Does it connect activities to life philosophy?
        
        Content to analyze:
        {content}
        
        Respond with this exact JSON format:
        {{"vulnerability_score": 0.8, "relatability_score": 0.7, "authenticity_score": 0.9, "cultural_intelligence": 0.6, "emotional_depth": 0.8, "flow_integration": 0.7, "analysis_notes": "Brief explanation of scores"}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an expert in analyzing writing styles, particularly Gilbert Cesarano's sympathetic and authentic approach."},
                    {"role": "user", "content": analysis_prompt}
                ],

            )
            
            content = response.choices[0].message.content
            if content:
                analysis_data = json.loads(content)
            else:
                analysis_data = {}
            
            return StyleAnalysis(
                vulnerability_score=analysis_data.get('vulnerability_score', 0.0),
                relatability_score=analysis_data.get('relatability_score', 0.0),
                authenticity_score=analysis_data.get('authenticity_score', 0.0),
                cultural_intelligence=analysis_data.get('cultural_intelligence', 0.0),
                emotional_depth=analysis_data.get('emotional_depth', 0.0),
                flow_integration=analysis_data.get('flow_integration', 0.0)
            )
            
        except Exception as e:
            print(f"Error analyzing content style: {e}")
            return StyleAnalysis(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    
    def transform_to_sympathetic_style(self, content: str, target_enhancement: str = "comprehensive") -> str:
        """
        Transform content to Gilbert's sympathetic writing style
        
        Args:
            content: Original content to transform
            target_enhancement: Type of enhancement ("vulnerability", "emotional_depth", 
                              "journey_focus", "universal_connection", "cultural_wisdom", "comprehensive")
        """
        
        # First analyze current style
        current_analysis = self.analyze_content_style(content)
        
        if target_enhancement == "comprehensive":
            # Apply all enhancements in sequence
            enhanced_content = self._apply_comprehensive_transformation(content, current_analysis)
        else:
            # Apply specific enhancement
            enhanced_content = self._apply_specific_enhancement(content, target_enhancement)
        
        # Final polish with Gilbert's linguistic patterns
        final_content = self._apply_linguistic_patterns(enhanced_content)
        
        return final_content
    
    def _apply_comprehensive_transformation(self, content: str, analysis: StyleAnalysis) -> str:
        """Apply comprehensive transformation based on analysis gaps"""
        
        transformation_prompt = f"""
        Transform this content to match Gilbert Cesarano's sympathetic writing style.
        
        GILBERT'S AUTHENTIC VOICE CHARACTERISTICS:
        - Raw honesty with personal vulnerability
        - Multicultural wisdom (German precision, Italian passion, Swiss discipline)
        - "Training grounds" philosophy - every experience teaches something
        - Spiritual integration with business wisdom
        - Flow state philosophy and circular reasoning
        - Sacred language ("sanctuary", "holy place", "temple")
        
        SYMPATHY ENHANCEMENTS NEEDED:
        1. ADD VULNERABILITY: Include personal failures, embarrassing moments, times when strategies failed
        2. SHOW EMOTIONAL COST: The price of growth, impact on family/relationships, moments of doubt
        3. FOCUS ON JOURNEY: Process over achievements, multiple attempts, learning through mistakes
        4. UNIVERSAL CONNECTION: Shared human experiences everyone can relate to
        5. CULTURAL STORYTELLING: How different cultures taught lessons through personal experience
        6. SELF-DEPRECATING HUMOR: Light humor about imperfections and learning moments
        
        LINGUISTIC PATTERNS TO MAINTAIN:
        - Ellipses for dramatic pauses
        - Fragmented sentences for emphasis
        - Question cascades at section ends
        - Strategic capitalization (FLOW, etc.)
        - Cultural expressions (AHO!)
        - Metaphorical framework connecting activities to life lessons
        
        EXAMPLE TRANSFORMATION:
        Instead of: "Success requires discipline and focus."
        Gilbert's style: "I've failed at discipline more times than I care to count. There was this morning last month when I hit snooze six times, skipped my workout, and felt like a complete fraud teaching others about focus. But that failure taught me something powerful - discipline isn't about being perfect. It's about showing up again after you've fallen down. The German side of me wants systematic perfection, the Italian side craves passionate commitment, but the human side? The human side just needs grace to begin again."
        
        Transform this content maintaining Gilbert's authentic voice while adding sympathetic depth:
        
        {content}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are Gilbert Cesarano's writing coach, helping to enhance his authentic voice with deeper sympathy and relatability while maintaining his unique multicultural wisdom and flow philosophy."},
                    {"role": "user", "content": transformation_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            print(f"Error in comprehensive transformation: {e}")
            return content
    
    def _apply_specific_enhancement(self, content: str, enhancement_type: str) -> str:
        """Apply specific enhancement type"""
        
        if enhancement_type not in self.sympathy_prompts:
            return content
        
        enhancement_prompt = f"""
        {self.sympathy_prompts[enhancement_type]}
        
        Transform this content in Gilbert Cesarano's voice:
        {content}
        
        Maintain his authentic characteristics:
        - Multicultural wisdom and experiences
        - Flow philosophy and training grounds concept
        - Spiritual integration with practical wisdom
        - Conversational tone with ellipses and emphasis
        - Question cascades for reflection
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an expert in Gilbert Cesarano's writing style, helping to enhance his content with greater sympathy and relatability."},
                    {"role": "user", "content": enhancement_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            print(f"Error in specific enhancement: {e}")
            return content
    
    def _apply_linguistic_patterns(self, content: str) -> str:
        """Apply Gilbert's specific linguistic patterns"""
        
        linguistic_prompt = f"""
        Apply Gilbert Cesarano's specific linguistic patterns to this content:
        
        LINGUISTIC PATTERNS:
        1. Use ellipses (..) for dramatic pauses and emphasis
        2. Strategic capitalization for key concepts (FLOW, Me vs Me!)
        3. Fragmented sentences for urgency and rhythm
        4. Question cascades at section ends
        5. Cultural expressions like "AHO!" for closing
        6. Sacred language integration ("sanctuary", "temple", "holy")
        7. Bridge phrases connecting ideas ("That's where...", "Here's what...", "And why like...")
        
        Content to enhance:
        {content}
        
        Return the content with these linguistic patterns applied naturally.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are a linguistic pattern specialist for Gilbert Cesarano's writing style."},
                    {"role": "user", "content": linguistic_prompt}
                ],
                max_tokens=1500,
                temperature=0.6
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            print(f"Error applying linguistic patterns: {e}")
            return content
    
    def generate_sympathetic_content(self, topic: str, content_type: str = "blog_post", target_length: str = "medium") -> str:
        """
        Generate new content from scratch in Gilbert's sympathetic style
        
        Args:
            topic: Topic to write about
            content_type: Type of content ("blog_post", "social_media", "email", "article")
            target_length: Length target ("short", "medium", "long")
        """
        
        length_guidelines = {
            "short": "300-500 words, focused single insight",
            "medium": "800-1200 words, comprehensive exploration",
            "long": "1500-2500 words, deep dive with multiple sections"
        }
        
        generation_prompt = f"""
        Write content about "{topic}" in Gilbert Cesarano's sympathetic writing style.
        
        CONTENT TYPE: {content_type}
        TARGET LENGTH: {length_guidelines.get(target_length, "medium length")}
        
        GILBERT'S SYMPATHETIC STYLE REQUIREMENTS:
        
        1. START WITH VULNERABILITY: Begin with a personal failure, struggle, or embarrassing moment related to the topic
        2. MULTICULTURAL WISDOM: Integrate lessons from German, Italian, Swiss cultures through personal experience
        3. TRAINING GROUNDS CONCEPT: Frame the topic as a "training ground" for life lessons
        4. EMOTIONAL HONESTY: Show the emotional cost, family impact, moments of doubt
        5. JOURNEY FOCUS: Emphasize the messy process over clean achievements
        6. UNIVERSAL CONNECTION: Connect to experiences everyone can relate to
        7. SPIRITUAL INTEGRATION: Include meaningful quotes (business + biblical/spiritual)
        8. FLOW PHILOSOPHY: Connect to flow state and the idea that "how you do anything is how you do everything"
        
        LINGUISTIC PATTERNS:
        - Use ellipses for dramatic pauses
        - Strategic capitalization for emphasis
        - Question cascades at section ends
        - Fragmented sentences for rhythm
        - Sacred language ("sanctuary", "temple", "holy")
        - End with "AHO!" or similar cultural expression
        
        STRUCTURE EXAMPLE:
        1. Vulnerable opening (personal failure/struggle)
        2. What this taught about the topic
        3. Cultural wisdom integration
        4. Business/life parallels
        5. Universal human connection
        6. Spiritual insight
        7. Provocative questions for reflection
        8. Cultural closing expression
        
        Write compelling, sympathetic content about: {topic}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are Gilbert Cesarano, writing in your signature sympathetic style that blends vulnerability, multicultural wisdom, and spiritual insight to create deeply relatable content."},
                    {"role": "user", "content": generation_prompt}
                ],
                max_tokens=2500,
                temperature=0.75
            )
            
            content = response.choices[0].message.content
            return content if content else ""
            
        except Exception as e:
            print(f"Error generating sympathetic content: {e}")
            return f"Error generating content about {topic}"
    
    def validate_style_authenticity(self, content: str) -> Dict[str, any]:
        """Validate that content matches Gilbert's authentic sympathetic style"""
        
        validation_prompt = f"""
        Evaluate this content for Gilbert Cesarano's authentic sympathetic writing style.
        
        EVALUATION CRITERIA (Rate 1-10):
        1. Vulnerability: Includes personal failures, struggles, embarrassing moments
        2. Authenticity: Feels genuine, unfiltered, not performative
        3. Relatability: Average readers can connect with experiences
        4. Cultural Depth: Integrates multicultural wisdom naturally
        5. Emotional Honesty: Shows emotional cost, family impact, doubt
        6. Journey Focus: Emphasizes process over achievements
        7. Spiritual Integration: Combines practical and spiritual wisdom
        8. Linguistic Patterns: Uses ellipses, fragments, question cascades
        9. Universal Connection: Connects to shared human experiences
        10. Flow Philosophy: Connects activities to life philosophy
        
        Content to evaluate:
        {content}
        
        Provide this exact JSON response:
        {{"overall_score": 8, "vulnerability": 7, "authenticity": 9, "relatability": 8, "cultural_depth": 6, "emotional_honesty": 7, "journey_focus": 8, "spiritual_integration": 6, "linguistic_patterns": 8, "universal_connection": 9, "flow_philosophy": 7, "strengths": ["vulnerability", "authenticity"], "improvements": ["cultural_depth", "spiritual_integration"], "authenticity_assessment": "Assessment details"}}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",  # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of Gilbert Cesarano's authentic sympathetic writing style."},
                    {"role": "user", "content": validation_prompt}
                ],

            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return {"error": "No response content"}
            
        except Exception as e:
            print(f"Error validating style authenticity: {e}")
            return {"error": "Validation failed"}


# Factory function for easy instantiation
def create_sympathetic_writing_agent() -> GilbertSympatheticWritingAgent:
    """Create and return a new instance of Gilbert's Sympathetic Writing Agent"""
    return GilbertSympatheticWritingAgent()


# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    agent = create_sympathetic_writing_agent()
    
    # Test content transformation
    sample_content = """
    Success in business requires discipline and strategic thinking. 
    Leaders must make difficult decisions and maintain focus on long-term goals. 
    The best executives combine analytical skills with emotional intelligence 
    to drive organizational performance and create shareholder value.
    """
    
    print("Original Content:")
    print(sample_content)
    print("\n" + "="*50 + "\n")
    
    # Transform to sympathetic style
    sympathetic_version = agent.transform_to_sympathetic_style(sample_content)
    print("Sympathetic Version:")
    print(sympathetic_version)
    print("\n" + "="*50 + "\n")
    
    # Validate the transformation
    validation_results = agent.validate_style_authenticity(sympathetic_version)
    print("Validation Results:")
    print(f"Overall Score: {validation_results.get('overall_score', 'N/A')}/10")
    print(f"Strengths: {', '.join(validation_results.get('strengths', []))}")
    print(f"Areas for Improvement: {', '.join(validation_results.get('improvements', []))}")