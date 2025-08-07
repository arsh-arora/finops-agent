"""
Default fallback agent implementation
Handles general requests when no specialized agent matches
"""

import structlog
from typing import List, Dict, Any
from ..base.agent import HardenedAgent
from ..base.registry import tool

logger = structlog.get_logger(__name__)


class DefaultAgent(HardenedAgent):
    """
    Default fallback agent for general purpose interactions
    
    Used when:
    - No specialized agent matches the request
    - Request is too general or ambiguous
    - Routing fails or has low confidence
    """
    
    _domain = "default"
    _capabilities = [
        "general_assistance",
        "question_answering",
        "basic_help",
        "conversation"
    ]
    
    def get_capabilities(self) -> List[str]:
        return self._capabilities.copy()
    
    def get_domain(self) -> str:
        return self._domain
    
    async def _process_message(
        self, 
        message: str, 
        memory_context: List[str], 
        plan: Dict[str, Any]
    ) -> str:
        """Process general messages with basic assistance"""
        logger.info(
            "default_agent_processing",
            agent_id=self.agent_id,
            message_length=len(message),
            memory_context_size=len(memory_context)
        )
        
        return f"I'm here to help with your request: {message}. For specialized assistance with cost analysis, GitHub operations, or research tasks, please specify your domain. Memory context: {len(memory_context)} relevant memories found."
    
    @tool(description="Provide general assistance and guidance")
    async def general_help(self, query: str) -> str:
        """Provide general help and guidance"""
        return f"General assistance for: {query}. I can help with basic questions and guide you to specialized agents for specific domains."
    
    @tool(description="Answer general questions")
    async def answer_question(self, question: str) -> str:
        """Answer general questions using available context"""
        return f"Answering question: {question}. For detailed analysis, consider using specialized agents for specific domains like FinOps, GitHub, or Research."