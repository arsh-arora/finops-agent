"""
OpenRouter LLM Client for agent routing and planning
"""

import os
import json
import aiohttp
from typing import Dict, Any, List, Optional


class OpenRouterClient:
    """OpenRouter LLM client for agent routing decisions"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-20b:free"):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (will use OPENROUTER_API_KEY env var if not provided)
            model: Model to use for routing decisions
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            # Use the hardcoded key from memory config as fallback
            self.api_key = "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"
        
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.call_count = 0
    
    async def complete(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """
        Complete a chat request using OpenRouter
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Optional model override
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response content as string
        """
        self.call_count += 1
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 200),
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        print(f"🤖 OpenRouter API success: {content[:100]}...")
                        return content
                    else:
                        error_text = await response.text()
                        print(f"⚠️ OpenRouter API error {response.status}: {error_text}")
                        return await self._fallback_routing(messages[0]["content"])
                        
        except aiohttp.ClientError as e:
            print(f"⚠️ OpenRouter client error: {e}")
            return await self._fallback_routing(messages[0]["content"])
        except Exception as e:
            print(f"⚠️ OpenRouter general error: {e}")
            return await self._fallback_routing(messages[0]["content"])
    
    async def _fallback_routing(self, content: str) -> str:
        """Fallback routing when OpenRouter API is unavailable"""
        content_lower = content.lower()
        
        # Domain keyword analysis with confidence scoring
        domain_keywords = {
            'finops': ['cost', 'budget', 'billing', 'spend', 'expense', 'financial', 'finops',
                      'aws', 'azure', 'gcp', 'cloud', 'optimization', 'savings', 'npv', 'irr'],
            'github': ['github', 'git', 'repository', 'repo', 'security', 'vulnerability', 
                      'cve', 'epss', 'code', 'scan', 'commit', 'pull request'],
            'document': ['document', 'pdf', 'word', 'excel', 'file', 'parse', 'extract',
                        'bounding box', 'bbox', 'content', 'text', 'docling'],
            'research': ['research', 'search', 'web', 'internet', 'find', 'investigate',
                        'tavily', 'fact check', 'verify', 'credibility', 'sources'],
            'deep_research': ['comprehensive', 'multi-hop', 'orchestrate', 'coordinate',
                             'synthesize', 'cross-domain', 'multi-agent', 'complex analysis']
        }
        
        # Score each domain
        best_domain = 'finops'  # Default
        best_score = 0
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > best_score:
                best_score = score
                best_domain = domain
        
        confidence = min(best_score / 3.0, 1.0)  # Scale confidence based on matches
        
        return json.dumps({
            "selected_domain": best_domain,
            "confidence_score": confidence,
            "reasoning": f"Fallback routing: detected {best_score} {best_domain} keywords"
        })


# Create a shared instance for the application
openrouter_client = OpenRouterClient()