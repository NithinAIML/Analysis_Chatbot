import logging
from typing import List, Dict, Any, Optional
import tiktoken
from config import MEMORY_K, MAX_TOKENS_LIMIT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Class for maintaining conversation history and context for the chatbot.
    """
    
    def __init__(self, memory_k: int = MEMORY_K, max_tokens: int = MAX_TOKENS_LIMIT):
        """
        Initialize the conversation memory.
        
        Args:
            memory_k: Number of recent conversation turns to keep
            max_tokens: Maximum number of tokens to maintain in memory
        """
        self.memory_k = memory_k
        self.max_tokens = max_tokens
        self.conversation_history = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        if role not in ["user", "assistant"]:
            logger.warning(f"Invalid message role: {role}. Using 'user' instead.")
            role = "user"
            
        self.conversation_history.append({"role": role, "content": content})
        
        # Trim history if needed
        self._trim_history()
    
    def _trim_history(self):
        """
        Trim the conversation history to stay within token and turn limits.
        """
        # Always keep at least the most recent turn
        if not self.conversation_history:
            return
            
        # First, enforce the memory_k limit (number of turns)
        if len(self.conversation_history) > self.memory_k * 2:  # Each turn is user + assistant
            self.conversation_history = self.conversation_history[-(self.memory_k * 2):]
        
        # Then, enforce the token limit
        while self._count_tokens() > self.max_tokens and len(self.conversation_history) > 2:
            # Remove the oldest turn (user + assistant)
            self.conversation_history = self.conversation_history[2:]
    
    def _count_tokens(self) -> int:
        """
        Count the number of tokens in the current conversation history.
        
        Returns:
            The total token count
        """
        # Join all messages
        text = " ".join([msg["content"] for msg in self.conversation_history])
        
        # Count tokens
        return len(self.tokenizer.encode(text))
    
    def get_recent_messages(self, k: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get the k most recent messages from the conversation history.
        
        Args:
            k: Number of recent messages to return (if None, return all)
        
        Returns:
            List of recent messages
        """
        if k is None or k >= len(self.conversation_history):
            return self.conversation_history
        else:
            return self.conversation_history[-k:]
    
    def get_context_string(self) -> str:
        """
        Get a formatted string representation of the conversation history.
        
        Returns:
            Formatted conversation history string
        """
        context = []
        
        for msg in self.conversation_history:
            role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
            context.append(f"{role_prefix}{msg['content']}")
        
        return "\n\n".join(context)
    
    def clear(self):
        """
        Clear the conversation history.
        """
        self.conversation_history = []