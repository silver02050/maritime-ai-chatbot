import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import os
import json

class MaritimeBERT:
    def __init__(self, model_path: str = None):
        """
        Initialize the MaritimeBERT model.
        
        Args:
            model_path: Path to the fine-tuned model weights. If None, will use default BERT.
        """
        # Use a smaller BERT variant for memory efficiency
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if model_path and os.path.exists(model_path):
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Move model to CPU and set to evaluation mode
        self.model = self.model.to('cpu')
        self.model.eval()
        
        # Enable model quantization for reduced memory footprint
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        # Load IMO regulations database
        self.imo_regulations = self._load_imo_regulations()
    
    def _load_imo_regulations(self) -> Dict:
        """
        Load IMO regulations from a JSON file.
        Returns a dictionary mapping regulation codes to their details.
        """
        regulations_path = os.path.join(os.path.dirname(__file__), 'imo_regulations.json')
        if os.path.exists(regulations_path):
            with open(regulations_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _extract_imo_references(self, text: str) -> List[Dict]:
        """
        Extract IMO regulation references from text.
        
        Args:
            text: Input text to search for IMO references
            
        Returns:
            List of dictionaries containing regulation details
        """
        references = []
        # Common IMO regulation patterns (e.g., SOLAS II-2/3.2, MARPOL Annex VI)
        patterns = [
            r'(SOLAS|MARPOL|COLREG|STCW)\s*([A-Za-z0-9\-/\.]+)',
            r'IMO\s*Resolution\s*([A-Za-z0-9\-/\.]+)',
            r'Annex\s*([IVX]+)\s*of\s*(SOLAS|MARPOL)'
        ]
        
        for pattern in patterns:
            import re
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                reg_code = match.group(0).strip()
                if reg_code in self.imo_regulations:
                    references.append({
                        'code': reg_code,
                        'details': self.imo_regulations[reg_code]
                    })
        
        return references
    
    def _format_imo_citation(self, reference: Dict) -> str:
        """
        Format an IMO regulation citation.
        
        Args:
            reference: Dictionary containing regulation details
            
        Returns:
            Formatted citation string
        """
        details = reference['details']
        return f"{reference['code']} - {details.get('title', '')} ({details.get('year', '')})"
    
    def fine_tune(self, 
                 train_data: List[Dict[str, str]], 
                 epochs: int = 3,
                 batch_size: int = 8,
                 learning_rate: float = 2e-5):
        """
        Fine-tune the model on maritime Q&A pairs.
        
        Args:
            train_data: List of dictionaries containing 'question', 'answer', and 'context'
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        # Implementation will be added when dataset is provided
        pass
    
    def generate_response(self, 
                         question: str, 
                         context: str = None) -> Tuple[str, float]:
        """
        Generate a response to a maritime question.
        
        Args:
            question: The question to answer
            context: Optional IMO regulation context
            
        Returns:
            Tuple of (answer, confidence_score)
        """
        # Extract IMO references from question and context
        imo_references = self._extract_imo_references(question)
        if context:
            imo_references.extend(self._extract_imo_references(context))
        
        # Add IMO references to context
        if imo_references:
            citations = "\n".join([self._format_imo_citation(ref) for ref in imo_references])
            if context:
                context = f"{context}\n\nIMO Regulations:\n{citations}"
            else:
                context = f"IMO Regulations:\n{citations}"
        
        # Tokenize inputs
        inputs = self.tokenizer(
            question,
            context if context else "",
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        
        # Move inputs to CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get start and end positions
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        # Calculate confidence score
        confidence = torch.max(outputs.start_logits) * torch.max(outputs.end_logits)
        
        # Decode answer
        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end]
        )
        
        return answer, confidence.item()
    
    def save_model(self, save_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path) 