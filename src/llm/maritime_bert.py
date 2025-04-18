import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from typing import Dict, List, Tuple
import os

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