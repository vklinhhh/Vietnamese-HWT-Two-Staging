import re
import unicodedata
from functools import lru_cache
import torch
from typing import Dict, List, Optional, Tuple


def strip_vietnamese_diacritics(text: str) -> str:
    """
    Remove all diacritics from Vietnamese text, mapping to base characters.
    For example: 'Tiếng Việt' -> 'Tieng Viet'
    """
    # Normalize to decompose characters
    text = unicodedata.normalize('NFD', text)
    # Remove all combining diacritical marks
    text = re.sub(r'[\u0300-\u036f]', '', text)
    # Manual replacements for Vietnamese special characters
    replacement_map = {
        'đ': 'd', 'Đ': 'D',
        'ă': 'a', 'Ă': 'A',
        'â': 'a', 'Â': 'A',
        'ê': 'e', 'Ê': 'E',
        'ô': 'o', 'Ô': 'O',
        'ơ': 'o', 'Ơ': 'O',
        'ư': 'u', 'Ư': 'U',
        'á': 'a', 'Á': 'A',
        'à': 'a', 'À': 'A',
        'ả': 'a', 'Ả': 'A',
        'ã': 'a', 'Ã': 'A',
        'ạ': 'a', 'Ạ': 'A',
        'ấ': 'a', 'Ấ': 'A',
        'ầ': 'a', 'Ầ': 'A',
        'ẩ': 'a', 'Ẩ': 'A',
        'ẫ': 'a', 'Ẫ': 'A',
        'ậ': 'a', 'Ậ': 'A',
        'ắ': 'a', 'Ắ': 'A',
        'ằ': 'a', 'Ằ': 'A',
        'ẳ': 'a', 'Ẳ': 'A',
        'ẵ': 'a', 'Ẵ': 'A',
        'ặ': 'a', 'Ặ': 'A',
        'é': 'e', 'É': 'E',
        'è': 'e', 'È': 'E',
        'ẻ': 'e', 'Ẻ': 'E',
        'ẽ': 'e', 'Ẽ': 'E',
        'ẹ': 'e', 'Ẹ': 'E',
        'ế': 'e', 'Ế': 'E',
        'ề': 'e', 'Ề': 'E',
        'ể': 'e', 'Ể': 'E',
        'ễ': 'e', 'Ễ': 'E',
        'ệ': 'e', 'Ệ': 'E',
        'í': 'i', 'Í': 'I',
        'ì': 'i', 'Ì': 'I',
        'ỉ': 'i', 'Ỉ': 'I',
        'ĩ': 'i', 'Ĩ': 'I',
        'ị': 'i', 'Ị': 'I',
        'ó': 'o', 'Ó': 'O',
        'ò': 'o', 'Ò': 'O',
        'ỏ': 'o', 'Ỏ': 'O',
        'õ': 'o', 'Õ': 'O',
        'ọ': 'o', 'Ọ': 'O',
        'ố': 'o', 'Ố': 'O',
        'ồ': 'o', 'Ồ': 'O',
        'ổ': 'o', 'Ổ': 'O',
        'ỗ': 'o', 'Ỗ': 'O',
        'ộ': 'o', 'Ộ': 'O',
        'ớ': 'o', 'Ớ': 'O',
        'ờ': 'o', 'Ờ': 'O',
        'ở': 'o', 'Ở': 'O',
        'ỡ': 'o', 'Ỡ': 'O',
        'ợ': 'o', 'Ợ': 'O',
        'ú': 'u', 'Ú': 'U',
        'ù': 'u', 'Ù': 'U',
        'ủ': 'u', 'Ủ': 'U',
        'ũ': 'u', 'Ũ': 'U',
        'ụ': 'u', 'Ụ': 'U',
        'ứ': 'u', 'Ứ': 'U',
        'ừ': 'u', 'Ừ': 'U',
        'ử': 'u', 'Ử': 'U',
        'ữ': 'u', 'Ữ': 'U',
        'ự': 'u', 'Ự': 'U',
        'ý': 'y', 'Ý': 'Y',
        'ỳ': 'y', 'Ỳ': 'Y',
        'ỷ': 'y', 'Ỷ': 'Y',
        'ỹ': 'y', 'Ỹ': 'Y',
        'ỵ': 'y', 'Ỵ': 'Y'
    }
    
    for vietnamese_char, latin_char in replacement_map.items():
        text = text.replace(vietnamese_char, latin_char)
    
    return text


@lru_cache(maxsize=10000)
def create_diacritic_mapping() -> Dict[str, str]:
    """
    Create a mapping of Vietnamese characters with diacritics to their base form.
    Uses LRU cache for performance.
    """
    # Base characters
    base_chars = "aăâbcdđeêghiklmnoôơpqrstuưvxyAĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY"
    # Characters with diacritics
    diacritics = "àáảãạèéẻẽẹìíỉĩịòóỏõọùúủũụỳýỷỹỵÀÁẢÃẠÈÉẺẼẸÌÍỈĨỊÒÓỎÕỌÙÚỦŨỤỲÝỶỸỴ"
    combinations = "ăắằẳẵặâấầẩẫậêếềểễệôốồổỗộơớờởỡợưứừửữựĂẮẰẲẴẶÂẤẦẨẪẬÊẾỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰ"
    
    # Initialize the mapping dictionary
    mapping = {}
    
    # Process all characters with diacritics
    for char in diacritics + combinations:
        simplified = strip_vietnamese_diacritics(char)
        mapping[char] = simplified
        
    # Add identity mapping for base characters and punctuation
    for char in base_chars + ".,!?;:()[]{}-–—\"'""''…0123456789 ":
        mapping[char] = char
        
    return mapping


def simplify_vietnamese_dataset(examples: Dict) -> Dict:
    """
    Transform dataset by simplifying Vietnamese text (removing diacritics).
    Used for the first stage of training.
    """
    diacritic_mapping = create_diacritic_mapping()
    
    # Get the labels from the examples
    labels = examples.get('label', [])
    simplified_labels = []
    
    # Process each label
    for label in labels:
        # Convert to string if not already
        if not isinstance(label, str):
            label = str(label)
            
        # Simplify by mapping each character
        simplified = ''.join(diacritic_mapping.get(char, char) for char in label)
        simplified_labels.append(simplified)
    
    # Create a new dictionary with simplified labels
    result = examples.copy()
    result['label'] = simplified_labels
    result['original_label'] = labels  # Keep original for stage 2
    
    return result


class DiacriticAwareBeamSearch:
    """
    A beam search implementation that is aware of Vietnamese diacritics,
    giving preference to sequences with correct diacritic placement.
    """
    def __init__(
        self, 
        vocab_size: int, 
        max_length: int, 
        num_beams: int = 4, 
        diacritic_bias: float = 0.2,
        length_penalty: float = 1.0
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_beams = num_beams
        self.diacritic_bias = diacritic_bias
        self.length_penalty = length_penalty
        self.diacritic_mapping = create_diacritic_mapping()
        
        # Extract chars with diacritics for bias
        self.diacritic_chars = set()
        for char, base in self.diacritic_mapping.items():
            if char != base:
                self.diacritic_chars.add(char)
    
    def _is_diacritic_char(self, token_str: str) -> bool:
        """Check if a token string represents a Vietnamese character with diacritics."""
        return any(char in self.diacritic_chars for char in token_str)
    
    def beam_search(
        self, 
        model: torch.nn.Module, 
        encoder_output: torch.Tensor, 
        tokenizer,
        sos_token_id: int, 
        eos_token_id: int
    ) -> List[List[int]]:
        """
        Perform beam search with diacritic awareness for Vietnamese text.
        
        Args:
            model: The decoder model
            encoder_output: The encoder output tensor
            tokenizer: The tokenizer used to decode tokens to characters
            sos_token_id: Start of sequence token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            List of generated token sequences
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize with start tokens
        curr_ids = torch.full(
            (batch_size, 1), 
            sos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Initial beam scores
        beam_scores = torch.zeros(batch_size, self.num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # Set score of non-first beam to very negative
        
        # Initialize beam storage
        generated_sequences = []
        done_beams = [[] for _ in range(batch_size)]
        
        # Start beam search
        for step in range(self.max_length):
            if len(generated_sequences) == batch_size:
                break
                
            # Get model logits for the current step
            logits = model(
                decoder_input_ids=curr_ids,
                encoder_hidden_states=encoder_output
            )
            
            # Apply diacritic bias
            if step > 0 and self.diacritic_bias > 0:
                # Convert current tokens to strings to check for diacritics
                curr_token_strs = [
                    tokenizer.decode([token_id.item()]) 
                    for token_id in curr_ids[:, -1]
                ]
                
                # Apply bias to tokens that represent diacritic characters
                for i, token_str in enumerate(curr_token_strs):
                    if self._is_diacritic_char(token_str):
                        # Boost the score for correct diacritic usage
                        beam_scores[i // self.num_beams, i % self.num_beams] += self.diacritic_bias
            
            # Get next token scores
            next_token_logits = logits[:, -1, :]
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores.view(-1, 1)
            
            # Reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, self.num_beams * vocab_size
            )
            
            # Get the top-k token indices
            next_top_scores, next_top_indices = torch.topk(
                next_token_scores, 
                k=self.num_beams, 
                dim=1
            )
            
            # Convert to beam indices and token indices
            next_beam_indices = next_top_indices // vocab_size
            next_token_indices = next_top_indices % vocab_size
            
            # Build the next beams
            next_beam_scores = next_top_scores
            next_beam_tokens = next_token_indices
            next_beam_prev = next_beam_indices
            
            # Create the new beams
            new_ids = []
            for batch_idx in range(batch_size):
                if len(done_beams[batch_idx]) == self.num_beams:
                    # This batch item is done
                    continue
                    
                for beam_idx in range(self.num_beams):
                    # Get the beam info
                    beam_token_idx = next_beam_tokens[batch_idx, beam_idx].item()
                    beam_prev_idx = next_beam_prev[batch_idx, beam_idx].item()
                    beam_score = next_beam_scores[batch_idx, beam_idx].item()
                    
                    # Get the previous beam
                    prev_ids = curr_ids[batch_idx * self.num_beams + beam_prev_idx].clone()
                    
                    # Add the new token
                    new_beam = torch.cat([prev_ids, torch.tensor([beam_token_idx], device=device)])
                    
                    # Check if beam is complete
                    if beam_token_idx == eos_token_id:
                        # Apply length penalty
                        final_score = beam_score / (len(new_beam) ** self.length_penalty)
                        done_beams[batch_idx].append((final_score, new_beam))
                    else:
                        new_ids.append(new_beam)
            
            # If all beams are done for all batches, break
            if all(len(beams) == self.num_beams for beams in done_beams):
                break
                
            # Update curr_ids for next step
            curr_ids = torch.stack(new_ids) if new_ids else None
            
            # If we have no more active beams, break
            if curr_ids is None:
                break
        
        # Get the best beam for each batch
        for batch_idx in range(batch_size):
            if not done_beams[batch_idx]:
                # If no complete beams, take the best incomplete beam
                if curr_ids is not None:
                    best_beam = curr_ids[batch_idx * self.num_beams]
                    generated_sequences.append(best_beam.tolist())
            else:
                # Sort by score and take the best
                best_beam = sorted(done_beams[batch_idx], key=lambda x: x[0], reverse=True)[0][1]
                generated_sequences.append(best_beam.tolist())
        
        return generated_sequences


# Evaluation metrics specific to Vietnamese OCR
def compute_diacritic_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute metrics specific to Vietnamese diacritics recognition.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with various metrics
    """
    # Create mapping for diacritics
    diacritic_mapping = create_diacritic_mapping()
    
    # Characters with diacritics
    diacritic_chars = set()
    for char, base in diacritic_mapping.items():
        if char != base:
            diacritic_chars.add(char)
    
    # Initialize counters
    total_chars = 0
    correct_chars = 0
    total_diacritics = 0
    correct_diacritics = 0
    diacritic_added_correctly = 0
    diacritic_added_incorrectly = 0
    diacritic_missed = 0
    
    for pred, ref in zip(predictions, references):
        # Calculate character-level metrics
        for i, ref_char in enumerate(ref):
            total_chars += 1
            if i < len(pred) and pred[i] == ref_char:
                correct_chars += 1
            
            # Diacritic-specific metrics
            if ref_char in diacritic_chars:
                total_diacritics += 1
                if i < len(pred) and pred[i] == ref_char:
                    correct_diacritics += 1
                    diacritic_added_correctly += 1
                elif i < len(pred) and pred[i] != ref_char:
                    diacritic_added_incorrectly += 1
                # We count missed if pred is too short or doesn't have diacritic
                else:
                    diacritic_missed += 1
    
    # Calculate metrics
    char_accuracy = correct_chars / max(1, total_chars)
    diacritic_accuracy = correct_diacritics / max(1, total_diacritics)
    diacritic_precision = diacritic_added_correctly / max(1, diacritic_added_correctly + diacritic_added_incorrectly)
    diacritic_recall = diacritic_added_correctly / max(1, total_diacritics)
    diacritic_f1 = 2 * (diacritic_precision * diacritic_recall) / max(1e-6, diacritic_precision + diacritic_recall)
    
    return {
        "char_accuracy": char_accuracy,
        "diacritic_accuracy": diacritic_accuracy,
        "diacritic_precision": diacritic_precision,
        "diacritic_recall": diacritic_recall,
        "diacritic_f1": diacritic_f1
    }