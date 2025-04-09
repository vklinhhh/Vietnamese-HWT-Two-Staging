import os
import argparse
import json
from typing import Dict, List, Tuple


def load_mapping_file(file_path: str) -> List[Tuple[str, str]]:
    """
    Load the mapping file that defines accent transformations.
    
    Args:
        file_path: Path to mapping file (format: unaccented-accented)
        
    Returns:
        List of (unaccented, accented) tuples
    """
    mappings = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('-')
                if len(parts) == 2:
                    unaccented, accented = parts
                    mappings.append((unaccented, accented))
    
    return mappings


def create_accent_mapping(mappings: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Create a mapping from unaccented-accented pairs to label indices.
    
    Args:
        mappings: List of (unaccented, accented) tuples
        
    Returns:
        Dict mapping "unaccented-accented" to label index
    """
    accent_map = {}
    
    # Add entries for each unaccented-accented pair
    for i, (unaccented, accented) in enumerate(mappings):
        key = f"{unaccented}-{accented}"
        accent_map[key] = i
    
    return accent_map


def generate_word_level_mapping(mappings: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    """
    Generate a word-level mapping for more efficient lookup.
    For each unaccented word, maps all possible accented variants to indices.
    
    Args:
        mappings: List of (unaccented, accented) tuples
        
    Returns:
        Dict mapping unaccented words to {accented_variant: index} dicts
    """
    word_map = {}
    
    # Group by unaccented word
    for i, (unaccented, accented) in enumerate(mappings):
        if unaccented not in word_map:
            word_map[unaccented] = {}
        
        # Add the accented variant with its index
        word_map[unaccented][accented] = i
    
    return word_map


def save_mappings(
    accent_map: Dict[str, int], 
    word_map: Dict[str, Dict[str, int]], 
    output_dir: str
):
    """
    Save the mappings to JSON files.
    
    Args:
        accent_map: Direct mapping from unaccented-accented pairs to indices
        word_map: Word-level mapping for efficient lookup
        output_dir: Directory to save the mapping files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save direct mapping
    with open(os.path.join(output_dir, 'accent_map.json'), 'w', encoding='utf-8') as f:
        json.dump(accent_map, f, ensure_ascii=False, indent=2)
    
    # Save word-level mapping
    with open(os.path.join(output_dir, 'word_map.json'), 'w', encoding='utf-8') as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)


def analyze_mappings(mappings: List[Tuple[str, str]]):
    """
    Analyze the mappings to understand the data.
    
    Args:
        mappings: List of (unaccented, accented) tuples
    """
    # Count unique unaccented words
    unaccented_set = set(u for u, _ in mappings)
    
    # Count unique accented words
    accented_set = set(a for _, a in mappings)
    
    # Count variants per unaccented word
    variants_count = {}
    for unaccented, _ in mappings:
        if unaccented not in variants_count:
            variants_count[unaccented] = 0
        variants_count[unaccented] += 1
    
    # Find words with the most variants
    top_variants = sorted(variants_count.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Print analysis
    print(f"Total mapping pairs: {len(mappings)}")
    print(f"Unique unaccented words: {len(unaccented_set)}")
    print(f"Unique accented words: {len(accented_set)}")
    print(f"Average variants per unaccented word: {len(mappings) / len(unaccented_set):.2f}")
    print("\nTop 10 words with most accent variants:")
    for word, count in top_variants:
        print(f"  {word}: {count} variants")
    
    # Analyze character-level patterns
    all_chars = set()
    for unaccented, accented in mappings:
        all_chars.update(unaccented)
        all_chars.update(accented)
    
    print(f"\nTotal unique characters: {len(all_chars)}")


def generate_character_mapping(mappings: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Generate a character-level mapping of base characters to their accented variants.
    
    Args:
        mappings: List of (unaccented, accented) tuples
        
    Returns:
        Dict mapping base characters to lists of accented variants
    """
    char_map = {}
    
    # Find all single-character mappings
    for unaccented, accented in mappings:
        if len(unaccented) == 1 and len(accented) == 1:
            if unaccented not in char_map:
                char_map[unaccented] = []
            
            if accented not in char_map[unaccented]:
                char_map[unaccented].append(accented)
    
    return char_map


def main():
    parser = argparse.ArgumentParser(description="Generate accent mapping files")
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input mapping file (format: unaccented-accented)")
    parser.add_argument("--output", type=str, default="./accent_mappings",
                        help="Directory to save the generated mappings")
    
    # Optional arguments
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the mappings and print statistics")
    
    args = parser.parse_args()
    
    # Load mappings
    print(f"Loading mappings from {args.input}")
    mappings = load_mapping_file(args.input)
    print(f"Loaded {len(mappings)} mappings")
    
    # Create mappings
    accent_map = create_accent_mapping(mappings)
    word_map = generate_word_level_mapping(mappings)
    char_map = generate_character_mapping(mappings)
    
    # Save mappings
    save_mappings(accent_map, word_map, args.output)
    
    # Save character mapping
    with open(os.path.join(args.output, 'char_map.json'), 'w', encoding='utf-8') as f:
        json.dump(char_map, f, ensure_ascii=False, indent=2)
    
    print(f"Saved mappings to {args.output}")
    
    # Analyze mappings if requested
    if args.analyze:
        print("\nAnalyzing mappings:")
        analyze_mappings(mappings)
        
        # Print character map statistics
        print("\nCharacter-level mapping statistics:")
        print(f"Base characters with accented variants: {len(char_map)}")
        
        for base_char, variants in sorted(char_map.items()):
            print(f"  {base_char}: {len(variants)} variants ({', '.join(variants)})")


if __name__ == "__main__":
    main()