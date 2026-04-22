def calculate_edit_distance(s1, s2):
    if not s1 or not s2:
        return len(s1) or len(s2)
    
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def calculate_mlops_scores(pred_text, true_text):
    """Calculates Model Performance Metrics on the fly."""
    if not true_text:
        return None
        
    pred_text = str(pred_text).upper().replace(" ", "")
    true_text = str(true_text).upper().replace(" ", "")
    
    # Exact Match
    exact_match = 1.0 if pred_text == true_text else 0.0
    
    # Edit Distance
    distance = calculate_edit_distance(pred_text, true_text)
    
    # Character Accuracy (1 - CER)
    max_len = max(len(pred_text), len(true_text))
    char_acc = 1.0 if max_len == 0 else max(0.0, 1.0 - (distance / max_len))
    
    return {
        "exact_match": exact_match,
        "edit_distance": distance,
        "character_accuracy": char_acc
    }
