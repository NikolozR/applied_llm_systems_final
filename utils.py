def distribute_roles(model_confidences):
    best_judge_model = None
    max_judge_score = -1.0

    for model_data in model_confidences:
        judge_score = next(
            (item['score'] for item in model_data['confidences'] if item['role'] == 'Judge'), 
            0.0
        )
        if judge_score > max_judge_score:
            max_judge_score = judge_score
            best_judge_model = model_data

    final_assignments = {}
    
    for model_data in model_confidences:
        model_name = model_data['model']
        if model_data == best_judge_model:
            final_assignments[model_name] = 'Judge'
        else:
            final_assignments[model_name] = 'Solver'
    return final_assignments