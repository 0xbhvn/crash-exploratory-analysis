import json
import csv

# Read the log file
truth_rows = []
with open('replay_predictions.log', 'r') as log_file:
    for line in log_file:
        try:
            data = json.loads(line.strip())
            if data.get('kind') == 'truth':
                truth_rows.append(data)
        except json.JSONDecodeError:
            continue

# Write to CSV
with open('truth_rows.csv', 'w', newline='') as csv_file:
    fieldnames = ['t', 'next_streak_number', 'starts_after_game_id', 'predicted_cluster',
                  'prediction_desc', 'confidence', 'prob_class_0', 'prob_class_1',
                  'prob_class_2', 'prob_class_3', 'model_percentiles', 'cluster_desc',
                  'actual_cluster', 'actual_streak_length', 'correct', 'resolved_at']

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for row in truth_rows:
        # Remove the 'kind' field from the row
        if 'kind' in row:
            del row['kind']
        # Convert model_percentiles list to string for CSV
        if 'model_percentiles' in row and isinstance(row['model_percentiles'], list):
            row['model_percentiles'] = str(
                row['model_percentiles']).replace('[', '').replace(']', '')
        writer.writerow(row)

print(f"Extracted {len(truth_rows)} truth rows to truth_rows.csv")
