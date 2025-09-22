import json
import csv

def json_to_csv(json_path, csv_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    headers = [
        'timestamp',
        'method',
        'class_id',
        'avg_confidence',
        'num_detections',
        'avg_uncertainty',
        'avg_consensus_strength',
        'threat_score'
    ]

    rows = []

    for entry in data:
        timestamp = entry.get('timestamp', '')
        threat_score = entry.get('threat_score', None)
        baseline_results = entry.get('baseline_results', {})

        for method, runs in baseline_results.items():
            conf_list = []
            num_det_list = []
            for run in runs:
                # run might have nested structure, so check the first element if dict
                if not run:
                    continue
                # handle case where run might be a dict with keys or direct list of dicts
                if isinstance(run, dict):
                    # Try with 'confidence' if it exists
                    confidences = [v.get('confidence', 0) for v in run.values() if isinstance(v, dict)]
                    if confidences:
                        avg_conf = sum(confidences) / len(confidences)
                        conf_list.append(avg_conf)
                        num_det_list.append(len(confidences))
                elif isinstance(run, list) and isinstance(run[0], dict):
                    confs = [d.get('confidence', 0) for d in run if 'confidence' in d]
                    if confs:
                        avg_conf = sum(confs) / len(confs)
                        conf_list.append(avg_conf)
                        num_det_list.append(len(confs))
                else:
                    # unknown structure, skip or print warning
                    print(f"Warning: Unknown run structure for method {method}: {run}")
                    continue

            if len(conf_list) == 0:
                avg_conf_over_runs = None
                avg_num_dets = 0
            else:
                avg_conf_over_runs = sum(conf_list) / len(conf_list)
                avg_num_dets = sum(num_det_list) // len(num_det_list)

            avg_unc = None
            avg_cons_strength = None
            if method == 'consensus':
                cons_runs = entry.get('consensus', [])
                unc_vals = []
                cons_vals = []
                for cons in cons_runs:
                    if not cons:
                        continue
                    unc_vals.append(sum(item.get('uncertainty', 0) for item in cons) / len(cons))
                    cons_vals.append(sum(item.get('consensus_strength', 0) for item in cons) / len(cons))
                avg_unc = sum(unc_vals) / len(unc_vals) if unc_vals else None
                avg_cons_strength = sum(cons_vals) / len(cons_vals) if cons_vals else None

            row = {
                'timestamp': timestamp,
                'method': method,
                'class_id': 'all',
                'avg_confidence': avg_conf_over_runs,
                'num_detections': avg_num_dets,
                'avg_uncertainty': avg_unc,
                'avg_consensus_strength': avg_cons_strength,
                'threat_score': threat_score
            }
            rows.append(row)

    with open(csv_path, 'w', newline='') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV saved to {csv_path}")

# Usage:
json_to_csv('results.json', 'results_summary.csv')
