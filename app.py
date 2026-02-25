# app.py
from flask import Flask, render_template, request, jsonify
import math
import random
import copy

app = Flask(__name__)


class DecisionEngine:
    def __init__(self):
        self.qualitative_map = {
            "very low": 1, "low": 3, "medium": 5,
            "high": 7, "very high": 9
        }

    def calculate_roc_weights(self, num_criteria):
        weights = []
        for i in range(1, num_criteria + 1):
            weight = sum(1.0 / j for j in range(i, num_criteria + 1)) / num_criteria
            weights.append(weight)
        return weights

    def _to_float(self, value):
        if value is None: return 5.0
        if isinstance(value, str):
            # Try numeric first, then qualitative label
            try:
                return float(value.strip())
            except ValueError:
                return float(self.qualitative_map.get(value.lower().strip(), 5.0))
        try:
            return float(value)
        except (ValueError, TypeError):
            return 5.0

    def run_topsis(self, options, criteria):
        # 1. Normalise + weight in one step
        norm_matrix = {}
        for crit in criteria:
            c_id = crit['id']
            sq_sum = sum(opt['values'][c_id] ** 2 for opt in options)
            denominator = math.sqrt(sq_sum) if sq_sum > 0 else 1
            for opt in options:
                norm_matrix.setdefault(opt['name'], {})
                norm_matrix[opt['name']][c_id] = (opt['values'][c_id] / denominator) * crit['weight']

        # 2. Ideal best and worst
        ideal_best, ideal_worst = {}, {}
        for crit in criteria:
            c_id = crit['id']
            vals = [norm_matrix[o['name']][c_id] for o in options]
            if crit['type'] == 'benefit':
                ideal_best[c_id], ideal_worst[c_id] = max(vals), min(vals)
            else:
                ideal_best[c_id], ideal_worst[c_id] = min(vals), max(vals)

        # 3. Score
        results = []
        for opt in options:
            name = opt['name']
            d_best  = math.sqrt(sum((norm_matrix[name][c['id']] - ideal_best[c['id']]) ** 2 for c in criteria))
            d_worst = math.sqrt(sum((norm_matrix[name][c['id']] - ideal_worst[c['id']]) ** 2 for c in criteria))
            score = d_worst / (d_best + d_worst) if (d_best + d_worst) > 0 else 0.5
            results.append({"name": name, "score": score})

        return sorted(results, key=lambda x: x['score'], reverse=True), ideal_best, ideal_worst, norm_matrix

    def simulate_decision(self, options, criteria, iterations=1000):
        """
        Monte Carlo: run TOPSIS [iterations] times with simulated values
        for dynamic criteria. Each option's baseline is its own entered value.
        A shared market shift hits all options, plus tiny personal noise.
        """
        win_counts = {opt['name']: 0 for opt in options}

        # Build float options without mutating originals
        float_options = [
            {
                "name": opt['name'],
                "values": {c['id']: self._to_float(opt['values'].get(c['id'])) for c in criteria}
            }
            for opt in options
        ]

        for _ in range(iterations):
            shared_shift = {c['id']: random.gauss(0, 1.2) for c in criteria if c.get('dynamic')}
            sim_options = []
            for opt in float_options:
                temp_vals = opt['values'].copy()
                for c_id, shift in shared_shift.items():
                    temp_vals[c_id] = max(1, min(9, temp_vals[c_id] + shift + random.gauss(0, 0.3)))
                sim_options.append({"name": opt['name'], "values": temp_vals})

            pass_results, _, _, _ = self.run_topsis(sim_options, criteria)
            win_counts[pass_results[0]['name']] += 1

        return sorted(
            [{"name": n, "confidence": round((c / iterations) * 100, 2)} for n, c in win_counts.items()],
            key=lambda x: x['confidence'], reverse=True
        )

    def explain_all(self, options, criteria):
        """
        Single TOPSIS pass for all options.
        GAP FIX: gap% is calculated on the original 1-9 scale, not on
        normalised internal values — so it's meaningful to users.
        gap=0% means this option IS the best for that criterion among all options.
        gap=100% means it's the worst.
        """
        results, ideal_best, ideal_worst, norm_matrix = self.run_topsis(options, criteria)

        # Ideal in original scale — the best actual value any option has per criterion
        ideal_raw = {}
        worst_raw = {}
        for crit in criteria:
            c_id = crit['id']
            raw_vals = [opt['values'][c_id] for opt in options]
            if crit['type'] == 'benefit':
                ideal_raw[c_id] = max(raw_vals)
                worst_raw[c_id] = min(raw_vals)
            else:
                ideal_raw[c_id] = min(raw_vals)
                worst_raw[c_id] = max(raw_vals)

        all_explanations = {}

        for opt in options:
            opt_name = opt['name']
            opt_norm = norm_matrix[opt_name]
            explanation = []

            for crit in criteria:
                c_id        = crit['id']
                actual      = opt['values'][c_id]
                ideal_val   = ideal_raw[c_id]
                worst_val   = worst_raw[c_id]

                # GAP FIX: use raw scale range, not normalised range
                raw_range = abs(ideal_val - worst_val)
                if raw_range > 0:
                    gap_pct = abs(actual - ideal_val) / raw_range * 100
                else:
                    gap_pct = 0.0

                all_vals = [norm_matrix[o['name']][c_id] for o in options]
                wv = opt_norm[c_id]
                is_best = (wv == max(all_vals) if crit['type'] == 'benefit' else wv == min(all_vals))

                explanation.append({
                    "name":      crit['name'],
                    "actual":    actual,
                    "ideal_raw": ideal_val,
                    "gap_pct":   round(gap_pct, 1),
                    "is_best":   is_best,
                    "type":      crit['type'],
                    "weight":    crit['weight'],
                    "dynamic":   crit.get('dynamic', False),
                })

            explanation.sort(key=lambda x: x['gap_pct'])
            all_explanations[opt_name] = explanation

        return all_explanations, results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json

        decision_goal = data.get('goal', 'Decision')
        criteria_data = data.get('criteria', [])
        options_data  = data.get('options', [])

        if not criteria_data or not options_data:
            return jsonify({'error': 'Please provide both criteria and options'}), 400

        engine = DecisionEngine()

        # Build criteria with ROC weights
        weights = engine.calculate_roc_weights(len(criteria_data))
        criteria = []
        for i, crit in enumerate(criteria_data):
            criteria.append({
                'id':      f'c{i}',
                'name':    crit['name'],
                'type':    crit['type'],
                'dynamic': crit.get('dynamic', False),
                'weight':  weights[i]
            })

        # Build options — convert values to floats immediately
        options = []
        for opt in options_data:
            option_values = {}
            for i, value in enumerate(opt['values']):
                option_values[f'c{i}'] = engine._to_float(value)
            options.append({'name': opt['name'], 'values': option_values})

        # Deep copy before simulation so originals are safe
        original_options = copy.deepcopy(options)

        # Run Monte Carlo simulation
        simulation_results = engine.simulate_decision(options, criteria, iterations=1000)

        # Run explain_all on original values (single TOPSIS pass)
        all_explanations, topsis_results = engine.explain_all(original_options, criteria)

        # Winner reasoning — single sentence
        winner_name = simulation_results[0]['name']
        winner_expl = all_explanations[winner_name]
        best_crit   = min(winner_expl, key=lambda e: (e['gap_pct'], -e['weight']))
        reasoning   = (
            f"'{winner_name}' is selected due to its {best_crit['name']} "
            f"(value={best_crit['actual']:.1f}) being closest to the ideal "
            f"(ideal={best_crit['ideal_raw']:.1f}, gap={best_crit['gap_pct']:.1f}%)."
        )

        # Build per-option breakdown
        option_breakdown = []
        for opt in original_options:
            opt_name   = opt['name']
            expl       = all_explanations[opt_name]
            confidence = next(r['confidence'] for r in simulation_results if r['name'] == opt_name)
            rank       = next(i + 1 for i, r in enumerate(simulation_results) if r['name'] == opt_name)

            option_breakdown.append({
                "name":       opt_name,
                "rank":       rank,
                "confidence": confidence,
                "strengths":  [
                    {"name": e['name'], "actual": e['actual'], "ideal": e['ideal_raw'], "gap": e['gap_pct']}
                    for e in expl if e['gap_pct'] <= 40
                ],
                "weaknesses": [
                    {"name": e['name'], "actual": e['actual'], "ideal": e['ideal_raw'], "gap": e['gap_pct']}
                    for e in expl if e['gap_pct'] > 40
                ],
            })

        return jsonify({
            'success':          True,
            'goal':             decision_goal,
            'criteria': [
                {
                    'name':    c['name'],
                    'type':    c['type'],
                    'dynamic': c['dynamic'],
                    'weight':  round(c['weight'] * 100, 2)
                }
                for c in criteria
            ],
            'simulation_results': simulation_results,   # confidence %
            'topsis_results': [                          # deterministic score
                {'rank': i+1, 'name': r['name'], 'score': round(r['score'] * 100, 2)}
                for i, r in enumerate(topsis_results)
            ],
            'reasoning':        reasoning,
            'option_breakdown': option_breakdown,
            'gap_note':         "Gap % is relative to options you entered, not an absolute scale."
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)