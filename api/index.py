from flask import Flask, render_template, request, jsonify
import math
import random
import copy

# IMPORTANT: template folder path for Vercel
app = Flask(__name__, template_folder="../templates")


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
        if value is None:
            return 5.0
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return float(self.qualitative_map.get(value.lower().strip(), 5.0))
        try:
            return float(value)
        except:
            return 5.0

    def run_topsis(self, options, criteria):

        norm_matrix = {}

        for crit in criteria:
            c_id = crit['id']
            sq_sum = sum(opt['values'][c_id] ** 2 for opt in options)
            denominator = math.sqrt(sq_sum) if sq_sum > 0 else 1

            for opt in options:
                norm_matrix.setdefault(opt['name'], {})
                norm_matrix[opt['name']][c_id] = (
                    opt['values'][c_id] / denominator
                ) * crit['weight']

        ideal_best = {}
        ideal_worst = {}

        for crit in criteria:
            c_id = crit['id']
            vals = [norm_matrix[o['name']][c_id] for o in options]

            if crit['type'] == 'benefit':
                ideal_best[c_id] = max(vals)
                ideal_worst[c_id] = min(vals)
            else:
                ideal_best[c_id] = min(vals)
                ideal_worst[c_id] = max(vals)

        results = []

        for opt in options:

            name = opt['name']

            d_best = math.sqrt(
                sum(
                    (norm_matrix[name][c['id']] - ideal_best[c['id']]) ** 2
                    for c in criteria
                )
            )

            d_worst = math.sqrt(
                sum(
                    (norm_matrix[name][c['id']] - ideal_worst[c['id']]) ** 2
                    for c in criteria
                )
            )

            score = d_worst / (d_best + d_worst) if (d_best + d_worst) > 0 else 0.5

            results.append({
                "name": name,
                "score": score
            })

        return sorted(results, key=lambda x: x['score'], reverse=True), ideal_best, ideal_worst, norm_matrix


    def simulate_decision(self, options, criteria, iterations=1000):

        win_counts = {opt['name']: 0 for opt in options}

        float_options = [
            {
                "name": opt['name'],
                "values": {c['id']: self._to_float(opt['values'].get(c['id'])) for c in criteria}
            }
            for opt in options
        ]

        for _ in range(iterations):

            shared_shift = {
                c['id']: random.gauss(0, 1.2)
                for c in criteria if c.get('dynamic')
            }

            sim_options = []

            for opt in float_options:

                temp_vals = opt['values'].copy()

                for c_id, shift in shared_shift.items():

                    temp_vals[c_id] = max(
                        1,
                        min(9, temp_vals[c_id] + shift + random.gauss(0, 0.3))
                    )

                sim_options.append({
                    "name": opt['name'],
                    "values": temp_vals
                })

            pass_results, _, _, _ = self.run_topsis(sim_options, criteria)

            win_counts[pass_results[0]['name']] += 1

        return sorted(
            [
                {
                    "name": n,
                    "confidence": round((c / iterations) * 100, 2)
                }
                for n, c in win_counts.items()
            ],
            key=lambda x: x['confidence'],
            reverse=True
        )


    def explain_all(self, options, criteria):

        results, ideal_best, ideal_worst, norm_matrix = self.run_topsis(options, criteria)

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

                c_id = crit['id']

                actual = opt['values'][c_id]
                ideal_val = ideal_raw[c_id]
                worst_val = worst_raw[c_id]

                raw_range = abs(ideal_val - worst_val)

                gap_pct = abs(actual - ideal_val) / raw_range * 100 if raw_range > 0 else 0.0

                explanation.append({
                    "name": crit['name'],
                    "actual": actual,
                    "ideal_raw": ideal_val,
                    "gap_pct": round(gap_pct, 1),
                    "weight": crit['weight']
                })

            explanation.sort(key=lambda x: x['gap_pct'])

            all_explanations[opt_name] = explanation

        return all_explanations, results



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    data = request.json

    engine = DecisionEngine()

    criteria_data = data['criteria']
    options_data = data['options']

    weights = engine.calculate_roc_weights(len(criteria_data))

    criteria = []

    for i, crit in enumerate(criteria_data):

        criteria.append({
            'id': f'c{i}',
            'name': crit['name'],
            'type': crit['type'],
            'dynamic': crit.get('dynamic', False),
            'weight': weights[i]
        })

    options = []

    for opt in options_data:

        vals = {}

        for i, v in enumerate(opt['values']):
            vals[f'c{i}'] = engine._to_float(v)

        options.append({
            'name': opt['name'],
            'values': vals
        })


    sim = engine.simulate_decision(options, criteria)

    explanations, topsis = engine.explain_all(options, criteria)

    return jsonify({
        "simulation_results": sim,
        "topsis_results": topsis,
        "explanations": explanations
    })


# IMPORTANT FOR VERCEL
def handler(request):
    return app(request.environ, lambda *args: None)
