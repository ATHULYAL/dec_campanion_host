from flask import Flask, render_template, request, jsonify
import math
import random
import copy
import os

# Correct template path for Vercel
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


class DecisionEngine:

    def __init__(self):
        self.qualitative_map = {
            "very low": 1,
            "low": 3,
            "medium": 5,
            "high": 7,
            "very high": 9
        }


    def calculate_roc_weights(self, num_criteria):

        weights = []

        for i in range(1, num_criteria + 1):

            weight = sum(
                1.0 / j
                for j in range(i, num_criteria + 1)
            ) / num_criteria

            weights.append(weight)

        return weights


    def _to_float(self, value):

        if value is None:
            return 5.0

        if isinstance(value, str):

            try:
                return float(value.strip())

            except ValueError:

                return float(
                    self.qualitative_map.get(
                        value.lower().strip(),
                        5.0
                    )
                )

        try:
            return float(value)

        except:
            return 5.0


    def run_topsis(self, options, criteria):

        norm_matrix = {}

        for crit in criteria:

            c_id = crit['id']

            sq_sum = sum(
                opt['values'][c_id] ** 2
                for opt in options
            )

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

            vals = [
                norm_matrix[o['name']][c_id]
                for o in options
            ]

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

            score = (
                d_worst / (d_best + d_worst)
                if (d_best + d_worst) > 0
                else 0.5
            )

            results.append({
                "name": name,
                "score": score
            })

        return sorted(
            results,
            key=lambda x: x['score'],
            reverse=True
        )


    def simulate_decision(self, options, criteria, iterations=300):

        win_counts = {
            opt['name']: 0
            for opt in options
        }

        float_options = [
            {
                "name": opt['name'],
                "values": {
                    c['id']: self._to_float(
                        opt['values'].get(c['id'])
                    )
                    for c in criteria
                }
            }
            for opt in options
        ]

        for _ in range(iterations):

            shared_shift = {

                c['id']: random.gauss(0, 1.2)

                for c in criteria

                if c.get('dynamic')
            }

            sim_options = []

            for opt in float_options:

                temp_vals = opt['values'].copy()

                for c_id, shift in shared_shift.items():

                    temp_vals[c_id] = max(
                        1,
                        min(
                            9,
                            temp_vals[c_id]
                            + shift
                            + random.gauss(0, 0.3)
                        )
                    )

                sim_options.append({

                    "name": opt['name'],
                    "values": temp_vals

                })


            pass_results = self.run_topsis(sim_options, criteria)

            win_counts[
                pass_results[0]['name']
            ] += 1


        return sorted(

            [
                {
                    "name": n,
                    "confidence": round(
                        (c / iterations) * 100,
                        2
                    )
                }
                for n, c in win_counts.items()
            ],

            key=lambda x: x['confidence'],
            reverse=True
        )


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():

    data = request.json

    engine = DecisionEngine()

    criteria_data = data['criteria']
    options_data = data['options']

    weights = engine.calculate_roc_weights(
        len(criteria_data)
    )

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

    return jsonify({

        "simulation_results": sim

    })


# THIS LINE IS CRITICAL FOR VERCEL
app = app
