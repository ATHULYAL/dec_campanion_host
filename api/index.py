from flask import Flask, render_template, request, jsonify
import math
import random
import copy
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)


class DecisionEngine:

    def __init__(self):
        self.qualitative_map = {
            "very low":1,"low":3,"medium":5,
            "high":7,"very high":9
        }


    def calculate_roc_weights(self,n):

        weights=[]

        for i in range(1,n+1):

            w=sum(1/j for j in range(i,n+1))/n

            weights.append(w)

        return weights


    def _to_float(self,v):

        try:
            return float(v)
        except:

            return float(
                self.qualitative_map.get(
                    str(v).lower().strip(),5
                )
            )


    def run_topsis(self,options,criteria):

        norm={}

        for c in criteria:

            cid=c['id']

            sq=sum(o['values'][cid]**2 for o in options)

            den=math.sqrt(sq) if sq>0 else 1

            for o in options:

                norm.setdefault(o['name'],{})

                norm[o['name']][cid]=(
                    o['values'][cid]/den
                )*c['weight']


        best={}
        worst={}

        for c in criteria:

            cid=c['id']

            vals=[norm[o['name']][cid] for o in options]

            if c['type']=="benefit":

                best[cid]=max(vals)
                worst[cid]=min(vals)

            else:

                best[cid]=min(vals)
                worst[cid]=max(vals)


        results=[]

        for o in options:

            name=o['name']

            d1=math.sqrt(sum(
                (norm[name][c['id']]-best[c['id']])**2
                for c in criteria
            ))

            d2=math.sqrt(sum(
                (norm[name][c['id']]-worst[c['id']])**2
                for c in criteria
            ))

            score=d2/(d1+d2) if d1+d2>0 else 0.5

            results.append({
                "name":name,
                "score":score
            })


        return sorted(
            results,
            key=lambda x:x['score'],
            reverse=True
        )


    def simulate(self,options,criteria):

        counts={o['name']:0 for o in options}

        for _ in range(300):

            sim=[]

            for o in options:

                vals=o['values'].copy()

                for c in criteria:

                    if c['dynamic']:

                        vals[c['id']]=max(
                            1,
                            min(9,
                                vals[c['id']]
                                +random.gauss(0,1)
                            )
                        )

                sim.append({
                    "name":o['name'],
                    "values":vals
                })


            res=self.run_topsis(sim,criteria)

            counts[res[0]['name']]+=1


        return sorted(

            [
                {
                    "name":n,
                    "confidence":round(c/3,1)
                }
                for n,c in counts.items()
            ],

            key=lambda x:x['confidence'],
            reverse=True
        )



@app.route("/")
def index():

    return render_template("index.html")



@app.route("/analyze",methods=["POST"])
def analyze():

    data=request.json

    engine=DecisionEngine()

    goal=data['goal']

    criteria_data=data['criteria']

    options_data=data['options']


    weights=engine.calculate_roc_weights(
        len(criteria_data)
    )


    criteria=[]

    for i,c in enumerate(criteria_data):

        criteria.append({

            "id":f"c{i}",
            "name":c['name'],
            "type":c['type'],
            "dynamic":c['dynamic'],
            "weight":weights[i]

        })


    options=[]

    for o in options_data:

        vals={}

        for i,v in enumerate(o['values']):

            vals[f"c{i}"]=engine._to_float(v)

        options.append({

            "name":o['name'],
            "values":vals

        })


    sim=engine.simulate(options,criteria)

    winner=sim[0]['name']


    breakdown=[]

    for i,o in enumerate(sim):

        breakdown.append({

            "name":o['name'],
            "rank":i+1,
            "confidence":o['confidence'],
            "strengths":[],
            "weaknesses":[]

        })


    return jsonify({

        "goal":goal,

        "criteria":[

            {
                "name":c['name'],
                "type":c['type'],
                "dynamic":c['dynamic'],
                "weight":round(c['weight']*100,2)

            }

            for c in criteria

        ],

        "simulation_results":sim,

        "reasoning":

        f"{winner} performed best overall across simulated futures.",

        "option_breakdown":breakdown,

        "gap_note":

        "Gap values depend on entered options."

    })


app = app
