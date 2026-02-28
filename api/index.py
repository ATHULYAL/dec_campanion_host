from flask import Flask, render_template, request, jsonify
import math
import random
import copy
import os
from collections import Counter

app = Flask(__name__, template_folder="templates")


class DecisionEngine:

    def __init__(self):

        self.qualitative_map = {
            "very low":1,
            "low":3,
            "medium":5,
            "high":7,
            "very high":9
        }

        self.value_label_map = {
            1:"very low",
            2:"very low",
            3:"low",
            4:"low",
            5:"medium",
            6:"medium",
            7:"high",
            8:"high",
            9:"very high"
        }


    def value_to_label(self,value):

        try:
            v=float(value)
        except:
            v=5

        r=max(1,min(9,round(v)))

        return self.value_label_map.get(r,"medium")


    def calculate_roc_weights_with_ties(self,priorities):

        n=len(priorities)

        base=[]

        for i in range(1,n+1):

            w=sum(1/j for j in range(i,n+1))/n

            base.append(w)

        unique=sorted(set(priorities))

        slot=0

        slots={}

        for p in unique:

            c=priorities.count(p)

            slots[p]=list(range(slot,slot+c))

            slot+=c

        weights_map={

            p:sum(base[s] for s in sl)/len(sl)

            for p,sl in slots.items()

        }

        return [weights_map[p] for p in priorities]


    def to_float(self,v):

        try:
            return float(v)

        except:

            return float(

                self.qualitative_map.get(

                    str(v).lower().strip(),

                    5

                )

            )


    def run_topsis(self,options,criteria):

        norm={}

        for c in criteria:

            cid=c["id"]

            s=sum(o["values"].get(cid,5)**2 for o in options)

            d=math.sqrt(s) if s>0 else 1

            for o in options:

                norm.setdefault(o["name"],{})

                norm[o["name"]][cid]=(o["values"].get(cid,5)/d)*c["weight"]


        best={}
        worst={}

        for c in criteria:

            cid=c["id"]

            vals=[norm[o["name"]][cid] for o in options]

            if c["type"]=="benefit":

                best[cid]=max(vals)
                worst[cid]=min(vals)

            else:

                best[cid]=min(vals)
                worst[cid]=max(vals)


        results=[]

        for o in options:

            name=o["name"]

            d1=math.sqrt(sum(
                (norm[name][c["id"]]-best[c["id"]])**2
                for c in criteria
            ))

            d2=math.sqrt(sum(
                (norm[name][c["id"]]-worst[c["id"]])**2
                for c in criteria
            ))

            score=d2/(d1+d2) if d1+d2>0 else 0.5

            results.append({

                "name":name,
                "score":score

            })


        return sorted(results,key=lambda x:x["score"],reverse=True)


    def simulate(self,options,criteria):

        counts={o["name"]:0 for o in options}

        for _ in range(100):

            sim=[]

            for o in options:

                vals=o["values"].copy()

                for c in criteria:

                    if c["dynamic"]:

                        vals[c["id"]]=max(
                            1,
                            min(
                                9,
                                vals.get(c["id"],5)+random.gauss(0,1)
                            )
                        )

                sim.append({

                    "name":o["name"],
                    "values":vals

                })

            res=self.run_topsis(sim,criteria)

            if res:

                counts[res[0]["name"]]+=1


        return [

            {

                "name":n,
                "confidence":round(c,1)

            }

            for n,c in counts.items()

        ]


@app.route("/")
def index():

    return render_template("index.html")


@app.route("/analyze",methods=["POST"])
def analyze():

    try:

        data=request.get_json() or {}

        engine=DecisionEngine()

        criteria_data=data.get("criteria",[])
        options_data=data.get("options",[])


        if not isinstance(criteria_data,list):
            criteria_data=[]

        if not isinstance(options_data,list):
            options_data=[]


        if len(criteria_data)==0:

            return jsonify({"error":"Add criteria"})


        if len(options_data)==0:

            return jsonify({"error":"Add options"})


        priorities=[

            int(c.get("priority",i+1))

            for i,c in enumerate(criteria_data)

        ]


        weights=engine.calculate_roc_weights_with_ties(priorities)


        criteria=[]

        for i,c in enumerate(criteria_data):

            criteria.append({

                "id":f"c{i}",
                "name":c.get("name","Criterion"),
                "type":c.get("type","benefit"),
                "dynamic":c.get("dynamic",False),
                "weight":weights[i]

            })


        options=[]

        for o in options_data:

            vals={}

            for i,v in enumerate(o.get("values",[])):

                vals[f"c{i}"]=engine.to_float(v)

            options.append({

                "name":o.get("name","Option"),
                "values":vals

            })


        sim=engine.simulate(options,criteria)

        winner=sim[0]["name"] if sim else "None"

        reasoning=f"Best option is {winner}"


        return jsonify({

            "simulation_results":sim,
            "reasoning":reasoning

        })


    except Exception as e:

        return jsonify({

            "error":str(e)

        })
