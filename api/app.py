import json
from flask import Flask
from flask import request, abort, Response
from flask_restful import reqparse

# Personal imports:

from modules import clean_text

# Support functions:




app = Flask(__name__)

### APP BODY:

@app.route("/classify", methods=["POST"])
def classify():
    
    response = {
            "tag_1": {"tag":"",
                      "confidence":""
                      },
            "tag_2": {"tag":"",
                      "confidence":""
                      },
                      
            "tag_3": {"tag":"",
                      "confidence":""
                      },
            "tag_4": {"tag":"",
                      "confidence":""
                      },
            "tag_5": {"tag":"",
                      "confidence":""
                      }
            }

    if not request.is_json:
        request.get_json(force=True)
        
    
    # Parsing the request to retrieve the argument:
    body_parser = reqparse.RequestParser()
    body_parser.add_argument("text", type=str, required=True)

    args = body_parser.parse_args()
    
    text = args["text"]
    text = clean_text(text)
    
    if text is "":
        abort(400, {"error_message": "Missing or inconsistent (only common words) input text"})

    
    
    return Response(response=json.dumps(response), status=200, mimetype="json")
    
    
### LAUNCHING THE APP:

if __name__ == "__main__":
    app.run(debug=True)
