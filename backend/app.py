"""
Flask Application Entry Point
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from api_routes import api

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://localhost:3001', 'http://localhost:5173'])
app.register_blueprint(api)
app.register_blueprint(api, url_prefix="/api")
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Code Review Environment',
        'version': '2.0.0',
        'description': 'OpenEnv RL environment for AI code review agents',
        'endpoints': {
            'reset': 'POST /reset',
            'step': 'POST /step',
            'state': 'GET /state',
            'health': 'GET /health'
        },
        'tasks': [
            {'id': 1, 'name': 'Bug Detection', 'difficulty': 'easy'},
            {'id': 2, 'name': 'Bug Classification', 'difficulty': 'medium'},
            {'id': 3, 'name': 'Fix Suggestion', 'difficulty': 'hard'}
        ],
        'features': [
            'Dynamic code generation via OpenAI',
            'Shaped reward (step penalty + confidence bonus)',
            'OpenEnv compliant (openenv.yaml)',
            'Infinite episode variety'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)