#!/usr/bin/env python
"""
Quick Start Script for Front-End Mentor Bot
Run this file from the PROJECT ROOT to start the application:
    python run.py
"""

import os
import sys

# Add the app/ directory to Python path so imports inside app.py resolve correctly
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'app'))

# Import the Flask app object from app/app.py
from app import app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("\n" + "=" * 60)
    print("🎓 FRONT-END MENTOR BOT")
    print("=" * 60)
    print("\nStarting application...")
    print(f"Open your browser to: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    # Run Flask app
    app.run(host='0.0.0.0', port=port, debug=True)
