# ai_comm_server.py

from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

@app.route('/receive-message', methods=['POST'])
def receive_message():
    data = request.json
    sender = data.get("from_agent", "Unknown")
    message = data.get("message", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n📥 Received message from {sender} at {timestamp}")
    print(f"📝 Message content:\n{message}\n")

    # Simulated AI logic response
    response_message = f"✔️ Hello {sender}, your message was received at {timestamp}."

    return jsonify({
        "status": "received",
        "response": response_message
    })

if __name__ == '__main__':
    print("🚀 AI Agent Receiver running on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001)
