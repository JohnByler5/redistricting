import threading

from quart import Quart, render_template

from main import main

app = Quart(__name__)
users = {}


@app.route('/')
async def home():
    return await render_template('index.html')


@app.route('/start-algorithm', methods=['POST'])
async def start_algorithm():
    thread = threading.Thread(target=main)
    thread.start()
    users['user_id'] = thread  # TODO: Actually implement user IDs
    return {'message': 'Algorithm started successfully!'}


@app.route('/stop-algorithm', methods=['POST'])
async def start_algorithm():
    thread = users['user_id']  # TODO: Actually implement user IDs
    # TODO: Kill thread here
    return {'message': 'Algorithm stopped successfully!'}


if __name__ == "__main__":
    app.run()
