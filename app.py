import threading

from quart import Quart, render_template

from main import main

app = Quart(__name__)


@app.route('/')
async def home():
    return await render_template('index.html')


@app.route('/start-algorithm', methods=['POST'])
async def start_algorithm():
    thread = threading.Thread(target=main)
    thread.start()
    return {'message': 'Algorithm started successfully!'}


if __name__ == "__main__":
    app.run()
