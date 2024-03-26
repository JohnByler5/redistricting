import asyncio
import copy
import json
import os
import threading

from quart import Quart, render_template, send_from_directory, url_for, websocket

from main import main

app = Quart(__name__)
users = {}

update_event = asyncio.Event()
lock = asyncio.Lock()


@app.route('/')
async def home():
    return await render_template('index.html')


@app.route('/start-algorithm', methods=['POST'])
async def start_algorithm():
    global users

    q = asyncio.Queue()
    thread = threading.Thread(target=main, args=(asyncio.get_event_loop(), q))
    thread.start()
    async with lock:
        users['user_id'] = (thread, q)  # TODO: Actually implement user IDs
        update_event.set()
    return {'message': 'Algorithm started successfully!'}


@app.route('/stop-algorithm', methods=['POST'])
async def stop_algorithm():
    global users

    async with lock:
        thread, _ = users['user_id']  # TODO: Actually implement user IDs
    # TODO: Kill thread here
    return {'message': 'Algorithm stopped successfully!'}


@app.route('/maps/images/<filename>')
async def images(filename):
    return await send_from_directory(os.path.join(app.root_path, 'maps', 'images'), filename)


@app.websocket('/ws')
async def ws():
    global users

    while True:
        # Outer loop is for each run cancellation or finish
        await update_event.wait()
        update_event.clear()

        last = None
        while True:
            await asyncio.sleep(1.0)
            if 'user_id' in users:
                async with lock:
                    _, q = users['user_id']  # TODO: Actually implement user IDs
                results = None
                # Clear the queue and get the last item
                print('Getting')
                while not q.empty():
                    results = await q.get()
                    q.task_done()
                if results is None:
                    continue
                print('Got')
                if results == 'DONE':
                    # Thread is finished
                    async with lock:
                        users.pop('user_id')
                    break
                if results == last:
                    continue
                last = copy.deepcopy(results)
                for key in ['currentMap', 'solutionMap']:
                    filename = results[key]['imageUrl'][results[key]['imageUrl'].rindex('/') + 1:]
                    results[key]['imageUrl'] = url_for('images', filename=filename, _external=True)
                await websocket.send(json.dumps(results))
                print('Sent')


if __name__ == "__main__":
    app.run()
