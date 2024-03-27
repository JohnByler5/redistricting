import asyncio
import copy
import json
import os
import threading

from quart import Quart, render_template, send_from_directory, url_for, websocket

from main import create_algorithm

app = Quart(__name__)
users = {}

update_event = asyncio.Event()
lock = asyncio.Lock()


def run_algorithm(algorithm, generations, event_loop=None, queue=None):
    def put(item):
        # Add an item to the async queue, if it exists
        if queue is not None and event_loop is not None:
            asyncio.run_coroutine_threadsafe(queue.put(item), event_loop)

    # This will likely take several hours
    for update in algorithm.run(generations=generations):
        put(update)

    put('OPERATION_COMPLETE')  # Indicate it is done


@app.route('/')
async def home():
    return await render_template('index.html')


@app.route('/start-algorithm', methods=['POST'])
async def start_algorithm():
    global users

    q = asyncio.Queue()
    thread = threading.Thread(target=run_algorithm, kwargs={
        'algorithm': create_algorithm(),
        'generations': 100_000,
        'event_loop': asyncio.get_event_loop(),
        'queue': q,
    })
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
async def websocket():
    global users

    while True:
        # Outer loop is for each run cancellation or finish
        await update_event.wait()
        update_event.clear()

        last = None
        while True:
            await asyncio.sleep(0.5)

            results = await get_results('user_id')  # TODO: Actually implement user IDs
            if results is None:
                continue
            if results == 'OPERATION_COMPLETE':
                # Algorithm is finished
                async with lock:
                    users.pop('user_id')
                await websocket.send(json.dumps({'event': 'OPERATION_COMPLETE'}))
                break

            if results == last:
                continue
            last = copy.deepcopy(results)

            for key in ['currentMap', 'solutionMap']:
                filename = results[key]['imageUrl'][results[key]['imageUrl'].rindex('/') + 1:]
                results[key]['imageUrl'] = url_for('images', filename=filename, _external=True)
            await websocket.send(json.dumps(results))


async def get_results(user_id):
    global users

    results = None
    async with lock:
        if user_id in users:
            _, q = users[user_id]
        # Clear the queue and get the last item
        while not q.empty():
            results = await q.get()
            q.task_done()

    return results


if __name__ == "__main__":
    app.run()
