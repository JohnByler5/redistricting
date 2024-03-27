import asyncio
import json
import threading

from redistricting_utils.run import create_algorithm

users = {}

lock = asyncio.Lock()
update_event = asyncio.Event()

with open('redistricting_app/config.json', 'r') as f:
    CONFIG = json.load(f)


async def run_algorithm(user_id):
    global lock, update_event, users

    algorithm = create_algorithm(CONFIG)
    generations = 100_000

    def run(event_loop):
        def put(item):
            # Add an item to the async queue
            asyncio.run_coroutine_threadsafe(q.put(item), event_loop)

        # This will likely take several hours
        for update in algorithm.run(generations=generations):
            put(update)

        put('OPERATION_COMPLETE')  # Indicate it is done

    q = asyncio.Queue()
    thread = threading.Thread(target=run, args=(asyncio.get_event_loop(),))
    thread.start()
    async with lock:
        users[user_id] = (thread, q)  # TODO: Actually implement user IDs
        update_event.set()
        print('Set')


async def quit_algorithm(user_id):
    global lock, users

    async with lock:
        thread, _ = users[user_id]

    # TODO: Kill thread here


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
