import asyncio
import json

from quart import url_for

from redistricting_utils.run import create_algorithm

users = {}

lock = asyncio.Lock()
update_event = asyncio.Event()

with open('redistricting_app/config.json', 'r') as f:
    CONFIG = json.load(f)


def run_algorithm(user_id, params):
    async def run():
        global lock, update_event, users

        algorithm = create_algorithm(
            config=CONFIG,
            state=params['state'],
            population_size=params['population_size'],
            starting_population_size=params['starting_population_size'],
            selection_pct=params['selection_pct'],
            weights=params['weights'],
        )
        generations = params['generations']
        update_every = 10

        q = asyncio.Queue()
        async with lock:
            users[user_id] = q  # TODO: Actually implement user IDs
            update_event.set()

        wait_event = asyncio.Event()
        for i, update in enumerate(algorithm.run(generations=generations), 1):
            if user_id not in users:
                break

            if i == 1 or i % update_every == 0:
                # Change file paths to URLs
                for map_type in ['currentMap', 'solutionMap']:
                    if map_type in update:
                        path = update[map_type]['imageUrl'][update[map_type]['imageUrl'].index('maps') + 5:]
                        update[map_type]['imageUrl'] = url_for('maps', filename=path)

                await q.put((update, wait_event))
                done, pending = await asyncio.wait([asyncio.create_task(wait_event.wait())], timeout=10)
                if pending:
                    pending.pop().cancel()
                    q.empty()
                    wait_event.clear()

        await q.put(('OPERATION_COMPLETE', asyncio.Event()))  # Indicate it is done

    asyncio.run(run())


async def quit_algorithm(user_id):
    global lock, users

    async with lock:
        users.pop(user_id)


async def get_results(user_id, timeout):
    global users

    async with lock:
        q = users.get(user_id, None)
        if q is None:
            return 'USER_ID_NOT_FOUND', None
        done, pending = await asyncio.wait([asyncio.create_task(q.get())], timeout=timeout)
        if done:
            return done.pop().result()  # (update, event)
        pending.pop().cancel()
        return 'TIMEOUT', None


async def exists(user_id):
    return user_id in users
