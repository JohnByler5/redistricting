import asyncio
import copy
import json
import os

from quart import current_app, render_template, send_from_directory, url_for, websocket

from .run_algorithm import lock, update_event, users, run_algorithm, quit_algorithm, get_results


async def home():
    return await render_template('index.html')


async def start_algorithm():
    await run_algorithm('used_id')    # TODO: Actually implement user IDs
    return {'message': 'Algorithm started successfully!'}


async def stop_algorithm():
    await quit_algorithm('user_id')  # TODO: Actually implement user IDs
    return {'message': 'Algorithm stopped successfully!'}


async def images(filename):
    return await send_from_directory(os.path.join(current_app.root_path, 'maps', 'solutions', 'images'), filename)


async def ws():
    while True:
        print('Waiting')

        # Outer loop is for each run cancellation or finish
        await update_event.wait()
        update_event.clear()

        print('Entering loop')

        last = None
        while True:
            await asyncio.sleep(0.5)

            print('Getting')
            results = await get_results('user_id')  # TODO: Actually implement user IDs
            print('Got', results)
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
            print('Sent')
