import asyncio
import json
import os

from quart import current_app, render_template, send_from_directory, stream_with_context, make_response

from .run_algorithm import update_event, users, run_algorithm, quit_algorithm, get_results, exists


async def home():
    return await render_template('index.html')


async def favicon():
    return await send_from_directory(os.path.join(current_app.static_folder, 'images'), 'favicon.ico',
                                     mimetype='image/vnd.microsoft.icon')


async def start_algorithm():
    # Run in thread so that the event loop be not blocked
    await asyncio.to_thread(run_algorithm, 'user_id')    # TODO: Actually implement user IDs
    return {'message': 'Algorithm started successfully!'}


async def stop_algorithm():
    await quit_algorithm('user_id')  # TODO: Actually implement user IDs
    return {'message': 'Algorithm stopped successfully!'}


async def maps(filename):
    return await send_from_directory(os.path.join(current_app.root_path, 'maps'), filename)


async def events():
    @stream_with_context
    async def event_stream():
        while True:
            # Outer loop is for each new algorithm start
            if not await exists('user_id'):
                done, pending = await asyncio.wait([asyncio.create_task(update_event.wait())], timeout=10)
                if pending:
                    pending.pop().cancel()
                    # Keep connection alive
                    yield f'data: {json.dumps({"event": "heartbeat"})}\n\n'
                    continue

                update_event.clear()

            last = None
            while True:
                results, event = await get_results('user_id', timeout=10)  # TODO: Actually implement user IDs

                if results is None or results == last:
                    # Keep connection alive
                    yield f'data: {json.dumps({"event": "heartbeat"})}\n\n'
                    event.set()
                    continue
                if results == 'TIMEOUT':
                    # Keep connection alive
                    yield f'data: {json.dumps({"event": "heartbeat"})}\n\n'
                    continue
                if results in ['USER_ID_NOT_FOUND', 'OPERATION_COMPLETE']:
                    # Algorithm is stopped/finished
                    yield f"data: {json.dumps({'event': results})}\n\n"
                    event.set()
                    break

                yield f"data: {json.dumps(results)}\n\n"
                event.set()

    response = await make_response(event_stream(), 200)
    response.headers['Content-Type'] = 'text/event-stream'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    return response
