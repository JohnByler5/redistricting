from quart import Quart

from .views import home, start_algorithm, stop_algorithm, images, ws

app = Quart(__name__)

app.route('/')(home)
app.route('/start-algorithm', methods=['POST'])(start_algorithm)
app.route('/stop-algorithm', methods=['POST'])(stop_algorithm)
app.route('/maps/images/<filename>')(images)
app.websocket('/ws')(ws)

if __name__ == "__main__":
    app.run()
