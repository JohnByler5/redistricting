
from quart import Quart

from .views import home, favicon, start_algorithm, stop_algorithm, maps, events

app = Quart(__name__)

app.route('/')(home)
app.route('/favicon.ico')(favicon)
app.route('/start-algorithm', methods=['POST'])(start_algorithm)
app.route('/stop-algorithm', methods=['POST'])(stop_algorithm)
app.route('/maps/<path:filename>')(maps)
app.route('/events')(events)

if __name__ == "__main__":
    app.run()
