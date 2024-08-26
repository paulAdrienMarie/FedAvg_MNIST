# put this in a web/ directory
# rename to server.py and use a __main__ section

from aiohttp import web
from routes import setup_routes

# use a constant with an explicit name so we understand what it is
app = web.Application(client_max_size=1024**2 * 10000000000)

setup_routes(app)

web.run_app(app)
