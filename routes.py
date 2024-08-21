from aiohttp import web
import os
from views import index, style, update_model
HERE = os.path.dirname(__file__)


def setup_routes(app):
    app.add_routes([
    web.static("/script", os.path.join(HERE,"static")),
    web.static("/artifacts", os.path.join(HERE,'artifacts')),
    web.static("/model", os.path.join(HERE,"model")),
    web.static("/dist", os.path.join(HERE,"dist"))
    ])
    app.router.add_get('/', index)
    app.router.add_get('/style.css', style)
    app.router.add_post('/update_model', update_model)

    