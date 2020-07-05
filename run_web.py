from gevent.pywsgi import WSGIServer
from web_test import app

http_server = WSGIServer(('0.0.0.0', 5000), app, log=app.logger)
http_server.serve_forever()