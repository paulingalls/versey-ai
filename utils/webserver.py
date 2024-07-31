import os
import json
import asyncio
import ssl
from aiohttp import web
from pyee.asyncio import AsyncIOEventEmitter

from utils.webrtcconnection import WebRTCConnection


class WebServer(AsyncIOEventEmitter):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.app = web.Application()
        self.connections = set()

    def setup_router(self):
        async def index(request):
            content = open(os.path.join(self.root, "index.html"), "r").read()
            return web.Response(content_type="text/html", text=content)

        async def javascript(request):
            content = open(os.path.join(self.root, "client.js"), "r").read()
            return web.Response(content_type="application/javascript", text=content)

        async def offer(request):
            params = await request.json()
            rtc = WebRTCConnection(sdp=params["sdp"], connection_type=params["type"])
            info = await rtc.handle_offer()
            self.connections.add(rtc)
            self.emit("connect", rtc)
            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"sdp": info["sdp"], "type": info["type"]}
                ),
            )

        self.app.router.add_get("/", index)
        self.app.router.add_get("/client.js", javascript)
        self.app.router.add_post("/offer", offer)

    async def shutdown(self):
        closed = [rtc.close for rtc in self.connections]
        await asyncio.gather(*closed)
        self.connections.clear()

    def start(self, host, port, cert_file, key_file):
        self.setup_router()

        async def on_shutdown(app):
            await self.shutdown()

        self.app.on_shutdown.append(on_shutdown)
        if cert_file:
            ssl_context = ssl.SSLContext()
            ssl_context.load_cert_chain(cert_file, key_file)
        else:
            ssl_context = None

        web.run_app(
            self.app, access_log=None, host=host, port=port, ssl_context=ssl_context
        )
