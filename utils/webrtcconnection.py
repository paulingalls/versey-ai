import logging
import uuid
from aiortc import RTCSessionDescription, RTCPeerConnection, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from pyee.asyncio import AsyncIOEventEmitter
from utils.transform import Transform


class WebRTCConnection(AsyncIOEventEmitter):
    def __init__(self, sdp, connection_type):
        super().__init__()
        self.sdp = sdp
        self.type = connection_type
        self.logger = logging.getLogger("pc")
        self.pc = RTCPeerConnection()
        self.pc_id = "PeerConnection(%s)" % uuid.uuid4()
        self.sink = MediaBlackhole()
        self.relay = MediaRelay()
        self.datachannel = None

    def log_info(self, msg, *args):
        self.logger.info(self.pc_id + " " + msg, *args)

    async def handle_offer(self):
        offer = RTCSessionDescription(sdp=self.sdp, type=self.type)
        self.setup_connection()
        await self.pc.setRemoteDescription(offer)
        self.log_info("Offer created for %s", self.pc_id)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    def close(self):
        return self.pc.close()

    def setup_connection(self):
        @self.pc.on("datachannel")
        def on_datachannel(channel):
            self.datachannel = channel

            @channel.on("message")
            def on_message(message):
                if isinstance(message, str):
                    self.log_info("Received %s", message)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.log_info("Connection state is %s", self.pc.connectionState)
            if self.pc.connectionState == "failed":
                await self.pc.close()
                self.connection_state = "closed"

        @self.pc.on("track")
        def on_track(track: MediaStreamTrack):
            self.log_info("Track %s received", track.kind)

            if track.kind == "audio":
                self.audio_track = track
                self.transform = Transform(self.relay.subscribe(track))
                self.pc.addTrack(self.transform)
                self.sink.addTrack(track)

                @self.transform.on("voiceStart")
                def on_frame_data(data):
                    self.send_string(f"voiceStart: {data}")

                @self.transform.on("voiceEnd")
                def on_frame_data(data):
                    self.send_string(f"voiceEnd: {data}")

            @track.on("ended")
            async def on_ended():
                self.log_info("Track %s ended", track.kind)
                await self.sink.stop()

    def send_string(self, string_to_send):
        if self.datachannel is not None:
            self.datachannel.send(string_to_send)
