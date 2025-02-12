import time
import websocket
import json
import base64
import logging
import os
import ssl
import threading

from dotenv import load_dotenv
from audio import AudioHandler
load_dotenv()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealTimeAgent:
    """
    Client for interacting with the OpenAI Realtime API via WebSocket.

    Possible events: https://platform.openai.com/docs/api-reference/realtime-client-events
    """

    def __init__(self, instructions, voice="alloy"):
        # WebSocket Configuration
        self.url = "wss://api.openai.com/v1/realtime"  # WebSocket URL
        self.model = "gpt-4o-mini-realtime-preview"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.ws = None
        self.audio_handler = AudioHandler()

        # SSL Configuration (skipping certificate verification)
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        self.audio_buffer = b''  # Buffer for streaming audio responses
        self.instructions = instructions
        self.voice = voice

        # VAD mode (set to null to disable)
        self.VAD_turn_detection = True
        self.VAD_config = {
            "type": "server_vad",
            # Activation threshold (0.0-1.0). A higher threshold will require louder audio to activate the model.
            "threshold": 0.5,
            # Audio to include before the VAD detected speech.
            "prefix_padding_ms": 300,
            # Silence to detect speech stop. With lower values the model will respond more quickly.
            "silence_duration_ms": 400
        }

        self.session_config = {
            "modalities": ["audio", "text"],
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": self.VAD_config if self.VAD_turn_detection else None,
            "input_audio_transcription": {  # Get transcription of user turns
                "model": "whisper-1"
            },
            "temperature": 0.6
        }

    def run(self):
        logger.info(f"Connecting to WebSocket: {self.url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = websocket.WebSocketApp(
            f"{self.url}?model={self.model}",
            header=headers,
            on_open=self.__on_open,
            on_message=self.__on_message,
            on_error=self.__on_error,
        )

        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    # To send a client event, serialize a dictionary to JSON
    # of the proper event type
    def __on_open(self, ws):
        logger.info("Connected to server.")

        # Configure session
        self.__send_event(
            {
                "type": "session.update",
                "session": self.session_config
            }
        )
        logger.info(f"Session set up to:\n{self.session_config}")

        # Start a separate thread to listen for audio input
        self.processing_thread = threading.Thread(target=self.__listen)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.listen_event = threading.Event()
        self.listen_event.set()

    # Receiving messages will require parsing message payloads from JSON
    def __on_message(self, ws, message):
        event = json.loads(message)
        # logger.info(f"Received event: {json.dumps(event, indent=2)}")
        if event["type"] == "response.audio.delta":
            # Append audio data to buffer
            audio_data = base64.b64decode(event["delta"])
            self.audio_handler.play_audio(audio_data)
        elif event["type"] == "response.audio.done":
            logger.info("Done playing audio response")
        elif event["type"] == "response.done":
            logger.debug("Response generation completed and starting to listen for audio input again")
            # Start to listen for audio input again
            self.listen_event.set()
        elif event["type"] == "conversation.item.created":
            logger.debug(f"Conversation item created: {event.get('item')}")
        elif event["type"] == "input_audio_buffer.speech_started":
            logger.debug("Speech started detected by server VAD")
        elif event["type"] == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped detected by server VAD")
            # Stop listening for audio input
            self.listen_event.clear()
        elif event["type"] == "session.created":
            logger.debug(f"Session created: {event.get('session')}")
        elif event["type"] == "session.updated":
            logger.debug(f"Session updated: {event.get('session')}")
        else:
            logger.debug(f"Unhandled event type: {event['type']}")

    def __on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def __listen(self):
        '''Keep sending audio chunks to the server'''
        self.audio_handler.start_recording()
        try:
            while True:
                self.listen_event.wait()
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    # Encode and send audio chunk
                    base64_chunk = base64.b64encode(chunk).decode('utf-8')
                    self.__send_event({
                        "type": "input_audio_buffer.append",
                        "audio": base64_chunk
                    })
                else:
                    logger.debug("No audio chunk received")
                    break
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")

        finally:
            # Stop recording even if an exception occurs
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")

    def __send_event(self, event):
        """
        Send an event to the WebSocket server.

        :param event: Event data to send (from the user)
        """
        self.ws.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")


if __name__ == "__main__":
    INSTRUCTIONS = '''
    你是一個講話非常有戲劇性，會用非常誇張語氣說話的人
    接下來的所有對話請你都要用非常誇張的語氣說話
    '''
    agent = RealTimeAgent(INSTRUCTIONS)
    agent.run()
