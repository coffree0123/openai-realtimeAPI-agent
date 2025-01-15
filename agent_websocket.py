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

logging.basicConfig(filename='my_log.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
            "threshold": 0.5,  # Activation threshold (0.0-1.0). A higher threshold will require louder audio to activate the model.
            "prefix_padding_ms": 300,  # Audio to include before the VAD detected speech.
            "silence_duration_ms": 600  # Silence to detect speech stop. With lower values the model will respond more quickly.
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

    def listen(self):
        '''Keep sending audio chunks to the server'''
        self.audio_handler.start_recording()
        try:
            while True:
                chunk = self.audio_handler.record_chunk()
                if chunk:
                    # Encode and send audio chunk
                    base64_chunk = base64.b64encode(chunk).decode('utf-8')
                    self.send_event({
                        "type": "input_audio_buffer.append",
                        "audio": base64_chunk
                    })
                else:
                    break
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")
    
        finally:
            # Stop recording even if an exception occurs
            self.audio_handler.stop_recording()
            logger.debug("Audio recording stopped")


    def run(self):
        logger.info(f"Connecting to WebSocket: {self.url}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        self.ws = websocket.WebSocketApp(
            f"{self.url}?model={self.model}",
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
        )

        self.ws.run_forever()

    # To send a client event, serialize a dictionary to JSON
    # of the proper event type
    def on_open(self, ws):
        logger.info("Connected to server.")

        # Configure session
        self.send_event(
            {
                "type": "session.update",
                "session": self.session_config
            }
        )
        logger.info(f"Session set up to:\n{self.session_config}")

        # Start a separate thread to listen for audio input
        processing_thread = threading.Thread(target=self.listen)
        processing_thread.daemon = True
        processing_thread.start()

    # Receiving messages will require parsing message payloads
    # from JSON
    def on_message(self, ws, message):
        event = json.loads(message)
        #print("Received event:", json.dumps(event, indent=2))
        if event["type"] == "response.audio.delta":
            # Access Base64-encoded audio chunks:
            # Append audio data to buffer
            audio_data = base64.b64decode(event["delta"])
            self.audio_buffer += audio_data
            if len(self.audio_buffer) > 50000:
                self.audio_handler.play_audio(self.audio_buffer)
                self.audio_buffer = b''
        elif event["type"] == "response.audio.done":
            pass
            # Play the complete audio response
            # if self.audio_buffer:
            #     self.audio_handler.play_audio(self.audio_buffer)
            #     logger.info("Done playing audio response")
            #     self.audio_buffer = b''
            # else:
            #     logger.warning("No audio data to play")
        elif event["type"] == "response.done":
            logger.debug("Response generation completed")
        elif event["type"] == "conversation.item.created":
            logger.debug(f"Conversation item created: {event.get('item')}")
        elif event["type"] == "input_audio_buffer.speech_started":
            logger.debug("Speech started detected by server VAD")
        elif event["type"] == "input_audio_buffer.speech_stopped":
            logger.debug("Speech stopped detected by server VAD")
        else:
            logger.debug(f"Unhandled event type: {event['type']}")
    

    def send_event(self, event):
        """
        Send an event to the WebSocket server.
        
        :param event: Event data to send (from the user)
        """
        self.ws.send(json.dumps(event))
        logger.debug(f"Event sent - type: {event['type']}")


if __name__ == "__main__":
    INSTRUCTIONS = f"""
    你是一名幽默溫和的聊天機器人, 但你只會說中文.
    """
    agent = RealTimeAgent(INSTRUCTIONS)
    agent.run()