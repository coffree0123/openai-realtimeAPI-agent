import pyaudio

class AudioHandler:
    """
    Handles audio input and output using PyAudio.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.chunk_size = 1024  # Number of audio frames per buffer
        self.format = pyaudio.paInt16  # Audio format (16-bit PCM)
        self.channels = 1  # Mono audio
        self.rate = 24000  # Sampling rate in Hz

        self.input_stream = None
        self.output_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            output=True
        )

    def start_recording(self):
        """
        Start the audio input stream.
        """
        self.input_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

    def stop_recording(self):
        """
        Clean up resources by stopping the stream and terminating PyAudio.
        """
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        self.p.terminate()

    def record_chunk(self):
        """Record a single chunk of audio"""
        if self.input_stream:
            return self.input_stream.read(self.chunk_size)
        return None
    
    def play_audio(self, audio_data):
        """
        Play audio data.
        
        :param audio_data: Received audio data (AI response)
        """
        self.output_stream.write(audio_data)
