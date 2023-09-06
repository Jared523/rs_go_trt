import pyaudio
import wave
import time
import threading

class Recoder:
    def __init__(self,
                wav_path=""):
        self.chunk = 5120
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.p = pyaudio.PyAudio()	#初始化
        self.mutex = threading.Lock()
        self.frames = []
        self.wav_path = wav_path

    def start(self):
        if self.wav_path == "":
            self.stream = self.p.open(format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.callback)#创建录音文件
        else:
            self.stream = wave.open(self.wav_path, 'rb')
            self.num_frames = self.stream.getnframes()

    def callback(self, in_data, frame_count, time_info, status):
        self.mutex.acquire()
        self.frames.append(in_data)
        self.mutex.release()
        return (in_data, pyaudio.paContinue)     

    def read(self):
        if self.wav_path == "":
            while len(self.frames) == 0:
                time.sleep(0.01)
            self.mutex.acquire()
            data = b''.join(self.frames)
            self.frames = []
            self.mutex.release()
            return data, True
        else:
            data = self.stream.readframes(self.chunk)
            self.num_frames -= self.chunk
            if self.num_frames <= 0:
                status = False
            else:
                status = True
            return data, status

    def end(self):
        if self.wav_path == "":
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    recoder = Recoder()
    recoder.start()
    frames = []
    for i in range(40):
        data = recoder.read()
        frames.append(data)
        time.sleep(0.4)
    recoder.end()
    
    wf = wave.open("test.wav", 'wb')	#保存
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    

            