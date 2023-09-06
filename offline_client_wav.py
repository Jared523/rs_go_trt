#!/bin/python3
# 此程序为电脑端录音程序，提供流式数据给到websocket端

import websockets
import asyncio
import json
import wave
import time
from recoder import Recoder

# wav_path="20230331-154401_cd90cf12cf9711ed.pcm.wav"
wav_path = ""

async def hello(uri):
    async with websockets.connect(uri) as websocket:
        websocket.binaryType = 'arraybuffer'
        recoder = Recoder(wav_path=wav_path)
        recoder.start()

        # await websocket.send('{"signal":"reload_context", "context_path":"", "context_score":5.0, "update":true}')
        # data = await websocket.recv()
        # print(data)
        # 三种模式  一种是流式识别   chunk_size 为   一种为一句话识别   说；，完之后 再识别 ，，更好，一个一个的，一动一动的，一顿一顿的，正好，，正好，一顿一顿的、，，、，，，那好，那好，：，：，那好，那好，做好么好（，：很好，：），：和号，：封号，：封号，由VAD切断 chunk_size 可设置为 1200 一种为离线识别  \
        #   可上传wav文件以及scp文件 chunk_size 设置为-1
        # await websocket.send('{ "signal": "start", "continuous_decoding": true, "nbest": 10}')
        while True:
            data, status = recoder.read()
            await websocket.send(data)
            data = await websocket.recv()
            data = json.loads(data)
            # if data["type"] == "final_result":
            print(data)
        await websocket.send('{ "signal": "end"}')
        await websocket.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(hello('ws://127.0.0.1:10088'))