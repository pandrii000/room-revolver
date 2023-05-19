from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response, JSONResponse
from typing import List
from pathlib import Path
from PIL import Image
from shutil import copy
from PIL import Image
import requests
import datetime
from typing import Optional, Any
from io import BytesIO
from inference import transcribe, define, remove, create  # replace, create
import os.path
from fastapi.middleware.cors import CORSMiddleware
from googletrans import Translator

translator = Translator()


def is_local_file(link):
    return os.path.isfile(link) and not os.path.islink(link)


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://x-seven.test.ut.in.ua",
    "http://localhost:8989",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


audio_dir = Path("audio")
result_dir = Path("result")

audio_dir.mkdir(exist_ok=True, parents=True)
result_dir.mkdir(exist_ok=True, parents=True)


@app.post("/process")
async def process(
    image: str,
    x: int,
    y: int,
    text: Optional[str] = None,
    audio: Optional[str] = None,
):
    if text is None:
        audio_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + ".mp3"
        audio_fpath = audio_dir / audio_fname

        if not is_local_file(audio):
            audio_data = requests.get(audio)
            # Save file data to local copy
            with open(audio_fpath, "wb") as f:
                f.write(audio_data.content)

        else:
            copy(audio, audio_fpath)

        text = transcribe(str(audio_fpath))

    text = translator.translate(text, dest="en").text

    actions = define(text)
    print(actions)

    response = requests.get(image)
    image_content = BytesIO(response.content)
    image: Image = Image.open(image_content)
    result: Image = None

    for task, target in actions["actions"]:
        print(task, target)

        if task == "remove":
            image = remove(image, None, x, y)
        if task == "create":
            image = create(image, {"object": target}, x, y)

        assert image is not None, image

    result = image.copy()

    result_id: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    result.save(result_dir / f"{result_id}.png")

    # Return the processed image and text as the response
    response = {"result_id": result_id, "text": text}
    return JSONResponse(response)


@app.post("/get_result")
async def get_result(result_id: str):
    image_path = result_dir / f"{result_id}.png"
    return FileResponse(image_path)


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host='0.0.0.0', port=8989)
    uvicorn.run(app, host="0.0.0.0", port=50337)
