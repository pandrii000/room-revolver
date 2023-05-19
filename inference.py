import sys

sys.path.append("/root/Inpaint-Anything")
sys.path.append("/root/Inpaint-Anything/lama")

from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from omegaconf import OmegaConf
from pathlib import Path
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from segment_anything import SamPredictor, sam_model_registry
from stable_diffusion_inpaint import fill_img_with_sd
from typing import List
from utils import dilate_mask
from utils.mask_processing import crop_for_filling_pre, crop_for_filling_post
import json
import numpy as np
import os
import requests
import torch
import whisper
import yaml


device_sd = "cuda:0"
device_sam = "cuda:1"
device_lama = "cuda:1"
device_whisper = "cuda:1"

audio_tmp = Path("/root/app/audio_tmp")


def load_lama(config_p, ckpt_p):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device_lama)

    train_config_path = os.path.join(predict_config.model.path, "config.yaml")

    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = "noop"

    checkpoint_path = os.path.join(
        predict_config.model.path, "models", predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location="cpu"
    )
    model.freeze()
    if not predict_config.get("refine", False):
        model.to(device)

    return model, predict_config


model_whisper = whisper.load_model("base", device=device_whisper)
model_sd = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
).to(device_sd)
model_sam = sam_model_registry["vit_h"](
    checkpoint="/root/EditAnything/models/sam_vit_h_4b8939.pth"
)
model_sam.to(device=device_sam)
model_sam_predictor = SamPredictor(model_sam)
model_lama, model_lama_config = load_lama(
    config_p="/root/Inpaint-Anything/lama/configs/prediction/default.yaml",
    ckpt_p="/root/Inpaint-Anything/pretrained_models/big-lama",
)

prompt_gpt_actions_extraction = """
Take the text and convert it to two variables, 
the first is actions, the second is item. 
Output the result in json format of:
{"actions":
[[action1, item1], [action2, item2], ...]
}

The only available actions: "create", "remove". 
If there is a replace action, break it into the "remove"-"create" sequences.
Text: """


def predict_masks_with_sam(
    img: np.ndarray,
    point_coords: List[List[float]],
    point_labels: List[int],
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    model_sam_predictor.set_image(img)
    masks, scores, logits = model_sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    return masks, scores, logits


def fill_img_with_sd(
    img: np.ndarray,
    mask: np.ndarray,
    text_prompt: str,
):
    img_crop, mask_crop = crop_for_filling_pre(img, mask)
    img_crop_filled = model_sd(
        prompt=text_prompt,
        image=Image.fromarray(img_crop),
        mask_image=Image.fromarray(mask_crop),
        num_inference_steps=50,
        guidance_scale=9.0,
    ).images[0]
    img_filled = crop_for_filling_post(img, mask, np.array(img_crop_filled))
    return img_filled


@torch.no_grad()
def inpaint_img_with_lama(
    img: np.ndarray,
    mask: np.ndarray,
    mod=8,
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.0)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch["image"] = img.permute(2, 0, 1).unsqueeze(0)
    batch["mask"] = mask[None, None]
    unpad_to_size = [batch["image"].shape[2], batch["image"].shape[3]]
    batch["image"] = pad_tensor_to_modulo(batch["image"], mod)
    batch["mask"] = pad_tensor_to_modulo(batch["mask"], mod)
    batch = move_to_device(batch, device_lama)
    batch["mask"] = (batch["mask"] > 0) * 1

    batch = model_lama(batch)
    cur_res = batch[model_lama_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
    return cur_res


def scale_image(image, size):
    # Get the original width and height
    width, height = image.size

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Calculate the new width and height
    if width > height:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def transcribe(audio_path: str):
    result = model_whisper.transcribe(audio_path)
    text = result["text"]
    return text


def define(text: str):
    actions = chatgpt_text2actions(text)
    return actions
    # actions = {'actions': [
    #     ['create', 'a large black table'],
    # ]}
    # return json.loads(text)


def remove(image: Image, params, x: int, y: int):
    img = np.array(image.convert("RGB"))
    masks, _, _ = predict_masks_with_sam(
        img,
        np.array(
            [
                [x, y],
            ]
        ).astype("float"),
        [
            1,
        ],
    )
    masks = masks.astype(np.uint8) * 255
    masks = [dilate_mask(mask, 15) for mask in masks]
    mask = masks[0]
    result = inpaint_img_with_lama(img, mask)
    return Image.fromarray(result)


def create(image: Image, params, x: int, y: int):
    img = np.array(image.convert("RGB"))
    masks, _, _ = predict_masks_with_sam(
        img,
        np.array(
            [
                [x, y],
            ]
        ).astype("float"),
        [
            1,
        ],
    )
    masks = masks.astype(np.uint8) * 255
    masks = [dilate_mask(mask, 50) for mask in masks]
    mask = masks[0]
    result = fill_img_with_sd(img, mask, params["object"])
    return Image.fromarray(result)


def chatgpt_text2actions(text):
    text = prompt_gpt_actions_extraction + text
    url = "https://tonai.tech/api/public/v1/services"
    headers = {"key": "..."}
    id_chatgpt = "..."
    messages = [{"role": "user", "content": text}]
    params = {
        "service_id": id_chatgpt,
        "messages": json.dumps(messages),
        "temperature": 0.8,
        "top_p": 0.2,
        "n": 1,
    }
    print("Params=", params)
    response = requests.post(url, headers=headers, json=params)
    if response.status_code == 200:
        text = response.text
    else:
        return "Error: " + str(response.status_code)

    try:
        actions = json.loads(text)
    except Exception as e:
        return "Error: " + str(e)

    # return actions
    return json.loads(
        json.loads(actions["messages"][-1]["text"])["choices"][0]["message"]["content"]
    )


if __name__ == "__main__":
    # text_for_chat = "replace two sofa on the right with a large white bed"
    text_for_chat = "i want to create a big black table with chairs around it"
    text = chatgpt_text2actions(text_for_chat)
    print(text)
