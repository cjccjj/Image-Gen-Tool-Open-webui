"""
title: Image generation with Aliyun Flux API
author: CJ Zhang
version: 0.0.1
license: MIT
git: https://github.com/cjccjj/Image-Gen-Tool-Open-webui/
description: Image generation using http get, images are not stored inside openwebui
"""

from pydantic import BaseModel, Field
import time
import requests
import random

FORMATS = {
    "default": (1024, 1024),
    "landscape": (1024, 576),
    "portrait": (576, 1024),
}


def send_image_generation_request(
    api_key: str,
    prompt: str,
    image_format: str,
    model: str,
    steps: int,
    seed: int,
) -> str:

    width, height = FORMATS[image_format]

    request = requests.post(
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
        headers={
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": {"prompt": prompt},
            "parameters": {
                "size": f"{width}*{height}",
                "steps": steps,
                "seed": seed,
            },
        },
    ).json()
    # print(request)
    return request["output"]["task_id"]


def poll_result(api_key: str, task_id: str) -> str:

    while True:
        result = requests.get(
            "https://dashscope.aliyuncs.com/api/v1/tasks/" + task_id,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        ).json()

        status = result["output"]["task_status"]

        if status not in ["PENDING", "RUNNING"]:
            break
        time.sleep(1)

    if status == "SUCCEEDED":
        return result["output"]["results"][0]["url"]
    else:
        raise RuntimeError(f"Image generation failed. Status: {status}")


class Tools:
    class Valves(BaseModel):
        api_key: str = Field("", description="Your Aliyun API key")

    def __init__(self):
        self.valves = self.Valves()

    async def create_flux_image(
        self,
        prompt: str,
        image_format: str,
        model: str,
        __event_emitter__=None,
    ) -> str:
        """
        Generate an images by prompts.

        :param prompt: the prompt to generate the image
        :param image_format: either 'default' for a square image, 'landscape' for a landscape format or 'portrait' for a portrait of mobile format
        :param model: model to use, 'flux-dev' by default or 'flux-schnell' for fast generation speed
        """
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Creating Image ...", "done": False},
            }
        )

        try:
            steps = 5 if model == "flux-schnell" else 25
            seed = random.randint(0, 65535)

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Creating Image ...", "done": False},
                }
            )
            start_time = time.time()

            task_id = send_image_generation_request(
                api_key=self.valves.api_key,
                prompt=prompt,
                image_format=image_format,
                model=model,
                steps=steps,
                seed=seed,
            )
            # print(f"flux image task: {task_id}")

            image_url = poll_result(api_key=self.valves.api_key, task_id=task_id)
            elapsed_time = time.time() - start_time
            message_str = f"![Image]({image_url})\nPrompt: `{prompt}`\nModel: `{model}`  Steps: `{steps}`  Seed: `{seed}`  Time Taken: `{elapsed_time:.2f} seconds`\n"
            # print(f"flux image url: {image_url}")

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Image Generated", "done": True},
                }
            )
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {
                        "content": message_str
                        # f"![Image]({image_url})\n Prompt: {prompt} \n Model: {model} Steps: {steps} Seed: {seed} Time taken:{elapsed_time:.2f} secs\n "
                    },
                }
            )
            return f"Notify the user that the image has been successfully generated, DO NOT mention the image or URL or anything else."

        except Exception as e:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": f"An error occured: {e}", "done": True},
                }
            )
            return f"Tell the user error: {e}"
