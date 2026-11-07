import os
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from PIL import Image
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from swift.llm import (
get_model_tokenizer, get_template, safe_snapshot_download, BaseArguments, PtEngine, RequestConfig, InferRequest, load_image
)
from swift.utils import seed_everything
from swift.tuners import Swift



def eval_model(args):
    seed_everything(42)

    model = args.model_path
    engine = PtEngine(model, max_batch_size=1)
    request_config = RequestConfig(max_tokens=512, temperature=0)

    # lora_checkpoint = safe_snapshot_download(args.model_path)  # 修改成checkpoint_dir
    # new_args = BaseArguments.from_pretrained(lora_checkpoint)
    # model = new_args.model
    # template_type = new_args.template  # None: 使用对应模型默认的template_type
    # default_system = None  # None: 使用对应模型默认的default_system

    # # 加载模型和对话模板
    # model, tokenizer = get_model_tokenizer(model, model_type=new_args.model_type)
    # model = Swift.from_pretrained(model, lora_checkpoint)
    # template_type = template_type or model.model_meta.template
    # template = get_template(template_type, tokenizer, default_system=default_system)
    # engine = PtEngine.from_model_template(model, template, max_batch_size=1)
    # request_config = RequestConfig(max_tokens=512, temperature=0)


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    for i in tqdm(range(len(questions))):
        image_file=questions[i]['image_id']
        qs = questions[i]['question']


        # query = tokenizer.from_list_format([
        # {'image': os.path.join(args.image_folder, image_file)},
        # {'text': "Find <|object_ref_start|>" + qs + "<|object_ref_end|>"},
        # ])
        # query = f"""<img>{os.path.join(args.image_folder, image_file)}</img>Find <|object_ref_start|>{qs}<|object_ref_end|>"""
        image = load_image(os.path.join(args.image_folder, questions[i]['dataset'], image_file))
        infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>Find <|object_ref_start|>{qs}<|object_ref_end|> in the image. The output format should be like [x1, y1, x2, y2]."""}], images=[image])
        # infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>Find <|object_ref_start|>{qs}<|object_ref_end|> in the image."""}], images=[image])
        # infer_request = InferRequest(messages=[{'role': 'user', 'content': f"""<image>Find <ref>{qs}</ref>"""}], images=[image])

        
        resp_list = engine.infer([infer_request], request_config)
        response = resp_list[0].choices[0].message.content
        # print(f'Response: {response}')

        ans_file.write(json.dumps({
                                "question_id": questions[i]["question_id"],
                                "image_id": questions[i]["image_id"],
                                "class": questions[i]["class"],
                                "dataset": questions[i]["dataset"],
                                "answer": response,
                                "bbox": questions[i]['bbox'],
                                "poly": questions[i]['poly'],
                                "question":questions[i]['question'],
                                "dataset": questions[i]['dataset']                         
                                }) + "\n")
  
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    args = parser.parse_args()

    eval_model(args)