"""Read-only geometry audit; run on the server without loading model weights."""
import json
from collections import Counter
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

root = Path('/root/autodl-tmp')
rows = [json.loads(x) for x in (root/'DVGBench/dvg_test.jsonl').read_text().splitlines()]
sizes = Counter()
changes = []
for r in rows:
    with Image.open(root/'DVGBench/images'/r['dataset']/r['image_id']) as im:
        w, h = im.size
    sizes[(w, h)] += 1
    rh, rw = smart_resize(h, w, factor=28, min_pixels=256*28*28, max_pixels=1280*1280)
    if (rw, rh) != (w, h):
        changes.append(dict(qid=r['question_id'], original=[w,h], processed=[rw,rh]))
p = AutoProcessor.from_pretrained(root/'Qwen2.5-VL-7B-Instruct', use_fast=False,
                                  min_pixels=256*28*28, max_pixels=1280*1280)
print(json.dumps(dict(n=len(rows), classes=dict(Counter(r['class'] for r in rows)),
    sizes={str(k):v for k,v in sizes.items()}, changed=len(changes), examples=changes[:10],
    processor=type(p.image_processor).__name__, min_pixels=p.image_processor.min_pixels,
    max_pixels=p.image_processor.max_pixels,
    template=p.apply_chat_template([{'role':'user','content':[{'type':'image'},
      {'type':'text','text':'Find <|object_ref_start|>test<|object_ref_end|>.'}]}],
      tokenize=False,add_generation_prompt=True)), ensure_ascii=False, indent=2))
