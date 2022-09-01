import time
import zipfile
from typing import Tuple

import numpy as np
import pandas
import requests
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from pathlib import Path

from pandas import DataFrame
from requests import HTTPError
from tqdm import trange

CVAT_URL = 'localhost:9080'
ORG = 'AutoDidact'
FORMAT = 'CVAT for video 1.1'
AUTH_TOKEN = ''
CSRF_TOKEN = ''
SESSION_ID = ''
OUT = Path('out')
LABELS = 'Bed', 'Staff', 'Devices', 'Patient'

def download_dataset(task_id: int) -> Path:
    out = OUT / f"{task_id}"
    zip_path = out.with_name(f"{task_id}.zip")
    zip_path.unlink(missing_ok=True)
    if out.exists():
        return out
    url = f"http://{CVAT_URL}/api/tasks/{task_id}/dataset?org={ORG}&format={FORMAT}"
    headers = {
        "Host": "localhost:9080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": url,
        "Authorization": f"Token {AUTH_TOKEN}",
        "X-CSRFTOKEN": CSRF_TOKEN,
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": f"csrftoken={CSRF_TOKEN}; sessionid={SESSION_ID}",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    requests.get(url, headers=headers).raise_for_status()
    print("Requesting dataset export:")
    for _ in range(19):
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        if resp.status_code == 202:
            # Accepted
            print('.', end='')
        elif resp.status_code  == 201:
            # Created
            print()
            break
        time.sleep(1)
    else:
        raise Exception("Failed to get the dataset within 20s")

    url2 = f"{url}&action=download"
    headers2 = {
        "Host": "localhost:9080",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:105.0) Gecko/20100101 Firefox/105.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": url,
        "DNT": "1",
        "Connection": "keep-alive",
        "Cookie": f"csrftoken={CSRF_TOKEN}; sessionid={SESSION_ID}",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    }
    print("Downloading dataset:")
    resp2 = requests.get(url2, headers=headers2)
    resp2.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(zip_path, 'wb') as fp:
        fp.write(resp2.content)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(out)
    zip_path.unlink()
    return out

def read_data_subset(subset_folder: Path) -> dict:
    assert subset_folder.is_dir()
    xml = BeautifulSoup(open(subset_folder / 'annotations.xml', 'r'), "xml")
    labels = {}
    for label in xml.select('annotations meta task labels label'):
        labels[label.select_one('name').text] = label.select_one('color').text
    assert labels.keys() == set(LABELS)
    data = {
        'version': xml.select_one('annotations version').text,
        'labels': labels,
        'source': xml.select_one('annotations meta source').text,
        'size': int(xml.select_one('annotations meta task size').text),
    }
    index = []
    gt = []
    for track in xml.select('annotations track'):
        assert track['label'] in LABELS
        for box in track.select('box'):
            index.append(int(box['frame']))
            gt.append({
                'label': LABELS.index(track['label']),
                'xtl': float(box['xtl']),
                'ytl': float(box['ytl']),
                'xbr': float(box['xbr']),
                'ybr': float(box['ybr']),
                'rotation': float(box.get('rotation', "0.0")),
            })
    data['gt'] = DataFrame(gt, index=index).sort_index()
    assert len(list((subset_folder / 'images').rglob('*.PNG'))) == int(data['size'])
    return data

def draw_bbox(draw: ImageDraw.ImageDraw, xtl: float, ytl: float, xbr: float, ybr: float, color: str, size: Tuple[int, int], label: str = None):
    bbox = [(xtl, ytl), (xbr, ytl), (xbr, ybr), (xtl, ybr), (xtl, ytl)]
    draw.line(bbox, fill=color, width=3)
    if label is not None:
        x0 = min(xtl, xbr)
        y0 = max(ytl, ybr)
        _, _, x1, y1 = ImageFont.load_default().getbbox(label)
        x1 += x0 + 4
        y1 += y0 + 4
        if x1 > size[0]:
            diff = x1 - size[0]
            x0 -= diff
        if y1 > size[1]:
            diff = y1 - size[1]
            y0 -= diff
        draw.rectangle((x0, y0, x1, y1), fill='#303030')
        draw.text((x0 + 2, y0 + 2), label, color)

def visualize_sample(idx: int, data: dict, folder: Path, save_to_disk: bool = False) -> Image.Image:
    out = folder / 'visualized'
    out.mkdir(exist_ok=True)
    img_path = folder / 'images' / f"frame_{idx:06}.PNG"
    assert img_path.exists()
    vis_path = out / img_path.name
    img = Image.open(img_path)
    # img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    ann = data['gt'].loc[[idx]]
    for frame, (label, xtl, ytl, xbr, ybr, rot) in ann.iterrows():
        cls = LABELS[int(label)]
        col = data['labels'][cls]
        draw_bbox(draw, xtl, ytl, xbr, ybr, col, img.size, cls)
    if save_to_disk:
        img.save(vis_path)
    return img

def visualize_dataset(data: dict, folder: Path, save_to_disk: bool = True):
    print('Visualizing dataset')
    for idx in trange(data['size']):
        visualize_sample(idx, data, folder, save_to_disk=save_to_disk)

if __name__ == '__main__':
    for task_id in range(200):
        try:
            path = download_dataset(task_id)
        except HTTPError as err:
            continue
        print("Task", task_id, "in", str(path))
        data = read_data_subset(path)
        visualize_dataset(data, path)
