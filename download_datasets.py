import time
import zipfile

import requests
import zlib
from pathlib import Path

from requests import HTTPError

CVAT_URL = 'localhost:9080'
ORG = 'AutoDidact'
FORMAT = 'CVAT for video 1.1'
AUTH_TOKEN = ''
CSRF_TOKEN = ''
SESSION_ID = ''
OUT = Path('out')

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

if __name__ == '__main__':
    for task_id in range(200):
        try:
            path = download_dataset(task_id)
        except HTTPError as err:
            continue
        print("Task", task_id, "in", str(path))
