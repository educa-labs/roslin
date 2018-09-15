import json 
import requests

def download(url,output_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in r:
                f.write(chunk)

if __name__ == '__main__':
    print("Dowloading")
    with open('../data.json') as f:
        projects = json.load(f)
    for project in projects:
        url = project["photo_url"]
        pid = project["id"]
        download(url,"images/{}.jpg".format(pid))