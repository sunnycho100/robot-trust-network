"""Find and download CC0 PBR texture sets from ambientCG (2K JPG) for the HQ look pass, into tex_cc0/ next to this file.
Usage (on the host): python fetch_tex.py   Picked last time: Metal009, Metal038, Metal027, Concrete048, Rubber004, Plastic013A.
"""
import io, json, urllib.request, zipfile, pathlib
out = pathlib.Path(__file__).resolve().parent / 'tex_cc0'
out.mkdir(parents=True, exist_ok=True)
UA = {'User-Agent': 'Mozilla/5.0 g1coop-port'}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


picked = {}
for role, q in [('sheet', 'galvanized'), ('sheet2', 'brushed metal'), ('scratch', 'scratched metal'), ('paint', 'painted metal'),
                ('concrete', 'concrete floor'), ('rubber', 'rubber'), ('plastic', 'plastic')]:
    try:
        j = json.loads(get('https://ambientcg.com/api/v2/full_json?type=Material&limit=5&include=downloadData&q=' + q.replace(' ', '+')))
    except Exception as e:
        print('[tex]', role, 'search failed', e)
        continue
    for a in j.get('foundAssets', []):
        links = [p for p in json.dumps(a).split('"') if p.startswith('https://') and '2K-JPG.zip' in p]
        if not links:
            continue
        aid = a['assetId']
        try:
            d = out / aid
            d.mkdir(exist_ok=True)
            zipfile.ZipFile(io.BytesIO(get(links[0]))).extractall(d)
            picked[role] = aid
            print('[tex]', role, 'got', aid, flush=True)
            break
        except Exception as e:
            print('[tex]', role, aid, 'download failed', e, flush=True)
json.dump(picked, open(out / 'picked.json', 'w'), indent=1)
print('[tex] picked', picked)
