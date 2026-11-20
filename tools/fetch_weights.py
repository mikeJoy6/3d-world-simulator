"""Download the released inference bundles and verify each pinned file before use."""
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = 'https://huggingface.co/datasets/jwang771/3d-world-simulator/resolve/main/'

def matches(path, row):
    if not path.exists() or path.stat().st_size != row['size']:
        return False
    digest = hashlib.sha256() if row['sha256'] else hashlib.sha1()
    if not row['sha256']:
        digest.update(f"blob {row['size']}\0".encode())
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == (row['sha256'] or row['git'])

def fetch(row):
    dest = ROOT / 'assets' / row['path']
    if matches(dest, row):
        return
    dest.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
        tmp = pathlib.Path(f.name)
    try:
        subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','15','--max-time','300', BASE+row['path'],'-o',str(tmp)],check=True)
        if not matches(tmp,row):
            raise ValueError(f"Checksum mismatch: {dest.name}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)

if __name__ == '__main__':
    rows=json.loads((ROOT/'tools/checkpoints.json').read_text())
    rows.sort(key=lambda row: ('/pusht/' not in row['path'], -row['size']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(fetch, rows))
    print(f"Verified {len(rows)} files, {sum(r['size'] for r in rows):,} bytes.")
