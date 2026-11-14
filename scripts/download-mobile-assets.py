"""Download pinned assets with size/SHA-256 checks; never execute downloaded code."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parents[1]

def download(item):
    dest = ROOT / item['destination']
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get('sha256')
    def valid():
        if not dest.is_file():
            return False
        if item.get('size') and dest.stat().st_size != item['size']:
            return False
        if expected:
            with dest.open('rb') as f:
                return hashlib.file_digest(f, 'sha256').hexdigest() == expected
        return True
    if valid():
        print(f'VALID {dest.name}', flush=True)
        return item
    partial = dest.with_name(dest.name + '.partial')
    if dest.exists() and item.get('size') and dest.stat().st_size < item['size']:
        if not partial.exists() or partial.stat().st_size < dest.stat().st_size:
            dest.replace(partial)
    for attempt in range(12):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'User-Agent': 'VOSR-Mobile/0.1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(item['url'], headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned a different byte range')
                if not append:
                    offset = 0
                with partial.open('ab' if append else 'wb') as out:
                    last = time.monotonic()
                    while data := response.read(1024 * 1024):
                        out.write(data)
                        offset += len(data)
                        if time.monotonic() - last >= 20:
                            print(f'DOWNLOAD {dest.name}: {offset / 1048576:.1f} MiB', flush=True)
                            last = time.monotonic()
            if item.get('size') and partial.stat().st_size != item['size']:
                raise ValueError(f'Incomplete download: {partial.stat().st_size}/{item["size"]}; preserving partial')
            if expected:
                with partial.open('rb') as f:
                    if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
                        partial.replace(partial.with_suffix('.corrupt'))
                        raise ValueError('SHA-256 mismatch; preserved as .corrupt')
            partial.replace(dest)
            if not valid():
                raise ValueError(f'Size or checksum mismatch: {dest}')
            print(f'DONE {dest.name}: {dest.stat().st_size} bytes', flush=True)
            return item
        except Exception as error:
            print(f'RETRY {attempt+1} {dest.name}: {error}', flush=True)
            if attempt == 11:
                raise
            time.sleep(min(2 ** attempt, 20))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item): item for item in manifest}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append({'destination': futures[future]['destination'], 'error': str(error)})
    result = {'checked_at_unix': time.time(), 'total': len(manifest), 'failures': failures}
    args.manifest.with_suffix('.result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    raise SystemExit(1 if failures else 0)
