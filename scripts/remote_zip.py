"""Inspect/extract selected official SDK ZIP entries using HTTP byte ranges."""
import argparse
import concurrent.futures
from collections import OrderedDict
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

class RangeReader(io.RawIOBase):
    def __init__(self, url, cache_dir=None):
        self.url = url
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.length = int(response.headers['Content-Range'].split('/')[-1])
        self.position = 0
        self.cache = OrderedDict()
        self.block = 2 * 1024 * 1024
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, index):
        start = index * self.block
        end = min(start + self.block, self.length) - 1
        path = self.cache_dir / f'{index:06}.bin' if self.cache_dir else None
        if path and path.exists() and path.stat().st_size == end-start+1:
            return path.read_bytes()
        for attempt in range(6):
            try:
                request = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                        raise RuntimeError('Server did not honor byte range')
                    data = response.read()
                    if len(data) != end-start+1:
                        raise RuntimeError('Truncated range')
                if path:
                    path.write_bytes(data)
                return data
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 10))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def read(self, count=-1):
        if count < 0:
            count = self.length - self.position
        count = min(count, self.length - self.position)
        result = bytearray()
        while count > 0:
            index = self.position // self.block
            if index not in self.cache:
                self.cache[index] = self.fetch(index)
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
            data = self.cache[index]
            offset = self.position % self.block
            part = data[offset:offset + count]
            result.extend(part)
            self.position += len(part)
            count -= len(part)
        return bytes(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    parser.add_argument('--listing', required=True, type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--extract', type=Path)
    parser.add_argument('--patterns', type=Path)
    args = parser.parse_args()
    reader = RangeReader(args.url, args.cache_dir)
    with zipfile.ZipFile(reader) as archive:
        listing = [{'name': i.filename, 'size': i.file_size, 'compressed': i.compress_size} for i in archive.infolist()]
        args.listing.write_text(json.dumps(listing, indent=2), encoding='utf-8')
        print(f'{len(listing)} entries; uncompressed {sum(i["size"] for i in listing)/1048576:.1f} MiB')
        if args.extract:
            patterns = json.loads(args.patterns.read_text(encoding='utf-8-sig'))
            entries = [i for i in archive.infolist() if any(re.search(p, i.filename) for p in patterns) and not i.is_dir()]
            needed = set()
            for entry in entries:
                start = entry.header_offset // reader.block
                end = min(reader.length-1, entry.header_offset+entry.compress_size+len(entry.filename.encode('utf-8'))+len(entry.extra)+1024)//reader.block
                needed.update(range(start, end+1))
            print(f'Extracting {len(entries)} files, requesting up to {len(needed)*2} MiB in byte ranges', flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                futures = [pool.submit(reader.fetch, index) for index in sorted(needed)]
                for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    future.result()
                    if n % 10 == 0:
                        print(f'Fetched {n}/{len(needed)} blocks', flush=True)
            root = args.extract.resolve()
            for entry in entries:
                destination = (root / entry.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError('Unsafe zip path')
                archive.extract(entry, root)
            print('Selected SDK extraction complete (ZIP CRC checked)', flush=True)
