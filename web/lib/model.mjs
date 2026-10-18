// Fetches the pinned MotionBricks GGUF bundle at runtime and verifies every file (PRD-04).
// Runs in the page worker: bytes are checked with SubtleCrypto before the component sees them.
// Verified files are kept in Cache Storage under the pinned revision, split into chunks because
// Chromium rejects single Cache Storage entries of about 400 MB and up.

const CACHE_PREFIX = 'motionbricks-model-';
const CHUNK_BYTES = 64 * 1024 * 1024;

async function sha256Hex(bytes) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

async function verified(file, bytes) {
    const actual = await sha256Hex(bytes);
    if (actual !== file.sha256) {
        throw new Error(`checksum mismatch for ${file.path}: expected ${file.sha256}, got ${actual}`);
    }
    return bytes;
}

function chunkKey(manifest, file, index) {
    return `${manifest.base_url}${file.path}?chunk=${index}`;
}

function chunkCount(file) {
    return Math.ceil(file.bytes / CHUNK_BYTES);
}

async function readWithProgress(response, file, onProgress) {
    const total = Number(response.headers.get('content-length')) || file.bytes;
    const bytes = new Uint8Array(file.bytes);
    let offset = 0;
    const reader = response.body.getReader();
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (offset + value.byteLength > bytes.byteLength) {
            throw new Error(`${file.path} is larger than the pinned ${file.bytes} bytes`);
        }
        bytes.set(value, offset);
        offset += value.byteLength;
        onProgress(file.path, offset, total);
    }
    if (offset !== file.bytes) {
        throw new Error(`${file.path} is ${offset} bytes, expected ${file.bytes}`);
    }
    return bytes;
}

// Returns the file's bytes from its chunks, or null when any chunk is missing.
async function readCached(cache, manifest, file) {
    const bytes = new Uint8Array(file.bytes);
    for (let index = 0; index < chunkCount(file); index++) {
        const hit = await cache.match(chunkKey(manifest, file, index));
        if (!hit) return null;
        bytes.set(new Uint8Array(await hit.arrayBuffer()), index * CHUNK_BYTES);
    }
    return bytes;
}

async function storeCached(cache, manifest, file, bytes) {
    for (let index = 0; index < chunkCount(file); index++) {
        const start = index * CHUNK_BYTES;
        const end = Math.min(file.bytes, start + CHUNK_BYTES);
        await cache.put(chunkKey(manifest, file, index), new Response(bytes.subarray(start, end)));
    }
}

async function forgetCached(cache, manifest, file) {
    for (let index = 0; index < chunkCount(file); index++) {
        await cache.delete(chunkKey(manifest, file, index));
    }
}

// Joins a multi-part file from its parts in order, reporting progress against the whole file.
async function fetchParts(manifest, file, onProgress) {
    const bytes = new Uint8Array(file.bytes);
    let offset = 0;
    for (const part of file.parts) {
        const response = await fetch(manifest.base_url + part);
        if (!response.ok) {
            throw new Error(`cannot download ${part}: HTTP ${response.status}`);
        }
        const data = new Uint8Array(await response.arrayBuffer());
        if (offset + data.byteLength > bytes.byteLength) {
            throw new Error(`${file.path} is larger than the pinned ${file.bytes} bytes`);
        }
        bytes.set(data, offset);
        offset += data.byteLength;
        onProgress(file.path, offset, file.bytes);
    }
    if (offset !== file.bytes) {
        throw new Error(`${file.path} is ${offset} bytes, expected ${file.bytes}`);
    }
    return bytes;
}

// Returns { 'pose.gguf': Uint8Array, ... } for the files of the bundle directory.
export async function fetchModel(manifest, onProgress = () => {}) {
    const cache = await caches.open(CACHE_PREFIX + manifest.revision);
    const files = {};
    for (const file of manifest.files) {
        // Multi-part files are joined from their parts and are not kept in Cache Storage.
        let bytes = file.parts ? null : await readCached(cache, manifest, file);
        if (bytes) {
            try {
                bytes = await verified(file, bytes);
            } catch (error) {
                await forgetCached(cache, manifest, file);
                throw error;
            }
            onProgress(file.path, file.bytes, file.bytes, 'cache');
        } else if (file.parts) {
            bytes = await verified(file, await fetchParts(manifest, file, onProgress));
        } else {
            const response = await fetch(manifest.base_url + file.path);
            if (!response.ok) {
                throw new Error(`cannot download ${file.path}: HTTP ${response.status}`);
            }
            bytes = await verified(file, await readWithProgress(response, file, onProgress));
            try {
                await storeCached(cache, manifest, file, bytes);
            } catch {
                // Caching is best-effort (quota, private windows): the verified bytes are still used.
                await forgetCached(cache, manifest, file).catch(() => {});
            }
        }
        files[file.path.split('/').pop()] = bytes;
    }
    return files;
}
