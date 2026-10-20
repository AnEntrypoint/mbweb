// Downloads the Kimodo G1 motion model and the text encoder named in animation/pin.env into .kimodo/weights.
// Each file is checked against its repository's MANIFEST.json (size and SHA-256) before it is used. Downloads
// resume from a .partial file, so an interrupted run continues where it stopped. Needs no hf CLI.
//
// Usage: node animation/fetch-weights.mjs
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const WEIGHTS = join(ROOT, '.kimodo/weights');
const pin = Object.fromEntries((await readFile(join(ROOT, 'animation/pin.env'), 'utf8'))
    .split('\n').filter((line) => /^[A-Z_]+=/.test(line)).map((line) => line.split(/=(.*)/s).slice(0, 2)));
const MOTION_REPO = pin.KIMODO_MOTION_REPO;
const MOTION_FILE = pin.KIMODO_MOTION_FILE;
const TEXT_REPO = 'LocalAI-io/Llama-3-Kimodo-GGML';

const manifestOf = async (repo) => {
    const response = await fetch(`https://huggingface.co/${repo}/resolve/main/MANIFEST.json`);
    if (!response.ok) throw new Error(`cannot read ${repo} manifest: HTTP ${response.status}`);
    return response.json();
};

const sha256File = (path) => new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    createReadStream(path).on('data', (chunk) => hash.update(chunk)).on('end', () => resolve(hash.digest('hex'))).on('error', reject);
});

async function fetchVerified(repo, entry) {
    const target = join(WEIGHTS, entry.path);
    const partial = `${target}.partial`;
    await mkdir(dirname(target), { recursive: true });
    if ((await stat(target).catch(() => null))?.size === entry.bytes && await sha256File(target) === entry.sha256) {
        console.log(`have   ${entry.path}`);
        return;
    }
    let have = (await stat(partial).catch(() => null))?.size ?? 0;
    if (have > entry.bytes) {
        await rm(partial);
        have = 0;
    }
    if (have < entry.bytes) {
        console.log(`get    ${entry.path} (${(entry.bytes / 1e9).toFixed(2)} GB, from ${have} B)`);
        const response = await fetch(`https://huggingface.co/${repo}/resolve/main/${entry.path}`, {
            headers: have ? { range: `bytes=${have}-` } : {},
        });
        if (!response.ok) throw new Error(`${entry.path}: HTTP ${response.status}`);
        // A server that ignores the range sends the whole file: start over rather than append it to the partial.
        const resumed = have > 0 && response.status === 206;
        const started = Date.now();
        const report = setInterval(async () => {
            const size = (await stat(partial).catch(() => null))?.size ?? 0;
            console.log(`       ${entry.path}: ${(size / 1e6).toFixed(0)} of ${(entry.bytes / 1e6).toFixed(0)} MB`);
        }, 60_000);
        try {
            await pipeline(Readable.fromWeb(response.body), createWriteStream(partial, { flags: resumed ? 'a' : 'w' }));
        } finally {
            clearInterval(report);
        }
        console.log(`       ${entry.path}: ${((Date.now() - started) / 60000).toFixed(1)} min`);
    }
    const size = (await stat(partial)).size;
    const digest = await sha256File(partial);
    if (size !== entry.bytes || digest !== entry.sha256) {
        await rm(partial);
        throw new Error(`${entry.path}: size or checksum does not match the manifest (partial removed; rerun to fetch again)`);
    }
    await rename(partial, target);
    console.log(`verified ${entry.path}`);
}

const motionManifest = await manifestOf(MOTION_REPO);
const motion = motionManifest.files.find((file) => file.path === MOTION_FILE);
if (!motion) throw new Error(`${MOTION_FILE} is not in ${MOTION_REPO}`);
await fetchVerified(MOTION_REPO, motion);

const textManifest = await manifestOf(TEXT_REPO);
const variant = textManifest.variants?.[pin.KIMODO_TEXT_QUANTIZATION];
if (!variant) throw new Error(`text quantization ${pin.KIMODO_TEXT_QUANTIZATION} is not in ${TEXT_REPO}`);
for (const path of [variant.weights, variant.tokenizer]) {
    const entry = textManifest.files.find((file) => file.path === path);
    if (!entry) throw new Error(`${path} is not in ${TEXT_REPO}`);
    await fetchVerified(TEXT_REPO, entry);
}
console.log('weights ready');
