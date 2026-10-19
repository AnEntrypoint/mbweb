// Browser smoke test for the built site (PRD-03, PRD-04, PRD-05 acceptance).
//
// Node 22 has no WebAssembly JS Promise Integration, which the transpiled component needs, so the
// test drives the built page in headless Chromium (JSPI is on by default from Chrome 137).
//
// Usage: npm run smoke   (needs site/ from scripts/build-site.sh and network access to Hugging Face)
// Env:   CHROMIUM_PATH   Chromium executable (default: Playwright's bundled browser)
//        RESULTS_JSON    if set, the measurements are written to this path
import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SITE = join(ROOT, 'site');
const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
};
const LOAD_TIMEOUT = 15 * 60 * 1000;

const failures = [];
const measurements = {};
function check(condition, message) {
    console.log(`${condition ? 'ok  ' : 'FAIL'} ${message}`);
    if (!condition) failures.push(message);
}

function serve() {
    const server = createServer(async (request, response) => {
        const path = normalize(new URL(request.url, 'http://localhost').pathname).replace(/^\/+/, '') || 'index.html';
        try {
            const body = await readFile(join(SITE, path));
            response.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
            response.end(body);
        } catch {
            response.writeHead(404);
            response.end();
        }
    });
    return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function waitForState(page, states) {
    await page.waitForFunction(
        (wanted) => wanted.includes(document.body.dataset.state),
        states,
        { timeout: LOAD_TIMEOUT },
    );
    return page.evaluate(() => document.body.dataset.state);
}

// Returns the canvas as a PNG data URL; equal strings mean identical pixels.
const canvasPixels = (page) => page.$eval('#stage', (canvas) => canvas.toDataURL());

async function scrubTo(page, frame) {
    await page.$eval('#scrub', (input, value) => {
        input.value = String(value);
        input.dispatchEvent(new Event('input'));
    }, frame);
}

async function pauseIfPlaying(page) {
    if ((await page.textContent('#play')).trim() === 'Pause') await page.click('#play');
}

const server = await serve();
const url = `http://127.0.0.1:${server.address().port}/`;
const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: ['--no-sandbox'],
});

try {
    const context = await browser.newContext();
    const page = await context.newPage();
    const consoleErrors = [];
    page.on('console', (message) => {
        if (message.type() === 'error') consoleErrors.push(message.text());
    });
    page.on('pageerror', (error) => consoleErrors.push(`pageerror: ${error.message}`));

    // Cold load: nothing in Cache Storage yet, so the page downloads and verifies the weights.
    let started = Date.now();
    await page.goto(url);
    const coldState = await waitForState(page, ['ready', 'error']);
    measurements.coldLoadSeconds = (Date.now() - started) / 1000;
    check(coldState === 'ready', `cold load reaches ready (${coldState}: ${await page.textContent('#status')})`);
    if (coldState !== 'ready') throw new Error('page did not become ready');

    const frames = Number(await page.getAttribute('body', 'data-frames'));
    check(frames >= 24 && frames <= 64 && frames % 4 === 0, `first clip has ${frames} frames (24..64, multiple of 4)`);
    measurements.frames = frames;

    // Live playback: queued clips take over one after another with no user action.
    await page.waitForFunction(() => Number(document.body.dataset.clip) >= 2, null, { timeout: 180 * 1000 });
    const secondFrames = Number(await page.getAttribute('body', 'data-frames'));
    // A chained clip plays its model output without the four source frames it repeats from the last clip.
    check(secondFrames >= 20 && secondFrames <= 60, `second clip has ${secondFrames} frames after its source frames`);
    await pauseIfPlaying(page);
    const playing = Number(await page.getAttribute('body', 'data-frames'));
    await scrubTo(page, 0);
    const clipStart = await canvasPixels(page);
    await scrubTo(page, playing - 1);
    check(clipStart !== (await canvasPixels(page)), 'live clip animates from its first frame to its last');

    // Warm load: a reload finds the verified weights in Cache Storage.
    started = Date.now();
    await page.reload();
    const warmState = await waitForState(page, ['ready', 'error']);
    measurements.warmLoadSeconds = (Date.now() - started) / 1000;
    check(warmState === 'ready', `warm load reaches ready (${warmState})`);

    check(consoleErrors.length === 0, `no console errors on a clean run (${consoleErrors.length})`);
    for (const message of consoleErrors) console.log(`     console: ${message}`);

    // PRD-04: a checksum mismatch aborts loading with a visible error.
    // Service workers are blocked here: their fetches bypass the route below, and the checksum check is shared by both load paths.
    const tampered = await browser.newContext({ serviceWorkers: 'block' });
    const tamperedPage = await tampered.newPage();
    await tampered.route('**/support.gguf', async (route) => {
        const response = await route.fetch();
        const body = Buffer.from(await response.body());
        body[body.length >> 1] ^= 0xff;
        await route.fulfill({ response, body });
    });
    await tamperedPage.goto(url);
    const tamperedState = await waitForState(tamperedPage, ['ready', 'error']);
    const tamperedStatus = await tamperedPage.textContent('#status');
    check(tamperedState === 'error' && /checksum mismatch/.test(tamperedStatus),
        `tampered support.gguf shows a checksum error (${tamperedState}: ${tamperedStatus})`);
    await tampered.close();

    await context.close();
} catch (error) {
    failures.push(error.message);
    console.log(`FAIL ${error.message}`);
} finally {
    await browser.close();
    server.close();
}

if (process.env.RESULTS_JSON) await writeFile(process.env.RESULTS_JSON, JSON.stringify(measurements, null, 2));
console.log(JSON.stringify(measurements));
console.log(`RESULT: ${failures.length === 0 ? 'PASS' : 'FAIL'}`);
if (failures.length > 0) process.exitCode = 1;
