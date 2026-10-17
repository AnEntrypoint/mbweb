// Instantiates the transpiled motion-bricks component and stages the verified GGUF bytes into it.
import { instantiate } from '../../build/js/motionbricks.js';
import { wasiImports } from './wasi.mjs';

// Bytes cross into the component in chunks of this size (PRD-04: stream into component memory).
const CHUNK_BYTES = 4 * 1024 * 1024;

// componentBase is the URL of the directory that holds the transpiled core modules.
export async function instantiateComponent(componentBase) {
    const compileCore = async (name) => {
        const response = await fetch(new URL(name, componentBase));
        if (!response.ok) throw new Error(`cannot fetch ${name}: HTTP ${response.status}`);
        return WebAssembly.compile(await response.arrayBuffer());
    };
    const component = await instantiate(compileCore, wasiImports());
    return component.inference;
}

// Streams each verified file into the component, releasing the page's copy as it goes.
// Only call this after every file has passed its checksum.
export async function stageBundle(inference, files) {
    for (const name of Object.keys(files)) {
        const bytes = files[name];
        await inference.beginFile(name, BigInt(bytes.byteLength));
        for (let offset = 0; offset < bytes.byteLength; offset += CHUNK_BYTES) {
            const end = Math.min(bytes.byteLength, offset + CHUNK_BYTES);
            await inference.writeChunk(name, BigInt(offset), bytes.subarray(offset, end));
        }
        delete files[name];
    }
}
