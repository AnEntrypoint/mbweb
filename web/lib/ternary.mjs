// Reads the ternary pose backbone exported by tools/ternary/export_web.py.
// weights.bin holds three tensor kinds back to back; manifest.json indexes them by name:
//   ternary: five {-1,0,1} codes per byte, base 3, least significant trit first; rows padded
//            to a multiple of five; one fp16 scale per row.
//   int8:    one signed byte per weight; one fp16 scale per row.
//   fp16:    little-endian half precision, dense.
// The decoded tensor is row-major Float32Array with the shape recorded in the manifest.

const TRIT_DIVISORS = [1, 3, 9, 27, 81];

// IEEE 754 half precision to single precision.
export function halfToFloat(bits) {
    const sign = bits & 0x8000 ? -1 : 1;
    const exp = (bits >> 10) & 0x1f;
    const frac = bits & 0x3ff;
    if (exp === 0) return sign * frac * 2 ** -24;
    if (exp === 0x1f) return frac ? NaN : sign * Infinity;
    return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

function readHalves(view, offset, count) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = halfToFloat(view.getUint16(offset + 2 * i, true));
    return out;
}

export function unpackTernaryRow(bytes, rowOffset, cols, out, outOffset) {
    const perRow = Math.ceil(cols / 5);
    for (let c = 0; c < cols; c++) {
        const byte = bytes[rowOffset + Math.floor(c / 5)];
        const trit = Math.floor(byte / TRIT_DIVISORS[c % 5]) % 3;
        out[outOffset + c] = trit - 1;
    }
    return perRow;
}

export class TernaryWeights {
    constructor(manifest, blob) {
        this.manifest = manifest;
        this.blob = blob;
        this.view = new DataView(blob);
        this.bytes = new Uint8Array(blob);
    }

    static async load(baseUrl) {
        const [manifestResponse, blobResponse] = await Promise.all([
            fetch(new URL('manifest.json', baseUrl)),
            fetch(new URL('weights.bin', baseUrl)),
        ]);
        if (!manifestResponse.ok) throw new Error(`manifest: HTTP ${manifestResponse.status}`);
        if (!blobResponse.ok) throw new Error(`weights: HTTP ${blobResponse.status}`);
        return new TernaryWeights(await manifestResponse.json(), await blobResponse.arrayBuffer());
    }

    names() {
        return Object.keys(this.manifest);
    }

    // Decodes one tensor to a row-major Float32Array with the shape from the manifest.
    tensor(name) {
        const entry = this.manifest[name];
        if (!entry) throw new Error(`no tensor ${name}`);
        if (entry.kind === 'fp16') {
            const count = entry.shape.reduce((a, b) => a * b, 1);
            return { shape: entry.shape, data: readHalves(this.view, entry.offset, count) };
        }
        const [rows, cols] = entry.shape;
        const scales = readHalves(this.view, entry.scale_offset, rows);
        const data = new Float32Array(rows * cols);
        if (entry.kind === 'int8') {
            for (let r = 0; r < rows; r++) {
                const base = entry.offset + r * cols;
                for (let c = 0; c < cols; c++) {
                    data[r * cols + c] = this.view.getInt8(base + c) * scales[r];
                }
            }
        } else if (entry.kind === 'ternary') {
            const rowBytes = Math.ceil(cols / 5);
            const row = new Float32Array(cols);
            for (let r = 0; r < rows; r++) {
                unpackTernaryRow(this.bytes, entry.offset + r * rowBytes, cols, row, 0);
                for (let c = 0; c < cols; c++) data[r * cols + c] = row[c] * scales[r];
            }
        } else {
            throw new Error(`unknown kind ${entry.kind} for ${name}`);
        }
        return { shape: entry.shape, data };
    }
}
