// JS host for the threaded motion-bricks core (wasm32-wasip1-threads). One shared memory is created here and
// imported by the main thread and every worker. wasi.thread-spawn asks the environment to start a worker with the
// same compiled module and memory; that worker runs the module's wasi_thread_start entry point.

// Minimal WASI preview 1 surface: stdio writes, a monotonic clock, and no filesystem. Everything the core
// does not reach is refused with a WASI error code, so an unexpected call fails loudly rather than silently.
const ERRNO_BADF = 8;
const ERRNO_NOENT = 44;

export function wasiImports({ memory, spawn, log }) {
    const view = () => new DataView(memory.buffer);
    const bytes = () => new Uint8Array(memory.buffer);
    const decoder = new TextDecoder();
    const wasi = {
        clock_time_get(_id, _precision, out) {
            view().setBigUint64(out, BigInt(Math.round(performance.now() * 1e6)), true);
            return 0;
        },
        environ_sizes_get(count, size) {
            view().setUint32(count, 0, true);
            view().setUint32(size, 0, true);
            return 0;
        },
        environ_get() { return 0; },
        fd_write(fd, iovs, iovsLength, written) {
            let total = 0;
            let text = '';
            for (let i = 0; i < iovsLength; i++) {
                const ptr = view().getUint32(iovs + i * 8, true);
                const len = view().getUint32(iovs + i * 8 + 4, true);
                text += decoder.decode(bytes().subarray(ptr, ptr + len));
                total += len;
            }
            if (text) log(fd, text);
            view().setUint32(written, total, true);
            return 0;
        },
        fd_close() { return 0; },
        fd_fdstat_get(_fd, out) {
            // Standard streams only: a character device with no rights beyond write.
            bytes().fill(0, out, out + 24);
            return 0;
        },
        fd_fdstat_set_flags() { return 0; },
        fd_prestat_get() { return ERRNO_BADF; },
        fd_prestat_dir_name() { return ERRNO_BADF; },
        fd_read() { return ERRNO_BADF; },
        fd_seek() { return ERRNO_BADF; },
        path_filestat_get() { return ERRNO_NOENT; },
        path_open() { return ERRNO_NOENT; },
        proc_exit(code) { throw new Error(`motion-bricks exited with code ${code}`); },
        sched_yield() { return 0; },
    };
    return {
        env: { memory },
        wasi_snapshot_preview1: wasi,
        wasi: { 'thread-spawn': (startArg) => spawn(startArg) },
    };
}

// Main thread: compile once, create the shared memory, and run the reactor initializer.
export async function loadThreadedCore(source, { launch, log = () => {}, initialPages = 1024 } = {}) {
    const module = source instanceof WebAssembly.Module ? source : await WebAssembly.compile(source);
    const memory = new WebAssembly.Memory({ initial: initialPages, maximum: 32768, shared: true });
    let nextThread = 1;
    const spawn = (startArg) => {
        const tid = nextThread++;
        launch({ module, memory, tid, startArg });
        return tid;
    };
    const instance = new WebAssembly.Instance(module, wasiImports({ memory, spawn, log }));
    instance.exports._initialize();
    return { exports: instance.exports, memory, module };
}

// Worker: run one thread of the module on the shared memory.
export function runThread({ module, memory, tid, startArg, log = () => {} }) {
    const spawn = () => -1; // worker threads do not spawn further threads
    const instance = new WebAssembly.Instance(module, wasiImports({ memory, spawn, log }));
    instance.exports.wasi_thread_start(tid, startArg);
}
