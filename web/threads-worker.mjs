// One thread of the threaded inference core: the inference worker posts the module, the shared memory and the
// thread id, and this worker runs the module's thread entry point on that memory.
import { runThread } from './lib/threads-host.mjs';

self.onmessage = ({ data }) => {
    if (data.ping) {
        self.postMessage({ ready: true });
        return;
    }
    const release = () => { if (data.busy) Atomics.store(data.busy, data.slot, 0); };
    try {
        runThread({ ...data, log: (fd, text) => console.log(text) });
        release();
    } catch (error) {
        release();
    }
};
