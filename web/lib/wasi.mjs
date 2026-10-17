// WASI 0.3.0 interfaces the transpiled component imports, keyed the way jco's async
// instantiation expects them. Shared by the Node smoke test and the browser page.
import * as cli from '@bytecodealliance/preview3-shim/cli';
import * as clocks from '@bytecodealliance/preview3-shim/clocks';
import * as filesystem from '@bytecodealliance/preview3-shim/filesystem';

export function wasiImports() {
    return {
        'wasi:cli/environment': cli.environment,
        'wasi:cli/exit': cli.exit,
        'wasi:cli/stderr': cli.stderr,
        'wasi:cli/stdin': cli.stdin,
        'wasi:cli/stdout': cli.stdout,
        'wasi:cli/terminal-input': cli.terminalInput,
        'wasi:cli/terminal-output': cli.terminalOutput,
        'wasi:cli/terminal-stderr': cli.terminalStderr,
        'wasi:cli/terminal-stdin': cli.terminalStdin,
        'wasi:cli/terminal-stdout': cli.terminalStdout,
        'wasi:clocks/monotonic-clock': clocks.monotonicClock,
        'wasi:clocks/system-clock': clocks.systemClock,
        'wasi:filesystem/preopens': filesystem.preopens,
        'wasi:filesystem/types': filesystem.types,
    };
}
