// Scans the staged diff for credential-shaped strings. Prints RESULT: PASS or RESULT: FAIL and exits non-zero on a hit.
import { execFileSync } from 'node:child_process';

const PATTERNS = [
    ['private key block', /-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----/],
    ['GitHub token', /\bgh[pousr]_[A-Za-z0-9]{36,}\b/],
    ['AWS access key id', /\bAKIA[0-9A-Z]{16}\b/],
    ['Slack token', /\bxox[abprs]-[A-Za-z0-9-]{10,}\b/],
    ['generic secret assignment', /\b(?:api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['"][^'"\s]{12,}['"]/i],
];

const diff = execFileSync('git', ['diff', '--cached', '--no-color', '-U0'], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
const hits = [];
let file = '';
for (const line of diff.split('\n')) {
    if (line.startsWith('+++ ')) file = line.slice(4).replace(/^b\//, '');
    if (!line.startsWith('+') || line.startsWith('+++')) continue;
    for (const [label, pattern] of PATTERNS) {
        if (pattern.test(line)) hits.push(`${file}: ${label}`);
    }
}
for (const hit of hits) console.log(`hit ${hit}`);
console.log(hits.length ? 'RESULT: FAIL' : 'RESULT: PASS');
process.exit(hits.length ? 1 : 0);
