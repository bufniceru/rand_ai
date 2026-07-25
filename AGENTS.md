# Rand AI repository instructions

## Electron executable

- After making any code change in this repository, rebuild the portable Electron executable before reporting the task complete.
- Run `npm run electron:build` from `C:\code_py\rand_ai\web`.
- A renderer-only build such as `npm run build` or `npm run build:electron` is not sufficient.
- Verify that the packaging command succeeds and report the resulting executable's absolute path, size, and last-write time.
- If executable packaging fails, report the failure explicitly and do not describe the code-change task as fully complete.
