const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const backendDir = path.join(rootDir, "backend");
const isWindows = process.platform === "win32";

function run(command) {
  const result = spawnSync(command, {
    stdio: "inherit",
    shell: true,
    cwd: backendDir,
  });

  if (typeof result.status === "number" && result.status !== 0) {
    process.exit(result.status);
  }
}

function commandExists(command) {
  const checkCmd = isWindows ? `where ${command}` : `command -v ${command}`;
  const result = spawnSync(checkCmd, {
    stdio: "ignore",
    shell: true,
    cwd: backendDir,
  });
  return result.status === 0;
}

if (commandExists("uv")) {
  run("uv run python run.py");
  process.exit(0);
}

const venvPython = isWindows
  ? path.join(backendDir, ".venv", "Scripts", "python.exe")
  : path.join(backendDir, ".venv", "bin", "python");

if (fs.existsSync(venvPython)) {
  run(`"${venvPython}" run.py`);
  process.exit(0);
}

console.error(
  "uv is not installed and backend/.venv is missing. Run `npm run setup:backend` first."
);
process.exit(1);
