const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const rootDir = path.resolve(__dirname, "..");
const backendDir = path.join(rootDir, "backend");
const isWindows = process.platform === "win32";

function run(command, options = {}) {
  const result = spawnSync(command, {
    stdio: "inherit",
    shell: true,
    cwd: backendDir,
    ...options,
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

function parsePythonVersion(rawOutput) {
  const match = rawOutput.match(/Python\s+(\d+)\.(\d+)\.(\d+)/i);
  if (!match) {
    return null;
  }

  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    text: `${match[1]}.${match[2]}.${match[3]}`,
  };
}

function getPythonVersion(command) {
  const result = spawnSync(`${command} --version`, {
    stdio: "pipe",
    shell: true,
    cwd: backendDir,
    encoding: "utf8",
  });

  if (result.status !== 0) {
    return null;
  }

  const output = `${result.stdout || ""}${result.stderr || ""}`.trim();
  return parsePythonVersion(output);
}

function isSupportedPython(version) {
  return version && version.major === 3 && version.minor === 11;
}

function resolvePythonCommand() {
  const candidates = isWindows
    ? ["py -3.12", "py -3.11", "python", "py -3", "py"]
    : ["python3.12", "python3.11", "python3", "python"];
  const discovered = [];

  for (const candidate of candidates) {
    const version = getPythonVersion(candidate);
    if (version) {
      discovered.push(`${candidate} (${version.text})`);
    }
    if (isSupportedPython(version)) {
      return candidate;
    }
  }

  if (discovered.length === 0) {
    console.error("Python is required but was not found in PATH.");
  } else {
    console.error(
      `Found Python, but none are compatible. MiroFish backend currently requires Python 3.11.x. Found: ${discovered.join(", ")}`
    );
  }
  process.exit(1);
}

function ensureCompatibleVenv(venvPython, pythonCmd) {
  if (!fs.existsSync(venvPython)) {
    run(`${pythonCmd} -m venv .venv`);
    return;
  }

  const venvVersion = getPythonVersion(`"${venvPython}"`);
  if (isSupportedPython(venvVersion)) {
    return;
  }

  console.log("Existing .venv uses an incompatible Python version. Recreating .venv.");
  fs.rmSync(path.join(backendDir, ".venv"), { recursive: true, force: true });
  run(`${pythonCmd} -m venv .venv`);
}

if (commandExists("uv")) {
  run("uv sync");
  process.exit(0);
}

console.log("uv was not found. Falling back to a local .venv + pip install.");
const pythonCmd = resolvePythonCommand();
const venvDir = path.join(backendDir, ".venv");
const venvPython = isWindows
  ? path.join(venvDir, "Scripts", "python.exe")
  : path.join(venvDir, "bin", "python");

ensureCompatibleVenv(venvPython, pythonCmd);

run(`"${venvPython}" -m pip install --upgrade pip`);
run(`"${venvPython}" -m pip install -r requirements.txt`);
