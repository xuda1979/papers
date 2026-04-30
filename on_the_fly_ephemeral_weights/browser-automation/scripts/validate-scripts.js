const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const projectRoot = path.resolve(__dirname, '..');

function getAutomationScripts() {
  return fs
    .readdirSync(projectRoot)
    .filter((name) => name.endsWith('.js'))
    .sort();
}

function validateScript(scriptName) {
  const scriptPath = path.join(projectRoot, scriptName);
  execFileSync(process.execPath, ['--check', scriptPath], {
    cwd: projectRoot,
    stdio: 'pipe',
  });
  return scriptName;
}

function main() {
  const scripts = getAutomationScripts();

  if (scripts.length === 0) {
    throw new Error(`No automation scripts found in ${projectRoot}`);
  }

  const validated = scripts.map(validateScript);
  console.log(`Validated ${validated.length} automation scripts.`);
}

main();
