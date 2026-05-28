module.exports = {
  apps: [{
    name: 'floor-route-api',
    script: 'src/index.ts',
    interpreter: 'node',
    interpreter_args: '--env-file=.env --import=tsx',
    cwd: '/opt/floor-route-worker',
  }]
}
