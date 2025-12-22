import * as esbuild from 'esbuild';

await esbuild.build({
    entryPoints: ['src/psd-utils.js'],
    bundle: true,
    outfile: '../web/ag-psd.bundle.js',
    format: 'iife',
    globalName: 'AgPsd',
    platform: 'browser',
    minify: true,
    sourcemap: false,
    target: ['es2020', 'chrome80', 'firefox80', 'safari14'],
    logLevel: 'info',
    plugins: [{
        name: 'ignore-util',
        setup(build) {
            // Replace require('util') with an empty object
            build.onResolve({ filter: /^util$/ }, () => {
                return { path: 'util', namespace: 'util-stub' };
            });
            build.onLoad({ filter: /.*/, namespace: 'util-stub' }, () => {
                return {
                    contents: 'export const inspect = () => ""; export default { inspect: () => "" };',
                    loader: 'js'
                };
            });
        }
    }]
});

console.log('Build completed successfully!');
