/**
 * SimplePSDStackNode Frontend PSD Generator
 * Uses ag-psd library to generate PSD files in the browser
 */

import { app } from "../../scripts/app.js";

// Track if ag-psd bundle is loaded
let agPsdLoaded = false;
let agPsdLoadPromise = null;

/**
 * Dynamically load the ag-psd bundle
 */
async function ensureAgPsdLoaded() {
    if (agPsdLoaded) return;

    if (agPsdLoadPromise) {
        return agPsdLoadPromise;
    }

    agPsdLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = './extensions/ComfyUI-fixableflow/ag-psd.bundle.js';
        script.onload = () => {
            agPsdLoaded = true;
            console.log("[SimplePSD] ag-psd bundle loaded successfully");
            resolve();
        };
        script.onerror = (e) => {
            console.error("[SimplePSD] Failed to load ag-psd bundle:", e);
            reject(new Error("Failed to load ag-psd bundle"));
        };
        document.head.appendChild(script);
    });

    return agPsdLoadPromise;
}

/**
 * Load an image from URL
 */
function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(new Error(`Failed to load image: ${url}`));
        img.src = url;
    });
}

/**
 * Create PSD file from layer information
 */
async function createPSD(layerInfo) {
    await ensureAgPsdLoaded();

    const { layers, width, height, prefix, timestamp } = layerInfo;

    console.log(`[SimplePSD] Creating PSD: ${width}x${height}, ${layers.length} layers`);

    // Create composite canvas
    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext('2d');

    // Process each layer
    const psdLayers = [];

    for (const layer of layers) {
        const url = `/view?filename=${encodeURIComponent(layer.filename)}&type=output&t=${Date.now()}`;
        console.log(`[SimplePSD] Loading layer: ${layer.name} from ${layer.filename}`);

        const img = await loadImage(url);

        // Create layer canvas
        const layerCanvas = document.createElement('canvas');
        layerCanvas.width = width;
        layerCanvas.height = height;
        const layerCtx = layerCanvas.getContext('2d');
        layerCtx.drawImage(img, 0, 0, width, height);

        // Also draw to composite
        compositeCtx.drawImage(img, 0, 0, width, height);

        psdLayers.push({
            name: layer.name,
            canvas: layerCanvas,
            left: 0,
            top: 0,
            right: width,
            bottom: height,
            blendMode: layer.blendMode || 'normal',
            opacity: layer.opacity !== undefined ? layer.opacity : 1
        });
    }

    // Create PSD structure
    const psd = {
        width,
        height,
        canvas: compositeCanvas,
        children: psdLayers
    };

    // Write PSD using ag-psd
    const psdBuffer = window.AgPsd.writePsd(psd);

    // Trigger download
    const blob = new Blob([psdBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${prefix}_${timestamp}.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SimplePSD] PSD downloaded: ${prefix}_${timestamp}.psd`);
}

app.registerExtension({
    name: "ComfyUI-fixableflow.SimplePSDStack",

    async nodeCreated(node) {
        // Only apply to SimplePSDStackNode nodes
        if (node.comfyClass === "SimplePSDStackNode") {
            console.log("[SimplePSD] Setting up frontend PSD generator");

            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "Generate & Download PSD",
                async () => {
                    console.log("[SimplePSD] Download button clicked");

                    try {
                        // Show loading state
                        downloadButton.name = "Generating PSD...";

                        // Fetch layer info from log file
                        const logResponse = await fetch('/view?filename=simple_psd_stack_info.log&type=output&t=' + Date.now());

                        if (!logResponse.ok) {
                            alert("Please run the workflow first to generate layers.\n\nワークフローを実行してレイヤーを生成してください。");
                            return;
                        }

                        const infoFilename = (await logResponse.text()).trim();
                        console.log("[SimplePSD] Layer info file:", infoFilename);

                        // Fetch layer info JSON
                        const infoResponse = await fetch(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`);

                        if (!infoResponse.ok) {
                            alert("Failed to load layer information.\n\nレイヤー情報の読み込みに失敗しました。");
                            return;
                        }

                        const layerInfo = await infoResponse.json();
                        console.log("[SimplePSD] Layer info:", layerInfo);

                        // Create PSD in browser
                        await createPSD(layerInfo);

                    } catch (error) {
                        console.error("[SimplePSD] Error:", error);
                        alert(`Failed to generate PSD: ${error.message}\n\nPSD生成に失敗しました: ${error.message}`);
                    } finally {
                        // Restore button text
                        downloadButton.name = "Generate & Download PSD";
                    }
                }
            );

            // Style the button
            downloadButton.color = "#10B981";
            downloadButton.bgcolor = "#059669";

            console.log("[SimplePSD] Frontend PSD generator ready");
        }
    }
});

console.log("[SimplePSD] Extension loaded");
