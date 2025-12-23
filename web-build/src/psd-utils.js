import { writePsd } from 'ag-psd';

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = (e) => reject(new Error(`Failed to load image: ${url}`));
        img.src = url;
    });
}

async function createPSDFromLayers(layerInfo) {
    const { layers, width, height } = layerInfo;

    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext('2d');

    const psdLayers = [];

    for (const layer of layers) {
        const url = `/view?filename=${encodeURIComponent(layer.filename)}&type=output&t=${Date.now()}`;
        const img = await loadImage(url);

        const layerCanvas = document.createElement('canvas');
        layerCanvas.width = width;
        layerCanvas.height = height;
        const layerCtx = layerCanvas.getContext('2d');
        layerCtx.drawImage(img, 0, 0, width, height);

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

    const psd = {
        width,
        height,
        canvas: compositeCanvas,
        children: psdLayers
    };

    return writePsd(psd);
}

async function createPSDFromUrls(imageUrls, width, height) {
    const compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext('2d');

    const psdLayers = [];

    for (const item of imageUrls) {
        const img = await loadImage(item.url);

        const layerCanvas = document.createElement('canvas');
        layerCanvas.width = width;
        layerCanvas.height = height;
        const layerCtx = layerCanvas.getContext('2d');
        layerCtx.drawImage(img, 0, 0, width, height);
        compositeCtx.drawImage(img, 0, 0, width, height);

        psdLayers.push({
            name: item.name || `Layer ${psdLayers.length + 1}`,
            canvas: layerCanvas,
            left: 0,
            top: 0,
            right: width,
            bottom: height,
            blendMode: item.blendMode || 'normal',
            opacity: item.opacity !== undefined ? item.opacity : 1
        });
    }

    const psd = {
        width,
        height,
        canvas: compositeCanvas,
        children: psdLayers
    };

    return writePsd(psd);
}

function downloadPSD(psdBuffer, filename) {
    const blob = new Blob([psdBuffer], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

window.AgPsd = {
    writePsd,
    loadImage,
    createPSDFromLayers,
    createPSDFromUrls,
    downloadPSD
};

export { writePsd, loadImage, createPSDFromLayers, createPSDFromUrls, downloadPSD };
