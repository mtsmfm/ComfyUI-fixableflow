/**
 * RGBLineArtDividerFast Web Extension
 * Smart auto-download with detailed debugging
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("RGBLineArtDividerFast: Setting up smart download");
            
            // Store the node's generated filename and debug info
            let nodeFilename = null;
            let lastExecutionOutput = null;
            let captureHistory = [];
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download PSD (Run workflow first)",
                () => {
                    if (nodeFilename) {
                        console.log("Downloading:", nodeFilename);
                        
                        // Create download URL using the known path structure
                        const downloadUrl = `/view?filename=${encodeURIComponent(nodeFilename)}&type=output`;
                        console.log("Download URL:", downloadUrl);
                        
                        // Create and click download link
                        const link = document.createElement('a');
                        link.href = downloadUrl;
                        link.download = nodeFilename;
                        link.style.display = 'none';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        
                        console.log("Download completed for:", nodeFilename);
                    } else {
                        // Detailed error message with debug info
                        const debugInfo = [
                            "No PSD file available. Please run the workflow first.",
                            "",
                            "Debug Information:",
                            `- Current filename: ${nodeFilename}`,
                            `- Expected path: /output/${nodeFilename || 'output_rgb_fast_normal_XXXXXXXXXX.psd'}`,
                            `- Last execution output: ${JSON.stringify(lastExecutionOutput)}`,
                            "",
                            "Capture History:",
                            ...captureHistory.slice(-5).map(h => `- ${h}`),
                            "",
                            "Please check:",
                            "1. The workflow has been executed",
                            "2. The server console shows 'PSD file saved'",
                            "3. The file exists in ComfyUI/output/ directory"
                        ].join('\n');
                        
                        alert(debugInfo);
                        console.error("Download failed - Debug info:", {
                            nodeFilename,
                            lastExecutionOutput,
                            captureHistory
                        });
                    }
                }
            );
            
            // Initial style - disabled appearance
            downloadButton.color = "#888888";
            downloadButton.bgcolor = "#333333";
            
            // Monitor node execution
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                console.log("=== Node Executed ===");
                console.log("Output type:", typeof output);
                console.log("Output:", output);
                
                lastExecutionOutput = output;
                captureHistory.push(`onExecuted: ${JSON.stringify(output).substring(0, 100)}`);
                
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Try to extract filename from output
                let capturedFilename = null;
                
                if (output) {
                    // Check if it's an array (ComfyUI standard output format)
                    if (Array.isArray(output)) {
                        console.log("Output is array, length:", output.length);
                        
                        // Check for 4th element (psd_filename)
                        if (output.length > 3) {
                            console.log("4th element:", output[3]);
                            
                            if (output[3]) {
                                // Handle both string and array formats
                                if (typeof output[3] === 'string') {
                                    capturedFilename = output[3];
                                } else if (Array.isArray(output[3]) && output[3][0]) {
                                    capturedFilename = output[3][0];
                                }
                            }
                        }
                        
                        // Also check each element for PSD filename
                        output.forEach((item, index) => {
                            if (typeof item === 'string' && item.includes('.psd')) {
                                console.log(`Found PSD in output[${index}]:`, item);
                                capturedFilename = item;
                            }
                        });
                    }
                    // Check for object format
                    else if (typeof output === 'object') {
                        console.log("Output is object, keys:", Object.keys(output));
                        
                        // Check various possible locations
                        const possibleKeys = ['psd_filename', 'psd_path', 'filename', 'text', 'string'];
                        
                        for (const key of possibleKeys) {
                            if (output[key]) {
                                console.log(`Checking output.${key}:`, output[key]);
                                
                                const value = output[key];
                                if (typeof value === 'string' && value.includes('.psd')) {
                                    capturedFilename = value;
                                    break;
                                } else if (Array.isArray(value)) {
                                    const psdFile = value.find(v => typeof v === 'string' && v.includes('.psd'));
                                    if (psdFile) {
                                        capturedFilename = psdFile;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                
                if (capturedFilename && capturedFilename.includes('.psd')) {
                    nodeFilename = capturedFilename;
                    downloadButton.name = `⬇ Download: ${capturedFilename}`;
                    downloadButton.color = "#4CAF50";
                    downloadButton.bgcolor = "#2E7D32";
                    console.log("✅ Captured filename from output:", capturedFilename);
                    captureHistory.push(`✅ Captured: ${capturedFilename}`);
                } else {
                    console.log("❌ No filename in output, waiting for other sources");
                    captureHistory.push(`❌ No filename in output`);
                }
                
                console.log("=== End Node Execution ===");
            };
            
            // Store node reference for global access
            node.rgbDividerNode = true;
            node.updateFilename = function(filename, source = "unknown") {
                console.log(`Updating filename from ${source}:`, filename);
                
                if (filename && filename.includes('.psd')) {
                    nodeFilename = filename;
                    downloadButton.name = `⬇ Download: ${filename}`;
                    downloadButton.color = "#4CAF50";
                    downloadButton.bgcolor = "#2E7D32";
                    captureHistory.push(`✅ Updated from ${source}: ${filename}`);
                    console.log("✅ Updated filename:", filename);
                }
            };
            
            console.log("RGBLineArtDividerFast: Smart download ready");
        }
    }
});

// Monitor WebSocket messages for PSD filenames
if (api.socket) {
    const ws = api.socket;
    const originalOnMessage = ws.onmessage;
    
    ws.onmessage = function(event) {
        try {
            const msg = JSON.parse(event.data);
            
            // Log all executed messages for debugging
            if (msg.type === 'executed') {
                console.log("WebSocket executed message:", msg);
            }
            
            // Check for execution complete messages
            if (msg.type === 'executed' && msg.data && msg.data.output) {
                console.log("WebSocket output keys:", Object.keys(msg.data.output));
                
                // Look for psd_filename in the output
                if (msg.data.output.psd_filename) {
                    const filename = Array.isArray(msg.data.output.psd_filename) ? 
                        msg.data.output.psd_filename[0] : msg.data.output.psd_filename;
                    
                    if (filename && filename.includes('.psd')) {
                        console.log("✅ Found PSD filename in WebSocket:", filename);
                        
                        // Update all RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.rgbDividerNode && node.updateFilename) {
                                    node.updateFilename(filename, "WebSocket");
                                }
                            });
                        }
                    }
                }
                
                // Also check psd_path (in case the output name wasn't updated)
                if (msg.data.output.psd_path) {
                    const filename = Array.isArray(msg.data.output.psd_path) ? 
                        msg.data.output.psd_path[0] : msg.data.output.psd_path;
                    
                    if (filename && filename.includes('.psd')) {
                        console.log("✅ Found PSD path in WebSocket:", filename);
                        
                        // Update all RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.rgbDividerNode && node.updateFilename) {
                                    node.updateFilename(filename, "WebSocket-path");
                                }
                            });
                        }
                    }
                }
            }
            
            // Also check console messages for PSD file saved
            if (msg.type === 'console' && msg.data) {
                const text = typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data);
                if (text.includes('[RGBLineArtDividerFast]') && text.includes('PSD file saved:')) {
                    console.log("Console message with PSD:", text);
                    
                    // Extract filename from the log
                    const match = text.match(/output_rgb_fast_normal_[A-Za-z0-9]+\.psd/);
                    if (match) {
                        const filename = match[0];
                        console.log("✅ Found PSD filename in console:", filename);
                        
                        // Update all RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.rgbDividerNode && node.updateFilename) {
                                    node.updateFilename(filename, "Console");
                                }
                            });
                        }
                    }
                }
            }
        } catch (e) {
            console.error("WebSocket parse error:", e);
        }
        
        if (originalOnMessage) {
            originalOnMessage.apply(this, arguments);
        }
    };
}

// Monitor API events
api.addEventListener("executed", (event) => {
    console.log("API executed event:", event.detail);
    
    if (event.detail && event.detail.output) {
        console.log("API output keys:", Object.keys(event.detail.output));
        
        // Check for psd_filename in the output
        if (event.detail.output.psd_filename) {
            const filename = Array.isArray(event.detail.output.psd_filename) ? 
                event.detail.output.psd_filename[0] : event.detail.output.psd_filename;
            
            if (filename && filename.includes('.psd')) {
                console.log("✅ Found PSD filename in API event:", filename);
                
                // Update all RGBLineArtDividerFast nodes
                if (app.graph && app.graph.nodes) {
                    app.graph.nodes.forEach(node => {
                        if (node.rgbDividerNode && node.updateFilename) {
                            node.updateFilename(filename, "API");
                        }
                    });
                }
            }
        }
        
        // Also check psd_path
        if (event.detail.output.psd_path) {
            const filename = Array.isArray(event.detail.output.psd_path) ? 
                event.detail.output.psd_path[0] : event.detail.output.psd_path;
            
            if (filename && filename.includes('.psd')) {
                console.log("✅ Found PSD path in API event:", filename);
                
                // Update all RGBLineArtDividerFast nodes
                if (app.graph && app.graph.nodes) {
                    app.graph.nodes.forEach(node => {
                        if (node.rgbDividerNode && node.updateFilename) {
                            node.updateFilename(filename, "API-path");
                        }
                    });
                }
            }
        }
    }
});

// Override console.log to capture server messages
const originalConsoleLog = console.log;
console.log = function() {
    const message = Array.from(arguments).join(' ');
    
    // Check for PSD file saved message
    if (message.includes('[RGBLineArtDividerFast]') && message.includes('.psd')) {
        console.warn("Intercepted PSD message:", message);
        
        const match = message.match(/output_rgb_fast_normal_[A-Za-z0-9]+\.psd/);
        if (match) {
            const filename = match[0];
            
            // Update all RGBLineArtDividerFast nodes
            if (app.graph && app.graph.nodes) {
                app.graph.nodes.forEach(node => {
                    if (node.rgbDividerNode && node.updateFilename) {
                        node.updateFilename(filename, "Console.log");
                    }
                });
            }
        }
    }
    
    originalConsoleLog.apply(console, arguments);
};

console.log("RGBLineArtDividerFast extension loaded - detailed debugging enabled");
