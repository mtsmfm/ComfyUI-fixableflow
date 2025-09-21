/**
 * RGBLineArtDividerFast Web Extension
 * Fallback download method - always download the latest PSD file
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("RGBLineArtDividerFast: Setting up download button");
            
            // Store the latest generated filename
            let latestPsdFilename = null;
            let executionCount = 0;
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download Latest PSD",
                async () => {
                    // First try to use stored filename
                    if (latestPsdFilename) {
                        console.log("Downloading stored file:", latestPsdFilename);
                        downloadFile(latestPsdFilename);
                        return;
                    }
                    
                    // Fallback: Try to get the latest file from the server
                    try {
                        console.log("Fetching latest PSD from server...");
                        
                        // Call API to get list of files in output directory
                        const response = await fetch('/api/output_files');
                        if (response.ok) {
                            const files = await response.json();
                            
                            // Filter for PSD files matching our pattern
                            const psdFiles = files.filter(f => 
                                f.includes('output_rgb_fast_normal_') && f.endsWith('.psd')
                            );
                            
                            if (psdFiles.length > 0) {
                                // Get the most recent file (assuming they're sorted or we take the last one)
                                const latestFile = psdFiles[psdFiles.length - 1];
                                console.log("Found latest PSD:", latestFile);
                                downloadFile(latestFile);
                                latestPsdFilename = latestFile;
                                updateButton(latestFile);
                            } else {
                                alert("No PSD files found. Please run the workflow first.");
                            }
                        }
                    } catch (error) {
                        console.error("Error fetching files:", error);
                        
                        // Ultimate fallback: Prompt user
                        const filename = prompt(
                            "Could not auto-detect PSD file.\n" +
                            "Please enter the filename from the server console:\n" +
                            "(e.g., output_rgb_fast_normal_G3HTgU8UPn.psd)"
                        );
                        
                        if (filename && filename.includes('.psd')) {
                            downloadFile(filename);
                            latestPsdFilename = filename;
                            updateButton(filename);
                        }
                    }
                }
            );
            
            // Helper function to download file
            function downloadFile(filename) {
                // Ensure we have just the filename, not the full path
                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                
                const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                console.log("Downloading from:", downloadUrl);
                
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = cleanFilename;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                console.log("Download initiated for:", cleanFilename);
            }
            
            // Helper function to update button
            function updateButton(filename) {
                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                downloadButton.name = `⬇ Download: ${cleanFilename}`;
                downloadButton.color = "#4CAF50";
                downloadButton.bgcolor = "#2E7D32";
            }
            
            // Initial button style
            downloadButton.color = "#3B82F6";
            downloadButton.bgcolor = "#1E40AF";
            
            // Monitor console logs for PSD filename
            const originalLog = console.log;
            console.log = function() {
                const message = Array.from(arguments).join(' ');
                
                // Check for our PSD save message
                if (message.includes('[RGBLineArtDividerFast]') && message.includes('PSD file saved:')) {
                    // Extract filename using a more flexible regex
                    const match = message.match(/output_rgb_fast_normal_[A-Za-z0-9]+\.psd/);
                    if (match) {
                        const filename = match[0];
                        console.warn("📁 Captured PSD filename:", filename);
                        latestPsdFilename = filename;
                        updateButton(filename);
                        executionCount++;
                    }
                }
                
                // Call original console.log
                originalLog.apply(console, arguments);
            };
            
            // Also try to monitor execution if it works
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                console.log("Node executed, output type:", typeof output);
                
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Increment counter to track executions
                executionCount++;
                downloadButton.name = `⬇ Download Latest PSD (Run #${executionCount})`;
            };
            
            console.log("RGBLineArtDividerFast: Download button ready");
        }
    }
});

// Try to intercept WebSocket messages as backup
if (api.socket) {
    const ws = api.socket;
    const originalOnMessage = ws.onmessage;
    
    ws.onmessage = function(event) {
        try {
            const msg = JSON.parse(event.data);
            
            // Check for console output containing PSD filename
            if (msg.type === 'status' && msg.data && msg.data.message) {
                const message = msg.data.message;
                if (message.includes('output_rgb_fast_normal_') && message.includes('.psd')) {
                    const match = message.match(/output_rgb_fast_normal_[A-Za-z0-9]+\.psd/);
                    if (match) {
                        console.log("📁 Found PSD in WebSocket status:", match[0]);
                        
                        // Update any RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.comfyClass === 'RGBLineArtDividerFast') {
                                    // Trigger update if possible
                                    const widgets = node.widgets || [];
                                    const downloadWidget = widgets.find(w => w.type === 'button' && w.name.includes('Download'));
                                    if (downloadWidget) {
                                        downloadWidget.name = `⬇ Download: ${match[0]}`;
                                        downloadWidget.color = "#4CAF50";
                                        downloadWidget.bgcolor = "#2E7D32";
                                    }
                                }
                            });
                        }
                    }
                }
            }
        } catch (e) {
            // Ignore parse errors
        }
        
        if (originalOnMessage) {
            originalOnMessage.apply(this, arguments);
        }
    };
}

console.log("RGBLineArtDividerFast extension loaded - fallback mode");
