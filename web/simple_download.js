/**
 * RGBLineArtDividerFast Web Extension
 * Always fetch latest from log file - no caching
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("[RGBDivider] Setting up download button");
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download PSD",
                async () => {
                    console.log("[RGBDivider] Download button clicked");
                    
                    // Always fetch the latest filename from log file
                    try {
                        const logFilename = await fetchLatestPsdFromLog();
                        if (logFilename) {
                            console.log("[RGBDivider] Found file in log:", logFilename);
                            downloadPsd(logFilename);
                        } else {
                            console.log("[RGBDivider] No file found in log, prompting user");
                            promptForManualInput();
                        }
                    } catch (error) {
                        console.error("[RGBDivider] Error reading log:", error);
                        promptForManualInput();
                    }
                }
            );
            
            // Function to fetch latest PSD filename from log (always gets fresh data)
            async function fetchLatestPsdFromLog() {
                try {
                    // Always fetch with cache-busting timestamp
                    const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
                    
                    if (response.ok) {
                        const text = await response.text();
                        const filename = text.trim();
                        
                        if (filename && filename.includes('.psd')) {
                            console.log("[RGBDivider] Log file contains:", filename);
                            return filename;
                        } else {
                            console.log("[RGBDivider] Log file exists but no valid PSD filename found");
                        }
                    } else {
                        console.log("[RGBDivider] Log file not found (404)");
                    }
                } catch (error) {
                    console.error("[RGBDivider] Failed to read log file:", error);
                }
                return null;
            }
            
            // Function to download PSD
            function downloadPsd(filename) {
                console.log("[RGBDivider] Starting download for:", filename);
                
                // Ensure we have just the filename, not the full path
                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                
                const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                console.log("[RGBDivider] Download URL:", downloadUrl);
                
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = cleanFilename;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                console.log("[RGBDivider] Download initiated successfully");
            }
            
            // Function for manual input
            function promptForManualInput() {
                const filename = prompt(
                    "ログファイルが見つかりません。\n" +
                    "ワークフローを実行するか、手動でファイル名を入力してください。\n" +
                    "例: output_rgb_fast_normal_G3HTgU8UPn.psd"
                );
                
                if (filename && filename.includes('.psd')) {
                    const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                    console.log("[RGBDivider] Manual input:", cleanFilename);
                    downloadPsd(cleanFilename);
                }
            }
            
            // Function to check and update button status
            async function updateButtonStatus() {
                const filename = await fetchLatestPsdFromLog();
                if (filename) {
                    const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                    downloadButton.name = `⬇ Download: ${cleanFilename}`;
                    downloadButton.color = "#4CAF50";
                    downloadButton.bgcolor = "#2E7D32";
                } else {
                    downloadButton.name = "⬇ Download PSD (Run workflow first)";
                    downloadButton.color = "#888888";
                    downloadButton.bgcolor = "#333333";
                }
            }
            
            // Initial button style
            downloadButton.color = "#888888";
            downloadButton.bgcolor = "#333333";
            
            // Monitor node execution to update button appearance
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                console.log("[RGBDivider] Node executed, updating button status...");
                
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Wait a bit for the log file to be written, then update button
                setTimeout(async () => {
                    await updateButtonStatus();
                }, 2000);
            };
            
            // Check for existing log file on load
            (async () => {
                await updateButtonStatus();
            })();
            
            console.log("[RGBDivider] Download button ready");
        }
    }
});

// Global helper function for debugging
window.checkPsdLog = async function() {
    try {
        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
        if (response.ok) {
            const text = await response.text();
            console.log("Log file content:", text);
            return text.trim();
        } else {
            console.log("Log file not found (404)");
        }
    } catch (error) {
        console.error("Error reading log:", error);
    }
    return null;
};

// Global function to manually trigger download
window.downloadLatestPsd = async function() {
    const filename = await window.checkPsdLog();
    if (filename && filename.includes('.psd')) {
        const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
        const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = cleanFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log("Downloaded:", cleanFilename);
    } else {
        console.log("No PSD file found in log");
    }
};

console.log("[RGBLineArtDividerFast] Extension loaded - always fetch latest mode");
