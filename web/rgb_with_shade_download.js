/**
 * RGBLineArtDividerWithShade Web Extension
 * ダウンロードボタンを追加
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerWithShade",
    
    async nodeCreated(node) {
        // RGBLineArtDividerWithShade ノードにのみ適用
        if (node.comfyClass === "RGBLineArtDividerWithShade") {
            console.log("[RGBWithShade] Setting up download button");
            
            // ダウンロードボタンを追加
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download PSD with Shade",
                async () => {
                    console.log("[RGBWithShade] Download button clicked");
                    
                    // 最新のファイル名をログファイルから取得
                    try {
                        // キャッシュバスティング用のタイムスタンプ付きでfetch
                        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
                        
                        if (response.ok) {
                            const text = await response.text();
                            const filename = text.trim();
                            
                            if (filename && filename.includes('.psd')) {
                                console.log("[RGBWithShade] Found file in log:", filename);
                                
                                // すぐにダウンロード
                                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                                const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                                console.log("[RGBWithShade] Download URL:", downloadUrl);
                                
                                const link = document.createElement('a');
                                link.href = downloadUrl;
                                link.download = cleanFilename;
                                link.style.display = 'none';
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                                
                                console.log("[RGBWithShade] Downloaded:", cleanFilename);
                                
                                // ユーザーフィードバック
                                node.widgets.find(w => w.name === "Download PSD").value = "✅ Downloaded!";
                                setTimeout(() => {
                                    node.widgets.find(w => w.name === "Download PSD").value = "⬇ Download PSD with Shade";
                                }, 2000);
                            } else {
                                console.log("[RGBWithShade] Log file exists but no valid PSD filename found");
                                promptForManualInput();
                            }
                        } else {
                            console.log("[RGBWithShade] Log file not found (404)");
                            promptForManualInput();
                        }
                    } catch (error) {
                        console.error("[RGBWithShade] Error reading log:", error);
                        promptForManualInput();
                    }
                    
                    // 手動入力用の関数
                    function promptForManualInput() {
                        const filename = prompt(
                            "PSDファイルが見つかりません。\n" +
                            "ワークフローを実行するか、手動でファイル名を入力してください。\n" +
                            "例: output_rgb_shade_ABC123.psd"
                        );
                        
                        if (filename && filename.includes('.psd')) {
                            const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                            console.log("[RGBWithShade] Manual input:", cleanFilename);
                            
                            const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                            const link = document.createElement('a');
                            link.href = downloadUrl;
                            link.download = cleanFilename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            console.log("[RGBWithShade] Manual download completed");
                            
                            // フィードバック
                            node.widgets.find(w => w.name === "Download PSD").value = "✅ Downloaded!";
                            setTimeout(() => {
                                node.widgets.find(w => w.name === "Download PSD").value = "⬇ Download PSD with Shade";
                            }, 2000);
                        }
                    }
                }
            );
            
            // ボタンスタイル
            downloadButton.color = "#10B981";  // 緑色
            downloadButton.bgcolor = "#059669";
            
            // PSDファイル名を表示するウィジェットを追加（オプション）
            node.addWidget(
                "text",
                "Last PSD File",
                "",
                () => {},
                { 
                    serialize: false,
                    multiline: false
                }
            );
            
            // ノードの出力が更新されたときにファイル名を表示
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(message) {
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // 出力からファイル名を取得（psd_filename）
                if (message && message.output && message.output.psd_filename) {
                    const filename = message.output.psd_filename[0];
                    const filenameWidget = node.widgets.find(w => w.name === "Last PSD File");
                    if (filenameWidget) {
                        filenameWidget.value = filename;
                        console.log("[RGBWithShade] Updated filename widget:", filename);
                    }
                }
            };
            
            console.log("[RGBWithShade] Download button ready");
        }
    }
});

// デバッグ用のグローバルヘルパー関数
window.checkShadeLog = async function() {
    try {
        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
        if (response.ok) {
            const text = await response.text();
            console.log("Shade log file content:", text);
            return text.trim();
        } else {
            console.log("Shade log file not found (404)");
        }
    } catch (error) {
        console.error("Error reading shade log:", error);
    }
    return null;
};

// 手動でダウンロードをトリガーする関数
window.downloadLatestShadePsd = async function() {
    const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
    if (response.ok) {
        const filename = (await response.text()).trim();
        if (filename && filename.includes('.psd')) {
            const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
            const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
            
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = cleanFilename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log("Downloaded Shade PSD:", cleanFilename);
            return true;
        }
    }
    console.log("No Shade PSD file found in log");
    return false;
};

console.log("[RGBLineArtDividerWithShade] Extension loaded");
