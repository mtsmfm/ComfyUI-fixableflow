[fixable-workflow-comfyui.json](https://github.com/user-attachments/files/24276307/fixable-workflow-comfyui.json)[JP](README.md) | [EN](README_EN.md)
# ComfyUI FixableFlow



## Vast.aiを用いた環境構築

<details>

左上のハンバーガーボタンをクリックしてサイドメニューを開き、Templatesを選択します。


<img width="1049" height="608" alt="スクリーンショット 2025-12-21 11 04 30" src="https://github.com/user-attachments/assets/eacfe7e1-fdd9-45a6-904a-de3d99c876b7" />

Templatesの中からComfyUIを選択し、RTX5090をレンタルしてインスタンスを起動します。
（5090でなくても構いませんが、VRAM32GB以上を推奨します）
また、コンテナのVolumeサイズは100GB程度確保していくことを推奨します。


<img width="1293" height="479" alt="スクリーンショット 2025-12-21 11 19 17" src="https://github.com/user-attachments/assets/c820fc5c-ffc2-4b54-afd4-76e8e374a9f1" />
<img width="1177" height="443" alt="スクリーンショット 2025-12-21 11 25 16" src="https://github.com/user-attachments/assets/3f01970a-4a0b-4ecc-a5c9-ef951c72938b" />

インスタンスが起動すると「Open」ボタンが表示されるので、ボタンを押してインスタン内部に入ります。


<img width="905" height="233" alt="スクリーンショット 2025-12-21 11 27 11" src="https://github.com/user-attachments/assets/019d38f5-d2d0-489a-b99e-a54015133ffc" />

インスタンス内部に入ると以下のような画面が表示されるので、まずはJupyter Terminalを開きます。


<img width="1285" height="719" alt="スクリーンショット 2025-12-21 11 39 59" src="https://github.com/user-attachments/assets/d133ea9f-92f0-4601-bf03-6b8ee6e7f217" />


Terminalを開いたら、以下のコマンドを入力します。



```bash
cd /workspace/ComfyUI/custom_nodes/
git clone https://github.com/mattyamonaca/ComfyUI-fixableflow.git
cd /workspace/ComfyUI/custom_nodes/ComfyUI-fixableflow

pip install -r requirements.txt

pip install cython
pip install pytoshop -I --no-cache-dir
pip install psd-tools

cd /workspace/ComfyUI/models/checkpoints/
wget https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors

cd /workspace/ComfyUI/models/loras/
wget https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/sdxl/sdxl-lineart_05.safetensors
wget https://huggingface.co/2vXpSwA7/iroiro-lora/resolve/main/sdxl/sdxl-flat.safetensors
wget https://huggingface.co/tori29umai/FramePack_LoRA/resolve/main/image2flat_V1_1024_dim4-000040.safetensors

cd /workspace/ComfyUI/models/controlnet
wget https://huggingface.co/kataragi/ControlNet-LineartXL/resolve/main/Katarag_lineartXL-fp16.safetensors

cd /workspace/ComfyUI/models/vae
wget https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
wget https://huggingface.co/hunyuanvideo-community/HunyuanVideo/resolve/main/vae/diffusion_pytorch_model.safetensors

cd /workspace/ComfyUI/models/clip
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/clip_l.safetensors
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/llava_llama3_fp16.safetensors

cd /workspace/ComfyUI/models/diffusion_models
wget https://huggingface.co/maybleMyers/framepack_h1111/resolve/main/FramePackI2V_HY_bf16.safetensors

cd /workspace/ComfyUI/models/lora
wget https://huggingface.co/mattyamonaca/framepack-shade-adder_lora/resolve/main/shade-adder-lora.safetensors
```

全ての処理が終わったら、再びインスタンス内部に入った時に表示された最初の画面を開き、ComfyUIを開きます。
ComfyUIを開いたら、以下のファイルをドラッグ&ドロップしてworkflowを読み込ませます。


[Uploading fixable-workflow-comfyui.json…]({
  "id": "1182fbe8-785f-4b30-99cc-9617d9a2d61f",
  "revision": 0,
  "last_node_id": 116,
  "last_link_id": 318,
  "nodes": [
    {
      "id": 4,
      "type": "PreviewImage",
      "pos": [
        4.108536225394843,
        -345.7139314877095
      ],
      "size": [
        140,
        246.00003051757812
      ],
      "flags": {},
      "order": 4,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 306
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.47",
        "Node name for S&R": "PreviewImage"
      },
      "widgets_values": []
    },
    {
      "id": 61,
      "type": "PreviewImage",
      "pos": [
        152.68017077945896,
        -346.83439231653136
      ],
      "size": [
        140,
        246.00003051757812
      ],
      "flags": {},
      "order": 5,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 296
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.47",
        "Node name for S&R": "PreviewImage"
      },
      "widgets_values": []
    },
    {
      "id": 83,
      "type": "PreviewImage",
      "pos": [
        298.9628828965973,
        -347.85233618420295
      ],
      "size": [
        140,
        246.00001525878906
      ],
      "flags": {},
      "order": 9,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 312
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.47",
        "Node name for S&R": "PreviewImage"
      },
      "widgets_values": []
    },
    {
      "id": 110,
      "type": "c78c882d-af86-4833-8c9d-c44b23dc6bd3",
      "pos": [
        -349.2055115322024,
        112.46829320162875
      ],
      "size": [
        210,
        112
      ],
      "flags": {},
      "order": 3,
      "mode": 0,
      "inputs": [
        {
          "name": "pixels",
          "type": "IMAGE",
          "link": 291
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "links": [
            294,
            296,
            311,
            315
          ]
        }
      ],
      "properties": {
        "proxyWidgets": [
          [
            "108",
            "seed"
          ],
          [
            "105",
            "text"
          ]
        ],
        "cnr_id": "comfy-core",
        "ver": "0.5.1"
      },
      "widgets_values": []
    },
    {
      "id": 113,
      "type": "Extract Line Art",
      "pos": [
        -369.7231599053478,
        -123.45782772194156
      ],
      "size": [
        270,
        126
      ],
      "flags": {},
      "order": 2,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 305
        }
      ],
      "outputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "links": [
            306,
            307,
            316
          ]
        },
        {
          "name": "alpha_mask",
          "type": "MASK",
          "links": null
        }
      ],
      "properties": {
        "aux_id": "mattyamonaca/ComfyUI-fixableflow",
        "ver": "c6a4e5063906d32e1cb41a6fdbd5edc7957c6500",
        "Node name for S&R": "Extract Line Art"
      },
      "widgets_values": [
        200,
        true,
        false
      ]
    },
    {
      "id": 116,
      "type": "OverlayImagesNode",
      "pos": [
        -95.832761695225,
        298.73226462270844
      ],
      "size": [
        142.726171875,
        46
      ],
      "flags": {},
      "order": 6,
      "mode": 0,
      "inputs": [
        {
          "name": "input1",
          "type": "IMAGE",
          "link": 316
        },
        {
          "name": "input2",
          "type": "IMAGE",
          "link": 315
        }
      ],
      "outputs": [
        {
          "name": "output",
          "type": "IMAGE",
          "links": [
            317
          ]
        }
      ],
      "properties": {
        "aux_id": "mattyamonaca/ComfyUI-fixableflow",
        "ver": "a58b686de9cfb50cb1565c9ec8ecaa94580ff08d",
        "Node name for S&R": "OverlayImagesNode"
      }
    },
    {
      "id": 97,
      "type": "7f4114d2-b7c6-4926-8173-dca48dc2cb11",
      "pos": [
        -591.1955144788442,
        109.35422119648115
      ],
      "size": [
        210,
        166
      ],
      "flags": {},
      "order": 1,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 238
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "links": [
            291
          ]
        }
      ],
      "properties": {
        "proxyWidgets": [
          [
            "37",
            "seed"
          ],
          [
            "36",
            "text"
          ],
          [
            "35",
            "text"
          ]
        ],
        "cnr_id": "comfy-core",
        "ver": "0.5.1"
      },
      "widgets_values": []
    },
    {
      "id": 96,
      "type": "f03511e2-4119-40d7-8655-b150b1d25ddd",
      "pos": [
        -609.7538281357454,
        -120.11278287898696
      ],
      "size": [
        210,
        166
      ],
      "flags": {
        "collapsed": false
      },
      "order": 0,
      "mode": 0,
      "inputs": [],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "links": [
            238,
            305
          ]
        }
      ],
      "properties": {
        "proxyWidgets": [
          [
            "31",
            "text"
          ],
          [
            "27",
            "seed"
          ],
          [
            "32",
            "text"
          ]
        ],
        "cnr_id": "comfy-core",
        "ver": "0.5.1"
      },
      "widgets_values": []
    },
    {
      "id": 114,
      "type": "ShadowExtractNode",
      "pos": [
        340.751852963048,
        292.4832948668759
      ],
      "size": [
        270,
        126
      ],
      "flags": {},
      "order": 8,
      "mode": 0,
      "inputs": [
        {
          "name": "shade",
          "type": "IMAGE",
          "link": 310
        },
        {
          "name": "base",
          "type": "IMAGE",
          "link": 311
        }
      ],
      "outputs": [
        {
          "name": "output",
          "type": "IMAGE",
          "links": [
            314,
            318
          ]
        }
      ],
      "properties": {
        "aux_id": "mattyamonaca/ComfyUI-fixableflow",
        "ver": "4312246ffb23e20938c36ec7be6814e67ec84591",
        "Node name for S&R": "ShadowExtractNode"
      },
      "widgets_values": [
        1,
        0.5,
        40
      ]
    },
    {
      "id": 111,
      "type": "PreviewImage",
      "pos": [
        450.44111655562494,
        -342.4272508156193
      ],
      "size": [
        140,
        246.00001525878906
      ],
      "flags": {},
      "order": 11,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 318
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.47",
        "Node name for S&R": "PreviewImage"
      },
      "widgets_values": []
    },
    {
      "id": 95,
      "type": "PreviewImage",
      "pos": [
        660.9073224205456,
        122.97009491298647
      ],
      "size": [
        140,
        246.00001525878906
      ],
      "flags": {},
      "order": 12,
      "mode": 0,
      "inputs": [
        {
          "name": "images",
          "type": "IMAGE",
          "link": 235
        }
      ],
      "outputs": [],
      "properties": {
        "cnr_id": "comfy-core",
        "ver": "0.3.47",
        "Node name for S&R": "PreviewImage"
      },
      "widgets_values": []
    },
    {
      "id": 101,
      "type": "1fbabbf1-28e0-4a3b-801f-c86dcbd7f0f1",
      "pos": [
        72.80377531542204,
        297.2875161250094
      ],
      "size": [
        210,
        112
      ],
      "flags": {},
      "order": 7,
      "mode": 0,
      "inputs": [
        {
          "name": "image",
          "type": "IMAGE",
          "link": 317
        }
      ],
      "outputs": [
        {
          "name": "IMAGE",
          "type": "IMAGE",
          "links": [
            310,
            312
          ]
        }
      ],
      "properties": {
        "proxyWidgets": [
          [
            "70",
            "text"
          ],
          [
            "69",
            "seed"
          ]
        ],
        "cnr_id": "comfy-core",
        "ver": "0.5.1"
      },
      "widgets_values": []
    },
    {
      "id": 94,
      "type": "SimplePSDStackNode",
      "pos": [
        335.3829521463468,
        123.71892966831774
      ],
      "size": [
        270,
        122
      ],
      "flags": {},
      "order": 10,
      "mode": 0,
      "inputs": [
        {
          "name": "base",
          "type": "IMAGE",
          "link": 294
        },
        {
          "name": "shade",
          "type": "IMAGE",
          "link": 314
        },
        {
          "name": "lineart",
          "type": "IMAGE",
          "link": 307
        }
      ],
      "outputs": [
        {
          "name": "composite",
          "type": "IMAGE",
          "links": [
            235
          ]
        }
      ],
      "properties": {
        "aux_id": "mattyamonaca/ComfyUI-fixableflow",
        "ver": "e43745449f1c9fe076540545f9382c7504a52cf0",
        "Node name for S&R": "SimplePSDStackNode"
      },
      "widgets_values": [
        "layered",
        "⬇ Download Latest PSD"
      ]
    }
  ],
  "links": [
    [
      235,
      94,
      0,
      95,
      0,
      "IMAGE"
    ],
    [
      238,
      96,
      0,
      97,
      0,
      "IMAGE"
    ],
    [
      291,
      97,
      0,
      110,
      0,
      "IMAGE"
    ],
    [
      294,
      110,
      0,
      94,
      0,
      "IMAGE"
    ],
    [
      296,
      110,
      0,
      61,
      0,
      "IMAGE"
    ],
    [
      305,
      96,
      0,
      113,
      0,
      "IMAGE"
    ],
    [
      306,
      113,
      0,
      4,
      0,
      "IMAGE"
    ],
    [
      307,
      113,
      0,
      94,
      2,
      "IMAGE"
    ],
    [
      310,
      101,
      0,
      114,
      0,
      "IMAGE"
    ],
    [
      311,
      110,
      0,
      114,
      1,
      "IMAGE"
    ],
    [
      312,
      101,
      0,
      83,
      0,
      "IMAGE"
    ],
    [
      314,
      114,
      0,
      94,
      1,
      "IMAGE"
    ],
    [
      315,
      110,
      0,
      116,
      1,
      "IMAGE"
    ],
    [
      316,
      113,
      0,
      116,
      0,
      "IMAGE"
    ],
    [
      317,
      116,
      0,
      101,
      0,
      "IMAGE"
    ],
    [
      318,
      114,
      0,
      111,
      0,
      "IMAGE"
    ]
  ],
  "groups": [],
  "definitions": {
    "subgraphs": [
      {
        "id": "f03511e2-4119-40d7-8655-b150b1d25ddd",
        "version": 1,
        "state": {
          "lastGroupId": 0,
          "lastNodeId": 104,
          "lastLinkId": 250,
          "lastRerouteId": 0
        },
        "revision": 0,
        "config": {},
        "name": "lineart",
        "inputNode": {
          "id": -10,
          "bounding": [
            -1336.5774338155559,
            -140.81303046893657,
            120,
            40
          ]
        },
        "outputNode": {
          "id": -20,
          "bounding": [
            58.815604778504394,
            -150.81303046893657,
            120,
            60
          ]
        },
        "inputs": [],
        "outputs": [
          {
            "id": "9945bf86-b186-4b7a-96e1-f05f3096e8db",
            "name": "IMAGE",
            "type": "IMAGE",
            "linkIds": [
              250
            ],
            "localized_name": "画像",
            "pos": [
              78.8156047785044,
              -130.81303046893657
            ]
          }
        ],
        "widgets": [],
        "nodes": [
          {
            "id": 97,
            "type": "CheckpointLoaderSimple",
            "pos": [
              -1108.8825310080679,
              -399.8086781802514
            ],
            "size": [
              281.8999938964844,
              98
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "モデル",
                "name": "MODEL",
                "type": "MODEL",
                "links": [
                  238
                ]
              },
              {
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP",
                "links": [
                  239
                ]
              },
              {
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE",
                "links": [
                  248
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CheckpointLoaderSimple"
            },
            "widgets_values": [
              "animagine-xl-3.1.safetensors"
            ]
          },
          {
            "id": 98,
            "type": "LoraLoader",
            "pos": [
              -809.6296211182934,
              -396.865174544236
            ],
            "size": [
              270,
              126
            ],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "モデル",
                "name": "model",
                "type": "MODEL",
                "link": 238
              },
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 239
              }
            ],
            "outputs": [
              {
                "localized_name": "モデル",
                "name": "MODEL",
                "type": "MODEL",
                "links": [
                  242
                ]
              },
              {
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP",
                "links": [
                  240,
                  241
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "LoraLoader"
            },
            "widgets_values": [
              "sdxl-lineart_05.safetensors",
              1,
              1
            ]
          },
          {
            "id": 101,
            "type": "FreeMemoryModel",
            "pos": [
              -519.3409617462733,
              -396.5026793126782
            ],
            "size": [
              270,
              58
            ],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "MODEL",
                "link": 242
              }
            ],
            "outputs": [
              {
                "localized_name": "モデル",
                "name": "MODEL",
                "type": "MODEL",
                "links": [
                  243
                ]
              }
            ],
            "properties": {
              "cnr_id": "ComfyUI-FreeMemory",
              "ver": "44fc13f97feec9fdb50ccf342ad64eeb52a95512",
              "Node name for S&R": "FreeMemoryModel"
            },
            "widgets_values": [
              false
            ]
          },
          {
            "id": 102,
            "type": "KSampler",
            "pos": [
              -228.01779187365094,
              -400.17546179534486
            ],
            "size": [
              270,
              262
            ],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "モデル",
                "name": "model",
                "type": "MODEL",
                "link": 243
              },
              {
                "localized_name": "ポジティブ",
                "name": "positive",
                "type": "CONDITIONING",
                "link": 244
              },
              {
                "localized_name": "ネガティブ",
                "name": "negative",
                "type": "CONDITIONING",
                "link": 245
              },
              {
                "localized_name": "潜在画像",
                "name": "latent_image",
                "type": "LATENT",
                "link": 246
              }
            ],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  247
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "KSampler"
            },
            "widgets_values": [
              1108154818735321,
              "randomize",
              20,
              8,
              "euler",
              "simple",
              1
            ]
          },
          {
            "id": 103,
            "type": "EmptyLatentImage",
            "pos": [
              -223.48949241400766,
              -89.86244687059884
            ],
            "size": [
              270,
              106
            ],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  246
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "EmptyLatentImage"
            },
            "widgets_values": [
              1024,
              1024,
              1
            ]
          },
          {
            "id": 104,
            "type": "VAEDecode",
            "pos": [
              -238.34542749583784,
              86.858517038023
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "サンプル",
                "name": "samples",
                "type": "LATENT",
                "link": 247
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 248
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  250
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEDecode"
            },
            "widgets_values": []
          },
          {
            "id": 100,
            "type": "CLIPTextEncode",
            "pos": [
              -807.1002786417904,
              7.741781261187161
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 241
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  245
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "worst quality, low quality, lowres, blurry, jpeg artifacts, bad anatomy,\nbad_proportions, extra arms, extra legs, extra fingers, missing fingers,\ndeformed hands, deformed feet, long torso, short legs, thick legs, overly large breasts, huge breasts, open mouth, messy hair, spiky hair, short hair, strong shadow, dramatic lighting, backlight, complex background,\nscenery, room, outdoor, realistic, photorealistic, 3d, cg render, oil painting, watercolor, shade"
            ]
          },
          {
            "id": 99,
            "type": "CLIPTextEncode",
            "pos": [
              -807.5672172440751,
              -234.1882160241942
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 240
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  244
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "1girl, solo, shy, soft expression, low twin tail, osage, hair between eyes, bangs, osage, hair over one eye, side swept bangs, jitome, blue eyes, school uniform, japanese school uniform, sailor collar, sailor uniform, cardigan, pleated skirt,thighhighs, white background, simple background, clean background, anime style, high quality, masterpiece, best quality"
            ]
          }
        ],
        "groups": [],
        "links": [
          {
            "id": 56,
            "origin_id": 96,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 84,
            "origin_id": 96,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 238,
            "origin_id": 97,
            "origin_slot": 0,
            "target_id": 98,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 239,
            "origin_id": 97,
            "origin_slot": 1,
            "target_id": 98,
            "target_slot": 1,
            "type": "CLIP"
          },
          {
            "id": 240,
            "origin_id": 98,
            "origin_slot": 1,
            "target_id": 99,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 241,
            "origin_id": 98,
            "origin_slot": 1,
            "target_id": 100,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 242,
            "origin_id": 98,
            "origin_slot": 0,
            "target_id": 101,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 243,
            "origin_id": 101,
            "origin_slot": 0,
            "target_id": 102,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 244,
            "origin_id": 99,
            "origin_slot": 0,
            "target_id": 102,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 245,
            "origin_id": 100,
            "origin_slot": 0,
            "target_id": 102,
            "target_slot": 2,
            "type": "CONDITIONING"
          },
          {
            "id": 246,
            "origin_id": 103,
            "origin_slot": 0,
            "target_id": 102,
            "target_slot": 3,
            "type": "LATENT"
          },
          {
            "id": 247,
            "origin_id": 102,
            "origin_slot": 0,
            "target_id": 104,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 248,
            "origin_id": 97,
            "origin_slot": 2,
            "target_id": 104,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 250,
            "origin_id": 104,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          }
        ],
        "extra": {
          "workflowRendererVersion": "LG"
        }
      },
      {
        "id": "7f4114d2-b7c6-4926-8173-dca48dc2cb11",
        "version": 1,
        "state": {
          "lastGroupId": 0,
          "lastNodeId": 97,
          "lastLinkId": 244,
          "lastRerouteId": 0
        },
        "revision": 0,
        "config": {},
        "name": "precolor",
        "inputNode": {
          "id": -10,
          "bounding": [
            -1607.826517252523,
            915.0831966644504,
            120,
            60
          ]
        },
        "outputNode": {
          "id": -20,
          "bounding": [
            519.4169528259404,
            915.0831966644504,
            120,
            60
          ]
        },
        "inputs": [
          {
            "id": "c6c25318-9856-4ea9-bde0-0e5adeda5c02",
            "name": "image",
            "type": "IMAGE",
            "linkIds": [
              237
            ],
            "localized_name": "画像",
            "pos": [
              -1507.826517252523,
              935.0831966644504
            ]
          }
        ],
        "outputs": [
          {
            "id": "4dd17700-1ca8-49bf-8736-fa76d7c46765",
            "name": "IMAGE",
            "type": "IMAGE",
            "linkIds": [
              182
            ],
            "localized_name": "画像",
            "pos": [
              539.4169528259404,
              935.0831966644504
            ]
          }
        ],
        "widgets": [],
        "nodes": [
          {
            "id": 37,
            "type": "KSampler",
            "pos": [
              -10.118446962710223,
              834.4282741118202
            ],
            "size": [
              270,
              262
            ],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "モデル",
                "name": "model",
                "type": "MODEL",
                "link": 127
              },
              {
                "localized_name": "ポジティブ",
                "name": "positive",
                "type": "CONDITIONING",
                "link": 73
              },
              {
                "localized_name": "ネガティブ",
                "name": "negative",
                "type": "CONDITIONING",
                "link": 74
              },
              {
                "localized_name": "潜在画像",
                "name": "latent_image",
                "type": "LATENT",
                "link": 75
              }
            ],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  79
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "KSampler"
            },
            "widgets_values": [
              761738068977132,
              "randomize",
              20,
              8,
              "euler",
              "simple",
              1
            ]
          },
          {
            "id": 41,
            "type": "VAEDecode",
            "pos": [
              319.41695282594037,
              1147.5736870164344
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 10,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "サンプル",
                "name": "samples",
                "type": "LATENT",
                "link": 79
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 87
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  182
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEDecode"
            },
            "widgets_values": []
          },
          {
            "id": 38,
            "type": "ControlNetApplyAdvanced",
            "pos": [
              -305.0011799425716,
              992.2438962245946
            ],
            "size": [
              273,
              186
            ],
            "flags": {},
            "order": 9,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "ポジティブ",
                "name": "positive",
                "type": "CONDITIONING",
                "link": 72
              },
              {
                "localized_name": "ネガティブ",
                "name": "negative",
                "type": "CONDITIONING",
                "link": 71
              },
              {
                "localized_name": "コントロールネット",
                "name": "control_net",
                "type": "CONTROL_NET",
                "link": 77
              },
              {
                "localized_name": "画像",
                "name": "image",
                "type": "IMAGE",
                "link": 244
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "shape": 7,
                "type": "VAE",
                "link": 88
              }
            ],
            "outputs": [
              {
                "localized_name": "ポジティブ",
                "name": "positive",
                "type": "CONDITIONING",
                "links": [
                  73
                ]
              },
              {
                "localized_name": "ネガティブ",
                "name": "negative",
                "type": "CONDITIONING",
                "links": [
                  74
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "ControlNetApplyAdvanced"
            },
            "widgets_values": [
              1.0000000000000002,
              0,
              0.8000000000000002
            ]
          },
          {
            "id": 36,
            "type": "CLIPTextEncode",
            "pos": [
              -839.1605224609375,
              1111.546617106948
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 240
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  71
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "worst quality, low quality, lowres, blurry, jpeg artifacts, bad anatomy,\nbad_proportions, extra arms, extra legs, extra fingers, missing fingers,\ndeformed hands, deformed feet, long torso, short legs, thick legs, overly large breasts, huge breasts, open mouth, messy hair, spiky hair, short hair, strong shadow, dramatic lighting, backlight, complex background,\nscenery, room, outdoor, realistic, photorealistic, 3d, cg render, oil painting, watercolor, shade"
            ]
          },
          {
            "id": 40,
            "type": "ControlNetLoader",
            "pos": [
              -728.3254927157855,
              1356.1625262023313
            ],
            "size": [
              284,
              58
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "コントロールネット",
                "name": "CONTROL_NET",
                "type": "CONTROL_NET",
                "links": [
                  77
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "ControlNetLoader"
            },
            "widgets_values": [
              "Katarag_lineartXL-fp16.safetensors"
            ]
          },
          {
            "id": 44,
            "type": "VAELoader",
            "pos": [
              -715.7420716283272,
              1468.2675539258432
            ],
            "size": [
              270,
              58
            ],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE",
                "links": [
                  87,
                  88
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAELoader"
            },
            "widgets_values": [
              "sdxl_vae.safetensors"
            ]
          },
          {
            "id": 35,
            "type": "CLIPTextEncode",
            "pos": [
              -844.22265625,
              871.5166015625
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 239
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  72
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "1girl, solo, shy, soft expression, low twin tail, osage, black hair,\nwavy hair, hair between eyes, bangs, hair over one eye, side swept bangs, jitome, blue eyes, school uniform, japanese school uniform, sailor collar, sailor uniform, beige cardigan, cardigan, black pleated skirt,\npleated skirt, black thighhighs, thighhighs, white background,\nsimple background, clean background, flat color, anime style, high quality, masterpiece, best quality"
            ]
          },
          {
            "id": 39,
            "type": "EmptyLatentImage",
            "pos": [
              -824.8553936727965,
              721.5971225813042
            ],
            "size": [
              270,
              106
            ],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  75
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "EmptyLatentImage"
            },
            "widgets_values": [
              1024,
              1024,
              1
            ]
          },
          {
            "id": 64,
            "type": "FreeMemoryModel",
            "pos": [
              -830.7708329527055,
              617.7831519062336
            ],
            "size": [
              270,
              58
            ],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "MODEL",
                "link": 238
              }
            ],
            "outputs": [
              {
                "localized_name": "モデル",
                "name": "MODEL",
                "type": "MODEL",
                "links": [
                  127
                ]
              }
            ],
            "properties": {
              "cnr_id": "ComfyUI-FreeMemory",
              "ver": "44fc13f97feec9fdb50ccf342ad64eeb52a95512",
              "Node name for S&R": "FreeMemoryModel"
            },
            "widgets_values": [
              false
            ]
          },
          {
            "id": 43,
            "type": "ImageInvert",
            "pos": [
              -987.5940416658926,
              393.8988394030574
            ],
            "size": [
              140,
              26
            ],
            "flags": {},
            "order": 11,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "画像",
                "name": "image",
                "type": "IMAGE",
                "link": 237
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  241
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "ImageInvert"
            },
            "widgets_values": []
          },
          {
            "id": 29,
            "type": "PreviewImage",
            "pos": [
              -384.9310917436466,
              395.6844952931815
            ],
            "size": [
              140,
              246.00003051757812
            ],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "画像",
                "name": "images",
                "type": "IMAGE",
                "link": 243
              }
            ],
            "outputs": [],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "PreviewImage"
            },
            "widgets_values": []
          },
          {
            "id": 34,
            "type": "CheckpointLoaderSimple",
            "pos": [
              -1427.826517252523,
              618.7169992246666
            ],
            "size": [
              281.8999938964844,
              98
            ],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "モデル",
                "name": "MODEL",
                "type": "MODEL",
                "links": [
                  238
                ]
              },
              {
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP",
                "links": [
                  239,
                  240
                ]
              },
              {
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE",
                "links": []
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CheckpointLoaderSimple"
            },
            "widgets_values": [
              "animagine-xl-3.1.safetensors"
            ]
          },
          {
            "id": 97,
            "type": "MorphologyOperation",
            "pos": [
              -827.3301945386308,
              396.37649594649133
            ],
            "size": [
              270,
              174
            ],
            "flags": {},
            "order": 12,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "image",
                "name": "image",
                "type": "IMAGE",
                "link": 241
              }
            ],
            "outputs": [
              {
                "localized_name": "processed",
                "name": "processed",
                "type": "IMAGE",
                "links": [
                  243,
                  244
                ]
              },
              {
                "localized_name": "comparison",
                "name": "comparison",
                "type": "IMAGE",
                "links": null
              }
            ],
            "properties": {
              "aux_id": "mattyamonaca/ComfyUI-fixableflow",
              "ver": "9f0339a225588bc6a7083b6cf10e224d1b75dada",
              "Node name for S&R": "MorphologyOperation"
            },
            "widgets_values": [
              "close",
              3,
              1,
              "ellipse",
              127
            ]
          }
        ],
        "groups": [],
        "links": [
          {
            "id": 127,
            "origin_id": 64,
            "origin_slot": 0,
            "target_id": 37,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 73,
            "origin_id": 38,
            "origin_slot": 0,
            "target_id": 37,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 74,
            "origin_id": 38,
            "origin_slot": 1,
            "target_id": 37,
            "target_slot": 2,
            "type": "CONDITIONING"
          },
          {
            "id": 75,
            "origin_id": 39,
            "origin_slot": 0,
            "target_id": 37,
            "target_slot": 3,
            "type": "LATENT"
          },
          {
            "id": 79,
            "origin_id": 37,
            "origin_slot": 0,
            "target_id": 41,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 87,
            "origin_id": 44,
            "origin_slot": 0,
            "target_id": 41,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 72,
            "origin_id": 35,
            "origin_slot": 0,
            "target_id": 38,
            "target_slot": 0,
            "type": "CONDITIONING"
          },
          {
            "id": 71,
            "origin_id": 36,
            "origin_slot": 0,
            "target_id": 38,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 77,
            "origin_id": 40,
            "origin_slot": 0,
            "target_id": 38,
            "target_slot": 2,
            "type": "CONTROL_NET"
          },
          {
            "id": 88,
            "origin_id": 44,
            "origin_slot": 0,
            "target_id": 38,
            "target_slot": 4,
            "type": "VAE"
          },
          {
            "id": 237,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 43,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 182,
            "origin_id": 41,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 238,
            "origin_id": 34,
            "origin_slot": 0,
            "target_id": 64,
            "target_slot": 0,
            "type": "MODEL"
          },
          {
            "id": 239,
            "origin_id": 34,
            "origin_slot": 1,
            "target_id": 35,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 240,
            "origin_id": 34,
            "origin_slot": 1,
            "target_id": 36,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 241,
            "origin_id": 43,
            "origin_slot": 0,
            "target_id": 97,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 243,
            "origin_id": 97,
            "origin_slot": 0,
            "target_id": 29,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 244,
            "origin_id": 97,
            "origin_slot": 0,
            "target_id": 38,
            "target_slot": 3,
            "type": "IMAGE"
          }
        ],
        "extra": {
          "workflowRendererVersion": "LG"
        }
      },
      {
        "id": "1fbabbf1-28e0-4a3b-801f-c86dcbd7f0f1",
        "version": 1,
        "state": {
          "lastGroupId": 0,
          "lastNodeId": 100,
          "lastLinkId": 248,
          "lastRerouteId": 0
        },
        "revision": 0,
        "config": {},
        "name": "shade",
        "inputNode": {
          "id": -10,
          "bounding": [
            1551.7208404006337,
            2540.548018945823,
            120,
            60
          ]
        },
        "outputNode": {
          "id": -20,
          "bounding": [
            2982.5334972025908,
            2540.548018945823,
            120,
            60
          ]
        },
        "inputs": [
          {
            "id": "56be2f50-8357-44a0-8542-1840572c463a",
            "name": "image",
            "type": "IMAGE",
            "linkIds": [
              181
            ],
            "localized_name": "image",
            "pos": [
              1651.7208404006337,
              2560.548018945823
            ]
          }
        ],
        "outputs": [
          {
            "id": "73761c32-a8cd-4df7-9706-cc7281bd0f93",
            "name": "IMAGE",
            "type": "IMAGE",
            "linkIds": [
              203,
              204
            ],
            "localized_name": "画像",
            "pos": [
              3002.5334972025908,
              2560.548018945823
            ]
          }
        ],
        "widgets": [],
        "nodes": [
          {
            "id": 70,
            "type": "CLIPTextEncode",
            "pos": [
              2018.0339671447916,
              2501.203466116203
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 242
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  145,
                  146
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "Please add shadow effects to the base image."
            ]
          },
          {
            "id": 55,
            "type": "ImageResize+",
            "pos": [
              1921.839073667252,
              2111.6059507383698
            ],
            "size": [
              270,
              218
            ],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "image",
                "name": "image",
                "type": "IMAGE",
                "link": 181
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  185
                ]
              },
              {
                "localized_name": "width",
                "name": "width",
                "type": "INT",
                "links": null
              },
              {
                "localized_name": "height",
                "name": "height",
                "type": "INT",
                "links": null
              }
            ],
            "properties": {
              "cnr_id": "comfyui_essentials",
              "ver": "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9",
              "Node name for S&R": "ImageResize+"
            },
            "widgets_values": [
              1024,
              1024,
              "nearest",
              "stretch",
              "always",
              0
            ]
          },
          {
            "id": 98,
            "type": "VAELoader",
            "pos": [
              1920.7824215925625,
              2388.111807306717
            ],
            "size": [
              270,
              58
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE",
                "links": [
                  240,
                  241
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAELoader"
            },
            "widgets_values": [
              "diffusion_pytorch_model.safetensors"
            ]
          },
          {
            "id": 80,
            "type": "VAEEncode",
            "pos": [
              2240.0494353665754,
              2383.0334812149426
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "ピクセル",
                "name": "pixels",
                "type": "IMAGE",
                "link": 185
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 241
              }
            ],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  186
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEEncode"
            },
            "widgets_values": []
          },
          {
            "id": 99,
            "type": "DualCLIPLoader",
            "pos": [
              1731.7208404006337,
              2505.4112658266636
            ],
            "size": [
              270,
              130
            ],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP",
                "links": [
                  242
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "DualCLIPLoader"
            },
            "widgets_values": [
              "clip_l.safetensors",
              "llava_llama3_fp16.safetensors",
              "hunyuan_video",
              "default"
            ]
          },
          {
            "id": 81,
            "type": "FramePackLoraSelect",
            "pos": [
              1850.8369783873409,
              2762.0405709116585
            ],
            "size": [
              270,
              106
            ],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "prev_lora",
                "name": "prev_lora",
                "shape": 7,
                "type": "FPLORA",
                "link": null
              }
            ],
            "outputs": [
              {
                "localized_name": "lora",
                "name": "lora",
                "type": "FPLORA",
                "links": [
                  197
                ]
              }
            ],
            "properties": {
              "aux_id": "ShmuelRonen/ComfyUI-FramePackWrapper_Plus",
              "ver": "2.0.1",
              "Node name for S&R": "FramePackLoraSelect",
              "cnr_id": "comfyui-framepackwrapper_plusone"
            },
            "widgets_values": [
              "shade-adder-lora.safetensors",
              1,
              false
            ]
          },
          {
            "id": 71,
            "type": "LoadFramePackModel",
            "pos": [
              2139.264082391159,
              2759.486988353274
            ],
            "size": [
              270,
              174
            ],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "compile_args",
                "name": "compile_args",
                "shape": 7,
                "type": "FRAMEPACKCOMPILEARGS",
                "link": null
              },
              {
                "localized_name": "lora",
                "name": "lora",
                "shape": 7,
                "type": "FPLORA",
                "link": 197
              }
            ],
            "outputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "FramePackMODEL",
                "links": [
                  147
                ]
              }
            ],
            "properties": {
              "aux_id": "ShmuelRonen/ComfyUI-FramePackWrapper_Plus",
              "ver": "2.0.1",
              "Node name for S&R": "LoadFramePackModel",
              "cnr_id": "comfyui-framepackwrapper_plusone"
            },
            "widgets_values": [
              "FramePackI2V_HY_bf16.safetensors",
              "bf16",
              "fp8_e4m3fn",
              "offload_device",
              "sdpa"
            ]
          },
          {
            "id": 73,
            "type": "VAEDecode",
            "pos": [
              2782.5334972025908,
              2509.7691216443595
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "サンプル",
                "name": "samples",
                "type": "LATENT",
                "link": 152
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 240
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  203,
                  204
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEDecode"
            },
            "widgets_values": []
          },
          {
            "id": 69,
            "type": "FramePackSingleFrameSampler",
            "pos": [
              2451.3277912023705,
              2509.4900871532764
            ],
            "size": [
              314.6372985839844,
              550
            ],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "FramePackMODEL",
                "link": 147
              },
              {
                "localized_name": "positive",
                "name": "positive",
                "type": "CONDITIONING",
                "link": 145
              },
              {
                "localized_name": "negative",
                "name": "negative",
                "type": "CONDITIONING",
                "link": 146
              },
              {
                "localized_name": "start_latent",
                "name": "start_latent",
                "type": "LATENT",
                "link": 186
              },
              {
                "localized_name": "image_embeds",
                "name": "image_embeds",
                "shape": 7,
                "type": "CLIP_VISION_OUTPUT",
                "link": null
              },
              {
                "localized_name": "initial_samples",
                "name": "initial_samples",
                "shape": 7,
                "type": "LATENT",
                "link": null
              },
              {
                "localized_name": "reference_latents",
                "name": "reference_latents",
                "shape": 7,
                "type": "REFERENCE_LATENT_LIST",
                "link": null
              },
              {
                "localized_name": "reference_image_embeds",
                "name": "reference_image_embeds",
                "shape": 7,
                "type": "REFERENCE_EMBEDS_LIST",
                "link": null
              },
              {
                "localized_name": "reference_masks",
                "name": "reference_masks",
                "shape": 7,
                "type": "REFERENCE_MASK_LIST",
                "link": null
              },
              {
                "localized_name": "input_mask",
                "name": "input_mask",
                "shape": 7,
                "type": "MASK",
                "link": null
              }
            ],
            "outputs": [
              {
                "localized_name": "samples",
                "name": "samples",
                "type": "LATENT",
                "links": [
                  152
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfyui-framepackwrapper_plusone",
              "ver": "2.0.1",
              "Node name for S&R": "FramePackSingleFrameSampler"
            },
            "widgets_values": [
              25,
              true,
              0.15,
              1,
              10,
              0,
              1044728344377331,
              "randomize",
              9,
              25,
              "unipc_bh1",
              0,
              0,
              "1"
            ]
          }
        ],
        "groups": [],
        "links": [
          {
            "id": 242,
            "origin_id": 99,
            "origin_slot": 0,
            "target_id": 70,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 185,
            "origin_id": 55,
            "origin_slot": 0,
            "target_id": 80,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 241,
            "origin_id": 98,
            "origin_slot": 0,
            "target_id": 80,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 197,
            "origin_id": 81,
            "origin_slot": 0,
            "target_id": 71,
            "target_slot": 1,
            "type": "FPLORA"
          },
          {
            "id": 147,
            "origin_id": 71,
            "origin_slot": 0,
            "target_id": 69,
            "target_slot": 0,
            "type": "FramePackMODEL"
          },
          {
            "id": 145,
            "origin_id": 70,
            "origin_slot": 0,
            "target_id": 69,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 146,
            "origin_id": 70,
            "origin_slot": 0,
            "target_id": 69,
            "target_slot": 2,
            "type": "CONDITIONING"
          },
          {
            "id": 186,
            "origin_id": 80,
            "origin_slot": 0,
            "target_id": 69,
            "target_slot": 3,
            "type": "LATENT"
          },
          {
            "id": 152,
            "origin_id": 69,
            "origin_slot": 0,
            "target_id": 73,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 240,
            "origin_id": 98,
            "origin_slot": 0,
            "target_id": 73,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 181,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 55,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 203,
            "origin_id": 73,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 204,
            "origin_id": 73,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          }
        ],
        "extra": {
          "workflowRendererVersion": "LG"
        }
      },
      {
        "id": "c78c882d-af86-4833-8c9d-c44b23dc6bd3",
        "version": 1,
        "state": {
          "lastGroupId": 0,
          "lastNodeId": 109,
          "lastLinkId": 290,
          "lastRerouteId": 0
        },
        "revision": 0,
        "config": {},
        "name": "base",
        "inputNode": {
          "id": -10,
          "bounding": [
            -1505.1751053055546,
            1048.2998829221062,
            120,
            60
          ]
        },
        "outputNode": {
          "id": -20,
          "bounding": [
            495.1642572874712,
            1048.2998829221062,
            120,
            60
          ]
        },
        "inputs": [
          {
            "id": "846d55a9-411a-44a8-8d59-246a6423200b",
            "name": "pixels",
            "type": "IMAGE",
            "linkIds": [
              261
            ],
            "localized_name": "ピクセル",
            "pos": [
              -1405.1751053055546,
              1068.2998829221062
            ]
          }
        ],
        "outputs": [
          {
            "id": "0cf99ea3-1c02-40e6-9c08-c4a6d115c78b",
            "name": "IMAGE",
            "type": "IMAGE",
            "linkIds": [
              195,
              287,
              288,
              289,
              290
            ],
            "localized_name": "画像",
            "pos": [
              515.1642572874712,
              1068.2998829221062
            ]
          }
        ],
        "widgets": [],
        "nodes": [
          {
            "id": 109,
            "type": "VAEDecode",
            "pos": [
              6.722639190658044,
              1148.218710271485
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 8,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "サンプル",
                "name": "samples",
                "type": "LATENT",
                "link": 259
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 260
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  283
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEDecode"
            },
            "widgets_values": []
          },
          {
            "id": 108,
            "type": "FramePackSingleFrameSampler",
            "pos": [
              -322.80283422472576,
              919.0023910122268
            ],
            "size": [
              314.6372985839844,
              550
            ],
            "flags": {},
            "order": 7,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "FramePackMODEL",
                "link": 255
              },
              {
                "localized_name": "positive",
                "name": "positive",
                "type": "CONDITIONING",
                "link": 256
              },
              {
                "localized_name": "negative",
                "name": "negative",
                "type": "CONDITIONING",
                "link": 257
              },
              {
                "localized_name": "start_latent",
                "name": "start_latent",
                "type": "LATENT",
                "link": 258
              },
              {
                "localized_name": "image_embeds",
                "name": "image_embeds",
                "shape": 7,
                "type": "CLIP_VISION_OUTPUT",
                "link": null
              },
              {
                "localized_name": "initial_samples",
                "name": "initial_samples",
                "shape": 7,
                "type": "LATENT",
                "link": null
              },
              {
                "localized_name": "reference_latents",
                "name": "reference_latents",
                "shape": 7,
                "type": "REFERENCE_LATENT_LIST",
                "link": null
              },
              {
                "localized_name": "reference_image_embeds",
                "name": "reference_image_embeds",
                "shape": 7,
                "type": "REFERENCE_EMBEDS_LIST",
                "link": null
              },
              {
                "localized_name": "reference_masks",
                "name": "reference_masks",
                "shape": 7,
                "type": "REFERENCE_MASK_LIST",
                "link": null
              },
              {
                "localized_name": "input_mask",
                "name": "input_mask",
                "shape": 7,
                "type": "MASK",
                "link": null
              }
            ],
            "outputs": [
              {
                "localized_name": "samples",
                "name": "samples",
                "type": "LATENT",
                "links": [
                  259
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfyui-framepackwrapper_plusone",
              "ver": "2.0.1",
              "Node name for S&R": "FramePackSingleFrameSampler"
            },
            "widgets_values": [
              15,
              true,
              0.15,
              1,
              10,
              0,
              1088812691834509,
              "randomize",
              9,
              25,
              "unipc_bh1",
              0,
              1,
              "5"
            ]
          },
          {
            "id": 107,
            "type": "VAEEncode",
            "pos": [
              -487.5651734388009,
              864.6108625718698
            ],
            "size": [
              140,
              46
            ],
            "flags": {},
            "order": 6,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "ピクセル",
                "name": "pixels",
                "type": "IMAGE",
                "link": 261
              },
              {
                "localized_name": "vae",
                "name": "vae",
                "type": "VAE",
                "link": 254
              }
            ],
            "outputs": [
              {
                "localized_name": "潜在",
                "name": "LATENT",
                "type": "LATENT",
                "links": [
                  258
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAEEncode"
            },
            "widgets_values": []
          },
          {
            "id": 105,
            "type": "CLIPTextEncode",
            "pos": [
              -741.3502500688529,
              956.9593734910004
            ],
            "size": [
              400,
              200
            ],
            "flags": {},
            "order": 3,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "クリップ",
                "name": "clip",
                "type": "CLIP",
                "link": 253
              }
            ],
            "outputs": [
              {
                "localized_name": "条件付け",
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [
                  256,
                  257
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "CLIPTextEncode"
            },
            "widgets_values": [
              "A static illustration changes style to a flat fill without shading without changing the content."
            ]
          },
          {
            "id": 106,
            "type": "VAELoader",
            "pos": [
              -734.6587082327275,
              1201.4282867954134
            ],
            "size": [
              270,
              58
            ],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "VAE",
                "name": "VAE",
                "type": "VAE",
                "links": [
                  254,
                  260
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "VAELoader"
            },
            "widgets_values": [
              "diffusion_pytorch_model.safetensors"
            ]
          },
          {
            "id": 104,
            "type": "DualCLIPLoader",
            "pos": [
              -1035.089392807694,
              952.0060662382357
            ],
            "size": [
              270,
              130
            ],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [],
            "outputs": [
              {
                "localized_name": "CLIP",
                "name": "CLIP",
                "type": "CLIP",
                "links": [
                  253
                ]
              }
            ],
            "properties": {
              "cnr_id": "comfy-core",
              "ver": "0.3.47",
              "Node name for S&R": "DualCLIPLoader"
            },
            "widgets_values": [
              "clip_l.safetensors",
              "llava_llama3_fp16.safetensors",
              "hunyuan_video",
              "default"
            ]
          },
          {
            "id": 103,
            "type": "LoadFramePackModel",
            "pos": [
              -1038.6810065772252,
              717.5973748319857
            ],
            "size": [
              270,
              174
            ],
            "flags": {},
            "order": 4,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "compile_args",
                "name": "compile_args",
                "shape": 7,
                "type": "FRAMEPACKCOMPILEARGS",
                "link": null
              },
              {
                "localized_name": "lora",
                "name": "lora",
                "shape": 7,
                "type": "FPLORA",
                "link": 252
              }
            ],
            "outputs": [
              {
                "localized_name": "model",
                "name": "model",
                "type": "FramePackMODEL",
                "links": [
                  255
                ]
              }
            ],
            "properties": {
              "aux_id": "ShmuelRonen/ComfyUI-FramePackWrapper_Plus",
              "ver": "2.0.1",
              "Node name for S&R": "LoadFramePackModel",
              "cnr_id": "comfyui-framepackwrapper_plusone"
            },
            "widgets_values": [
              "FramePackI2V_HY_bf16.safetensors",
              "bf16",
              "fp8_e4m3fn",
              "offload_device",
              "sdpa"
            ]
          },
          {
            "id": 102,
            "type": "FramePackLoraSelect",
            "pos": [
              -1325.1751053055546,
              737.265340814472
            ],
            "size": [
              270,
              106
            ],
            "flags": {},
            "order": 2,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "prev_lora",
                "name": "prev_lora",
                "shape": 7,
                "type": "FPLORA",
                "link": null
              }
            ],
            "outputs": [
              {
                "localized_name": "lora",
                "name": "lora",
                "type": "FPLORA",
                "links": [
                  252
                ]
              }
            ],
            "properties": {
              "aux_id": "ShmuelRonen/ComfyUI-FramePackWrapper_Plus",
              "ver": "2.0.1",
              "Node name for S&R": "FramePackLoraSelect",
              "cnr_id": "comfyui-framepackwrapper_plusone"
            },
            "widgets_values": [
              "image2flat_V1_1024_dim4-000040.safetensors",
              1,
              true
            ]
          },
          {
            "id": 76,
            "type": "ImageResize+",
            "pos": [
              165.1642572874712,
              1151.6872148994557
            ],
            "size": [
              270,
              218
            ],
            "flags": {},
            "order": 5,
            "mode": 0,
            "inputs": [
              {
                "localized_name": "image",
                "name": "image",
                "type": "IMAGE",
                "link": 283
              }
            ],
            "outputs": [
              {
                "localized_name": "画像",
                "name": "IMAGE",
                "type": "IMAGE",
                "links": [
                  195,
                  287,
                  288,
                  289,
                  290
                ]
              },
              {
                "localized_name": "width",
                "name": "width",
                "type": "INT",
                "links": []
              },
              {
                "localized_name": "height",
                "name": "height",
                "type": "INT",
                "links": null
              }
            ],
            "properties": {
              "cnr_id": "comfyui_essentials",
              "ver": "9d9f4bedfc9f0321c19faf71855e228c93bd0dc9",
              "Node name for S&R": "ImageResize+"
            },
            "widgets_values": [
              1024,
              1024,
              "nearest",
              "stretch",
              "always",
              0
            ]
          }
        ],
        "groups": [],
        "links": [
          {
            "id": 283,
            "origin_id": 109,
            "origin_slot": 0,
            "target_id": 76,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 259,
            "origin_id": 108,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 0,
            "type": "LATENT"
          },
          {
            "id": 260,
            "origin_id": 106,
            "origin_slot": 0,
            "target_id": 109,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 255,
            "origin_id": 103,
            "origin_slot": 0,
            "target_id": 108,
            "target_slot": 0,
            "type": "FramePackMODEL"
          },
          {
            "id": 256,
            "origin_id": 105,
            "origin_slot": 0,
            "target_id": 108,
            "target_slot": 1,
            "type": "CONDITIONING"
          },
          {
            "id": 257,
            "origin_id": 105,
            "origin_slot": 0,
            "target_id": 108,
            "target_slot": 2,
            "type": "CONDITIONING"
          },
          {
            "id": 258,
            "origin_id": 107,
            "origin_slot": 0,
            "target_id": 108,
            "target_slot": 3,
            "type": "LATENT"
          },
          {
            "id": 254,
            "origin_id": 106,
            "origin_slot": 0,
            "target_id": 107,
            "target_slot": 1,
            "type": "VAE"
          },
          {
            "id": 253,
            "origin_id": 104,
            "origin_slot": 0,
            "target_id": 105,
            "target_slot": 0,
            "type": "CLIP"
          },
          {
            "id": 252,
            "origin_id": 102,
            "origin_slot": 0,
            "target_id": 103,
            "target_slot": 1,
            "type": "FPLORA"
          },
          {
            "id": 261,
            "origin_id": -10,
            "origin_slot": 0,
            "target_id": 107,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 195,
            "origin_id": 76,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 287,
            "origin_id": 76,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 288,
            "origin_id": 76,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 289,
            "origin_id": 76,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          },
          {
            "id": 290,
            "origin_id": 76,
            "origin_slot": 0,
            "target_id": -20,
            "target_slot": 0,
            "type": "IMAGE"
          }
        ],
        "extra": {
          "workflowRendererVersion": "LG"
        }
      }
    ]
  },
  "config": {},
  "extra": {
    "ds": {
      "scale": 0.9978036371039809,
      "offset": [
        387.21686620029925,
        202.93457411337576
      ]
    },
    "frontendVersion": "1.34.9",
    "workflowRendererVersion": "LG"
  },
  "version": 0.4
})



**ComfyUI LayerDivider** is custom nodes that generating layered psd files inside ComfyUI, original implement is [mattyamonaca/layerdivider](https://github.com/mattyamonaca/layerdivider)

![image1](docs/layerdivider-color-base.png)
![image2](docs/layerdivider-seg-mask.png)
https://github.com/jtydhr88/ComfyUI-LayerDivider/assets/860985/3ceb0638-1ed7-4e01-b231-03c4408c95e3

## Environment
I tested the following environment, it might work on other environment, but I don't test:
### Common
1. Windows 10/Ubuntu
2. GTX 3090
3. Cuda 12.1

### Env 1 - see Method 1
1. ComfyUI embedded python (python 3.11) and ComfyUI Manager

### Env 2 - see Method 2
1. conda
2. Python 3.11

### Env 3 - see Method 3
1. conda
2. Python 3.11

### Env 4 - see Method 4
1. Ubuntu
2. conda/Python 3.11
3. cuda 12.1

## (Common) Installation - CUDA & cuDNN
This repo requires specific versions of CUDA and cuDNN to be installed locally:
- For CUDA, I only install and test CUDA 12.1, you can find it from https://developer.nvidia.com/cuda-12-1-0-download-archive
- For cuDNN, it MUST be v8.9.2 - CUDA 12.x (according to https://github.com/mdboom/pytoshop/issues/9), you can find it from https://developer.nvidia.com/rdp/cudnn-archive
- After install and unzip, make sure you configure the PATH of your system variable ![Path](docs/paths.png)

## (Common) Installation - Visual Studio Build Tools
It might also require Visual Studio Build Tools.
However, I am not sure because my local already installed previously. 
If it needs, you can find from [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/?q=build+tools).

## (Method 1) Installation - ComfyUI Embedded Plugin & ComfyUI Manager 
1. You could clone this repo inside **comfyUI/custom_notes** directly `git clone https://github.com/jtydhr88/ComfyUI-LayerDivider.git`
2. or use ComfyUI Manager ![manager](docs/comfyui-manager.png)
3. However, no matter which way you choose, it will fail at first time ![error](docs/error.png)
4. Stop ComfyUI
5. Then go to **custom_nodes\ComfyUI-LayerDivider**, and run **install_windows_portable_win_py311_cu121.bat**

Done!

(If you prefer to use conda and python 3.10, you could follow the next)
## (Method 2) Installation - ComfyUI
You could use conda to manage and create the ComfyUI runtime environment:
- use cmd/terminal to enter the comfyui root folder (which includes run_cpu.bat and run_nvidia_gpu.bat) 
- `conda create --name comfy-py-310 python=3.10`
- `conda activate comfy-py-310`
- `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121`
- `pip install -r ComfyUI\requirements.txt`

Then you can run `python -s ComfyUI\main.py --windows-standalone-build` to check ComfyUI running properly. 

## (Method 2) Installation - ComfyUI LayerDivider
Then we can clone and configure this repo for ComfyUI:
- `cd ComfyUI\custom_nodes`
- `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
- `pip install Cython`
- `pip install pytoshop -I --no-cache-dir`
- `pip install psd-tools --no-deps`
- `git clone https://github.com/jtydhr88/ComfyUI-LayerDivider.git`
- `cd ComfyUI-LayerDivider`
- `pip install -r requirements.txt`

Congratulation! You complete all installation!

## (Method 3) Installation - ComfyUI LayerDivider
Assume you already have a conda python3.11 env
- activate your env
- go into this folder and run install_conda_win_py311_cu121.bat

Congratulation! You complete all installation!

## (Method 4) Ubuntu Installation - ComfyUI LayerDivider
Assume you already have a python3.11 env + cuda 12.1
- clone this repo inside custom_nodes folder
- cd ComfyUI-LayerDivider/
- pip install -r requirements.txt

Then make sure run them one by one:
- pip install cython
- pip install pytoshop -I --no-cache-dir
- pip install psd_tools
- pip install onnxruntime-gpu==1.17.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

Congratulation! You complete all installation!


## Node Introduction
Currently, this extension includes two modes with four custom nodes, plus two layer modes(normal and composite) for each mode:

### Mode
There are two main layered segmentation modes:
- Color Base - Layers based on similar colors, with parameters:
  - loops 
  - init_cluster 
  - ciede_threshold 
  - blur_size
- Segment Mask - First, the image is divided into segments using [SAM - segment anything](https://segment-anything.com/) to generate corresponding masks, then layers are created based on these masks.
  - Load SAM Mask Generator, with parameters (These come from segment anything, please refer to [here](https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/automatic_mask_generator.py#L61) for more details):
    - pred_iou_thresh 
    - stability_score_thresh 
    - min_mask_region_area
  - LayerDivider - Segment Mask, with parameters:
    - area_th: determines the number of partitions. The smaller the value, the more partitions there will be; the larger the value, the fewer partitions there will be.

### Layer Mode
Using in Divide Layer node to decide the layer mode:
- normal - Generates three layers for each region:
  - base - The base layer is the starting point for image processing
  - bright - The bright layer focuses on the brightest parts of the image, enhancing the brightness and gloss of these areas
  - shadow - The shadow layer deals with the darker parts of the image, emphasizing the details of shadows and dark areas.
- composite - Generates five layers for each region:
  - base - The base layer is the starting point of the image
  - screen - The screen layer simulates the effect of light overlay. It multiplies the color values of the image with the color values of the layer above it and then inverts the result, producing a brighter effect than the original image
  - multiply - The multiply layer simulates the effect of multiple images being overlaid. It directly multiplies the color values of the image with the color values of the layer above it, resulting in a darker effect than the original image.
  - subtract - The subtract layer subtracts the color values of the layer above from the base image, resulting in an image with lower color values.
  - addition - The addition layer adds the color values of the layer above to the base image, resulting in an image with higher color values.

## Example workflows
Here are two workflows for reference:
- [layerdivider-color-base.json](workflows/layerdivider-color-base.json) ![color-base](docs/layerdivider-color-base.png)
- [layerdivider-seg-mask.json](workflows/layerdivider-seg-mask-workflow.json) ![color-base](docs/layerdivider-seg-mask.png)

## Example outputs
- [output_color_base_composite.psd](docs/output_color_base_composite.psd)
- [output_color_base_normal.psd](docs/output_color_base_normal.psd)
- [output_seg_mask_composite.psd](docs/output_seg_mask_composite.psd)
- [output_seg_mask_normal.psd](docs/output_seg_mask_normal.psd)

## Known issues
Sometimes, composite mode will fail on some images, such as ComfyUI example image, still under invesgating the cause

## Credit & Thanks
- [mattyamonaca/layerdivider](https://github.com/mattyamonaca/layerdivider) - Original implement
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - A powerful and modular stable diffusion GUI.

## My extensions for ComfyUI
- [ComfyUI-Unique3D](https://github.com/jtydhr88/ComfyUI-Unique3D) - ComfyUI Unique3D is custom nodes that running Unique3D into ComfyUI
- [ComfyUI-LayerDivider](https://github.com/jtydhr88/ComfyUI-LayerDivider) - ComfyUI InstantMesh is custom nodes that generating layered psd files inside ComfyUI
- [ComfyUI-InstantMesh](https://github.com/jtydhr88/ComfyUI-InstantMesh) - ComfyUI InstantMesh is custom nodes that running InstantMesh into ComfyUI
- [ComfyUI-ImageMagick](https://github.com/jtydhr88/ComfyUI-ImageMagick) - This extension implements custom nodes that integreated ImageMagick into ComfyUI
- [ComfyUI-Workflow-Encrypt](https://github.com/jtydhr88/ComfyUI-Workflow-Encrypt) - Encrypt your comfyui workflow with key

## My extensions for stable diffusion webui
- [3D Model/pose loader](https://github.com/jtydhr88/sd-3dmodel-loader) A custom extension for AUTOMATIC1111/stable-diffusion-webui that allows you to load your local 3D model/animation inside webui, or edit pose as well, then send screenshot to txt2img or img2img as your ControlNet's reference image.
- [Canvas Editor](https://github.com/jtydhr88/sd-canvas-editor) A custom extension for AUTOMATIC1111/stable-diffusion-webui that integrated a full capability canvas editor which you can use layer, text, image, elements and so on, then send to ControlNet, basing on Polotno.
- [StableStudio Adapter](https://github.com/jtydhr88/sd-webui-StableStudio) A custom extension for AUTOMATIC1111/stable-diffusion-webui to extend rest APIs to do some local operations, using in StableStudio.
- [Txt/Img to 3D Model](https://github.com/jtydhr88/sd-webui-txt-img-to-3d-model) A custom extension for sd-webui that allow you to generate 3D model from txt or image, basing on OpenAI Shap-E.
- [3D Editor](https://github.com/jtydhr88/sd-webui-3d-editor) A custom extension for sd-webui that with 3D modeling features (add/edit basic elements, load your custom model, modify scene and so on), then send screenshot to txt2img or img2img as your ControlNet's reference image, basing on ThreeJS editor.


