# ascii-christmas-tree 🎄

A rotating 3D Christmas tree drawn entirely with coloured ASCII characters.

**Live demo:** https://hcs180.github.io/ascii-christmas-tree/

用彩色 ASCII 字元畫出會旋轉的 3D 聖誕樹，有網頁版和終端機版兩種。

## 網頁版 / Web

直接打開上面的網址，或在本機用瀏覽器開 `index.html`。

- 手機、電腦都能看，畫面大小會自動調整
- 拖曳（或手指滑動）可以轉動聖誕樹，放開後會慢慢停下來

## 終端機版 / Terminal

```bash
pip install numpy
python christmas_tree.py            # Ctrl+C 結束
python christmas_tree.py --speed 0.6 --fps 24
```

需要支援 24-bit 色彩的終端機（Windows Terminal、VS Code、iTerm2、macOS Terminal 都可以）。視窗開大一點、字體調小一點效果最好。

## 原理 / How it works

1. 場景先建成約 7 萬個 3D 點：樹葉（分層圓錐）、金色彩帶（螺旋）、吊球、閃爍燈泡、禮物盒、地面和頂端的星星，每個點都有法向量和顏色。
2. 每一幀把所有點旋轉、透視投影到字元格上，用 z-buffer 讓每一格只保留最靠近鏡頭的點。
3. 依照光線方向算漫射和高光亮度，再從 `.:-~=+*^vxo%&#W@`（暗 → 亮）挑出字元並上色。
4. 背景的星星和飄雪畫在空白的格子上，左下角顯示旋轉角度、燈泡數和字元數。

Inspired by an ASCII-art reel on Instagram; this is an independent implementation.
