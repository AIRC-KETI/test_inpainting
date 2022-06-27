
### 1. Your own model should be in: 
```bash
${ROOT}
├── data
├── model
    └── [your model.py]
├── model_layout2img
├── scripts
├── utils
├── utils_layout2img
├── INSTALL.md
├── README.md
├── requirements.txt
├── test_model.py
└── test_samples.py
```

### 2. test model should get two or more input with 'dict' type content

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1, 1]                | [Height,Width, 3]  | masked input (gray color)                     |
| mask               | float,int | [0, 1]                | [Height, Width, 1] | mask(1 to mask in each pixel)                 |
| label (optional)   | int       | [0, # of label index) | [# of objects]     | object index (defined in each dataset)        |
| bbox (optional)    | float     | [0, 1]                | [# of objects, 4]  | bound box with (x, y, w, h) relative position |
| triples (optional) | int       | [0,# of triples)      | [# of triples, 3]  | triples (index defined in each dataset)       |

### 3. The result of the model should also be output in the form of a dict.

| key                | type      | range                 | shape              | descrption                                    |
|--------------------|-----------|-----------------------|--------------------|-----------------------------------------------|
| image_contents     | float     | [-1,1]                | [Height,Width, 3]  |inpaintied results                     |

### Model List
- [X] TripleLostGAN (Ours)
- [X] LostGAN
- [ ] CAL2IM
- [ ] CIAFILL
- [ ] Hyein et al.