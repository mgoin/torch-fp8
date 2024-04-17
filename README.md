
Scripts to compare FP8 weight and activation quantization performance against FP16 within PyTorch

The below numbers were collected on an NVIDIA RTX 4090 using `torch==2.4.0.dev20240417+cu121`

![image](https://github.com/mgoin/torch-fp8/assets/3195154/fb769578-5539-4c37-93cb-e8f044b4f958)

Output similarity:
```
python scaled_mm_example.py
Testing with x=torch.Size([16, 16]) and w=torch.Size([16, 16])
cos_sim 0.9990
```

Benchmarking:
```
python benchmark_scaled_mm.py
M, N, K, Cosine similarity, FP16 time, FP8 time, FP8 dynamic quant time, Speedup, Speedup dynamic quant
1, 4096, 12288, 0.9990, 1.2405, 1.6594, 1.9328, 0.748, 0.642
2, 4096, 12288, 0.9995, 1.3623, 1.6419, 1.9112, 0.830, 0.713
4, 4096, 12288, 0.9995, 1.3666, 1.6452, 1.9343, 0.831, 0.707
8, 4096, 12288, 0.9990, 1.3703, 1.6376, 1.9694, 0.837, 0.696
16, 4096, 12288, 0.9995, 1.3813, 1.6393, 2.1957, 0.843, 0.629
32, 4096, 12288, 0.9995, 1.3468, 0.2522, 1.0030, 5.340, 1.343
64, 4096, 12288, 0.9995, 1.5338, 0.8611, 1.4468, 1.781, 1.060
128, 4096, 12288, 0.0000, 1.5056, 0.8600, 1.5193, 1.751, 0.991
256, 4096, 12288, 0.0000, 1.8017, 0.9122, 1.5905, 1.975, 1.133
1, 4096, 4096, 0.9995, 0.4376, 0.5789, 0.9214, 0.756, 0.475
2, 4096, 4096, 0.9990, 0.1489, 0.5752, 0.9299, 0.259, 0.160
4, 4096, 4096, 0.9990, 0.1646, 0.5787, 0.9239, 0.284, 0.178
8, 4096, 4096, 0.9995, 0.1723, 0.5771, 0.9268, 0.299, 0.186
16, 4096, 4096, 0.9990, 0.1902, 0.5757, 0.9318, 0.330, 0.204
32, 4096, 4096, 0.9990, 0.2382, 0.1193, 0.9896, 1.997, 0.241
64, 4096, 4096, 0.9990, 0.3026, 0.1813, 0.9825, 1.668, 0.308
128, 4096, 4096, 0.9985, 0.3722, 0.3167, 0.9847, 1.175, 0.378
256, 4096, 4096, 0.9985, 0.6471, 0.3547, 1.1716, 1.824, 0.552
1, 4096, 22016, 0.9995, 2.1628, 3.0480, 3.2505, 0.710, 0.665
2, 4096, 22016, 0.9990, 2.2017, 2.9144, 3.2160, 0.755, 0.685
4, 4096, 22016, 0.9990, 2.1653, 2.9169, 3.2556, 0.742, 0.665
8, 4096, 22016, 0.9990, 2.1702, 2.9197, 3.5074, 0.743, 0.619
16, 4096, 22016, 0.9990, 2.1750, 2.9164, 3.4882, 0.746, 0.624
32, 4096, 22016, 0.9995, 2.2520, 1.1427, 1.8096, 1.971, 1.244
64, 4096, 22016, 0.0000, 2.6013, 1.5340, 2.2587, 1.696, 1.152
128, 4096, 22016, 0.0000, 2.5697, 1.5463, 2.3586, 1.662, 1.090
256, 4096, 22016, 0.0000, 3.2884, 1.7407, 2.6083, 1.889, 1.261
1, 11008, 4096, 0.9990, 1.0823, 0.5845, 0.9331, 1.852, 1.160
2, 11008, 4096, 0.9995, 1.1350, 0.5828, 0.9547, 1.947, 1.189
4, 11008, 4096, 0.9990, 1.1383, 0.5804, 0.9262, 1.961, 1.229
8, 11008, 4096, 0.9995, 1.1117, 0.5835, 0.9497, 1.905, 1.171
16, 11008, 4096, 0.9995, 1.1205, 0.5824, 0.9543, 1.924, 1.174
32, 11008, 4096, 0.9995, 1.1497, 0.3212, 0.9991, 3.579, 1.151
64, 11008, 4096, 0.9990, 1.2416, 0.3394, 1.0191, 3.659, 1.218
128, 11008, 4096, 0.0000, 1.3398, 0.5110, 1.0898, 2.622, 1.229
256, 11008, 4096, 0.0000, 1.8516, 0.9034, 1.5081, 2.049, 1.228
```
