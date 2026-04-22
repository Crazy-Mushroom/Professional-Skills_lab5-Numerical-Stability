import os
import numpy as np

INPUT_PATH = "features/embeddings_float64.npy"
OUTPUT_DIR = "features"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# load the original float64 embeddings (which serves as a baseline reference in the whole pipeline)
emb = np.load(INPUT_PATH)

print("[INFO] Original dtype:", emb.dtype)


# (1) Convert to float 32
emb_f32 = emb.astype(np.float32)
np.save(os.path.join(OUTPUT_DIR, "embeddings_float32.npy"), emb_f32)


# (2) Convert to float 16
emb_f16 = emb.astype(np.float16)
np.save(os.path.join(OUTPUT_DIR, "embeddings_float16.npy"), emb_f16)


# (3) 8 bit, here use int8 quantization
def quantize_int8(x):
    """
    Apply int8 quantization since it is symmetrically linear quantization centered at 0,
    which makes it suitbale for cosine distance computation in the next stage.

    Values range from -max_val to max_val,
    so, map -127 to +127.
    scale = max_val / 127 tells how much each integer step is worth
    """

    # find the largest absolute value (key idea is that int8 is symmetric!)
    max_val = np.max(np.abs(x))  
    
    # scale factor = each int8 step = scale units in original space
    scale = max_val / 127.0
    
    # Important! Avoid division by scale and round to nearest integer
    if scale == 0:
        scale = 1e-8

    x_q = np.round(x / scale).astype(np.int8)

    return x_q, scale


# (4) Convert int8 values back to the original float values
def dequantize_int8(x_q, scale): 
    return x_q.astype(np.float32) * scale

# quantize the embeddings 
emb_q, scale = quantize_int8(emb)

# save the quantized int8 embeddings
np.save(os.path.join(OUTPUT_DIR, "embeddings_int8.npy"), emb_q)

# Sve scale factor! 
# With scale factor, I can convert int8 values back to the float values to compute distance
np.save(os.path.join(OUTPUT_DIR, "embeddings_int8_scale.npy"), np.array([scale]))

print("[INFO] Saved all precision versions")


"""Unsigned Int 8 (An extra comparative test to see if symmetricity affects the cosine distance)"""
# #  UINT8 (Min-Max Quantisation) is not centered around 0, I wonder if this make it less suitable for cosine distance computation
# def quantize_uint8(x):
#     xmin = x.min()
#     xmax = x.max()

#     scale = (xmax - xmin) / 255.0

#     if scale == 0:
#         scale = 1e-8

#     x_q = np.round((x - xmin) / scale).astype(np.uint8)

#     return x_q, xmin, scale


# def dequantize_uint8(x_q, xmin, scale):
#     return x_q.astype(np.float32) * scale + xmin


# # uint8  quantize 
# emb_q, xmin, scale = quantize_uint8(emb)

## save results
# np.save(os.path.join(OUTPUT_DIR, "embeddings_uint8.npy"), emb_q)

# # save the parameters for converting quantized values to float values for computation
# np.save(os.path.join(OUTPUT_DIR, "embeddings_uint8_params.npy"),
#         np.array([xmin, scale]))

# print("[INFO] Saved all precision versions")