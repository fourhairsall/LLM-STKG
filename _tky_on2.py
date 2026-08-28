import numpy as np, time, sys
N_full = 61858
N_sub = 6000
# BGE emb float64 footprint
emb_bytes = N_full*768*8
sim_matrix_bytes_full_f64 = N_full*N_full*8
sim_matrix_bytes_full_f32 = N_full*N_full*4
print("TKY_FULL_N", N_full)
print("BGE_VEC_MB_f64", round(emb_bytes/1e6,1))
print("SIM_MAT_GB_f64", round(sim_matrix_bytes_full_f64/1e9,2))
print("SIM_MAT_GB_f32", round(sim_matrix_bytes_full_f32/1e9,2))
# measure O(N^2) similarity time on subsample, extrapolate
rng = np.random.RandomState(0)
Vsub = rng.randn(N_sub,768).astype(np.float32)
Vsub /= (np.linalg.norm(Vsub,axis=1,keepdims=True)+1e-8)
t0=time.time()
S = Vsub @ Vsub.T
_ = (S>=0.90).sum()
t1=time.time()
sub_time = t1-t0
# O(N^2) scaling: time ~ a*N^2 ; a = sub_time / N_sub^2
a = sub_time/(N_sub**2)
full_time_est = a*(N_full**2)
print("SUB_6000_matmul_s", round(sub_time,3))
print("FULL_61858_matmul_est_s", round(full_time_est,1))
print("FULL_61858_matmul_est_min", round(full_time_est/60,1))
