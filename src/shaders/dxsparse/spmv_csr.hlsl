// Sparse Matrix-Vector Product (SpMV) in Compressed Sparse Row (CSR) Format

RWStructuredBuffer<float> g_y : register(u0);
StructuredBuffer<uint> g_row_ptr : register(t0);
StructuredBuffer<uint> g_col_ind : register(t1);
StructuredBuffer<float> g_values : register(t2);
StructuredBuffer<float> g_x : register(t3);

cbuffer SpMVCB : register(b0) {
    uint g_num_rows;
    uint3 g_pad;
};

[numthreads(64, 1, 1)]
void spmv_csr_kernel(uint3 id : SV_DispatchThreadID) {
    uint row = id.x;
    if (row >= g_num_rows) return;

    uint start = g_row_ptr[row];
    uint end = g_row_ptr[row + 1];

    float sum = 0.0f;
    for (uint i = start; i < end; ++i) {
        uint col = g_col_ind[i];
        float val = g_values[i];
        sum += val * g_x[col];
    }
    g_y[row] = sum;
}
