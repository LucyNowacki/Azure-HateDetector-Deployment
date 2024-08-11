// Definitions for __nv_bfloat16 if supported
#ifdef BFLOAT16_SUPPORTED
template <> __device__ __nv_bfloat16 hexp(const __nv_bfloat16 x) {
    return __float2bfloat16(expf(__bfloat162float(x)));
}

template <> __device__ __nv_bfloat16 hlog(const __nv_bfloat16 x) {
    return __float2bfloat16(logf(__bfloat162float(x)));
}

template <> __device__ __nv_bfloat16 hsinh(const __nv_bfloat16 x) {
    float e2x = expf(-2.0f * __bfloat162float(x));
    return __float2bfloat16((1.0f - e2x) / (1.0f + e2x));
}

template <> __device__ __nv_bfloat16 hsigmoid(const __nv_bfloat16 x) {
    return __float2bfloat16(1.0f / (1.0f + expf(__bfloat162float(x))));
}

template <> __device__ __nv_bfloat16 hlogaddexp(const __nv_bfloat16 x) {
    return __float2bfloat16(logf(1.0f + expf(__bfloat162float(x))));
}
#endif

// Definitions for __half
template <> __device__ __half hexp(const __half x) {
    return __float2half(expf(__half2float(x)));
}

template <> __device__ __half hlog(const __half x) {
    return __float2half(logf(__half2float(x)));
}

template <> __device__ __half hsinh(const __half x) {
    float e2x = expf(-2.0f * __half2float(x));
    return __float2half((1.0f - e2x) / (1.0f + e2x));
}

template <> __device__ __half hsigmoid(const __half x) {
    return __float2half(1.0f / (1.0f + expf(-__half2float(x))));
}

template <> __device__ __half hlogaddexp(const __half x) {
    return __float2half(logf(1.0f + expf(__half2float(x))));
}

// CONSTANTS for __half
template <> __device__ __forceinline__ __half dscalar_three() {
    return __float2half(3.0f);
}

template <> __device__ __forceinline__ __half dscalar_two() {
    return __float2half(2.0f);
}

template <> __device__ __forceinline__ __half dscalar_one() {
    return __float2half(1.0f);
}

template <> __device__ __forceinline__ __half dscalar_half() {
    return __float2half(0.5f);
}

template <> __device__ __forceinline__ __half dscalar_zero() {
    return __float2half(0.0f);
}

template <> __forceinline__ __half scalar_one() { return __float2half(1.0f); }

template <> __forceinline__ __half scalar_zero() { return __float2half(0.0f); }

template <> __device__ __forceinline__ __half dscalar(double x) {
    return __float2half((float)x);
}

// Arithmetic functions for __half
template <>
__device__ __forceinline__ __half add_g(const __half a, const __half b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ __half sub_g(const __half a, const __half b) {
    return __hsub(a, b);
}

template <> __device__ __forceinline__ __half neg_g(const __half a) {
    return __hneg(a);
}

template <>
__device__ __forceinline__ __half mul_g(const __half a, const __half b) {
    return __hmul(a, b);
}

template <>
__device__ __forceinline__ __half div_g(const __half a, const __half b) {
    return __hdiv(a, b);
}

// Comparison operations for __half
template <>
__device__ __forceinline__ bool gt_g(const __half a, const __half b) {
    return __hgt(a, b);
}

template <>
__device__ __forceinline__ bool lt_g(const __half a, const __half b) {
    return __hlt(a, b);
}

template <> __device__ __forceinline__ bool gt_zero_g(const __half a) {
    return __hgt(a, __float2half(0.0f));
}

template <> __device__ __forceinline__ bool eq_zero_g(const __half a) {
    return __heq(a, __float2half(0.0f));
}

template <> __device__ __forceinline__ bool lt_zero_g(const __half a) {
    return __hlt(a, __float2half(0.0f));
}

// Other functions for __half
template <> __device__ __forceinline__ __half exp_g(const __half x) {
    return hexp(x);
}

template <> __device__ __forceinline__ __half log_g(const __half x) {
    return hlog(x);
}

template <> __device__ __forceinline__ __half tanh_g(const __half x) {
    __half zero = dscalar_zero<__half>();
    __half one = dscalar_one<__half>();
    __half two = dscalar_two<__half>();
    __half e2x;
    __half negx = x;
    if (gt_g(x, zero)) {
        negx = __hneg(x);
    }
    e2x = hexp(__hmul(two, negx));
    e2x = __hdiv(__hsub(one, e2x), __hadd(one, e2x));
    if (gt_g(x, zero)) {
        return e2x;
    } else {
        return __hneg(e2x);
    }
}

template <> __device__ __forceinline__ __half sigmoid_g(const __half x) {
    __half one = dscalar_one<__half>();
    __half expx;
    __half negx = x;
    if (gt_zero_g(x)) {
        negx = __hneg(x);
    }
    expx = __hdiv(one, __hadd(one, hexp(negx)));
    if (gt_zero_g(x)) {
        return expx;
    } else {
        return sub_g(one, expx);
    }
}

template <> __device__ __forceinline__ __half logsigmoid_g(const __half x) {
    __half one = dscalar_one<__half>();
    __half negx = x;
    if (gt_zero_g(x)) {
        negx = __hneg(x);
    }
    __half logaddexpnx = hlog(__hadd(one, hexp(negx)));
    if (gt_zero_g(x)) {
        return __hneg(logaddexpnx);
    } else {
        return __hsub(x, logaddexpnx);
    }
}

template <>
__device__ __forceinline__ __half sigmoid_unstable_g(const __half x) {
    __half one = dscalar_one<__half>();
    return __hdiv(one, __hadd(one, hexp(__hneg(x))));
}

template <>
__device__ __forceinline__ __half d_sigmoid_g(const __half sigmoid_output) {
    return __hmul(sigmoid_output, __hsub(dscalar_one<__half>(), sigmoid_output));
}

template <>
__device__ __forceinline__ __half d_tanh_g(const __half tanh_output) {
    return __hsub(dscalar_one<__half>(), __hmul(tanh_output, tanh_output));
}

template <> __device__ __forceinline__ __half max_g(const __half a, const __half b) {
    return __hmax(a, b);
}

template <> __device__ __forceinline__ __half min_g(const __half a, const __half b) {
    return __hmin(a, b);
}

template <> __device__ __forceinline__ float type2float(const __half x) {
    return __half2float(x);
}

template <> __device__ __forceinline__ __half float2type(const float x) {
    return __float2half(x);
}

template <> __device__ __forceinline__ bool isnan_g(const __half x) {
    return __hisnan(x);
}

template <> __device__ __forceinline__ bool isinf_g(const __half x) {
    return __hisinf(x);
}
