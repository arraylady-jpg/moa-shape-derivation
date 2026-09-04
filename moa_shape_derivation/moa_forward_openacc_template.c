/* ============================================================================
 * moa_forward_openacc_tiled.c
 * OpenACC realization of Paper I's forward pass, v3: properly tiled,
 * matching Triton's actual strategy rather than v1/v2's per-row-gang
 * design (moa_forward_openacc.c). This is not a tuning tweak -- it is
 * a structural rewrite intended to remove the two suspected causes of
 * the v1/v2 scaling anomaly (moa_delta_benchmarking_tr.pdf, Sections
 * 7.11, 9) at the source, rather than working around them:
 *
 *   1. UNBOUNDED PRIVATE ARRAY, ELIMINATED. v1/v2 held arow[N] private
 *      per gang -- 64 KiB at n=8192, scaling with N with no bound.
 *      Here, each gang holds only BLOCK_M x BLOCK_N-shaped tiles
 *      (score tile, etc.) and BLOCK_M-shaped running statistics --
 *      FIXED size regardless of N, derived below from measured
 *      hardware shape (see moa_delta_benchmarking_tr.pdf, Section 9's
 *      nvaccelinfo table), not left to the compiler's discretion.
 *
 *   2. ATOMICS, ELIMINATED ENTIRELY. v1/v2 needed atomic updates
 *      because many WORKERS within the same gang all contributed to
 *      the same Out[ir,:] row. Here, each gang owns BLOCK_M query
 *      rows EXCLUSIVELY and loops over ALL key/value tiles
 *      SEQUENTIALLY within itself (online softmax: running max,
 *      running sum, incrementally rescaled accumulator -- the same
 *      algorithm moa_forward_triton.py already uses), writing its
 *      own output exactly once at the end. No two gangs, and no two
 *      workers within a gang, ever write the same memory -- matching
 *      exactly why Triton's kernel needs no atomics either.
 *
 * TILE SIZE DERIVATION (not chosen arbitrarily): all three GPUs
 * tested this project (V100, A100, H100) measure identically at 48
 * KiB shared-memory-per-block via nvaccelinfo, despite differing
 * substantially in SM count and L2 cache size. At fp64, D=64, holding
 * Q/K/V tiles, the score tile, and the output accumulator
 * simultaneously:
 *   BLOCK_M=BLOCK_N=32: 72 KiB (OVER budget)
 *   BLOCK_M=BLOCK_N=16: 34 KiB (fits, real margin)
 * BLOCK_M=BLOCK_N=16 is used below for exactly this reason -- derived
 * from a measured hardware constant common to all three GPUs, not
 * swept or guessed.
 *
 * num_workers(16)/vector_length(32) are carried over unchanged from
 * the v1/v2 derivation (64 max resident warps / 4 schedulers per SM,
 * also identical across all three GPUs per the same measurement).
 *
 * CORRECTNESS: as with every OpenACC file in this project, a local
 * (no-GPU) smoke test can confirm the arithmetic is right but cannot
 * expose a race condition under a serial fallback. This design has a
 * structural argument for race-freedom (no two gangs or workers ever
 * write the same address) rather than relying on atomics to paper
 * over one -- but the real-hardware self-test below is still the
 * check that actually matters, on all three GPUs, before any timing
 * from this file is trusted.
 * ============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef B
#define B      2
#endif
#ifndef N
#define N      512      /* override at compile time: -DN=1024 etc. */
#endif
#define D      64
#ifndef BLOCK_M
#define BLOCK_M {{BLOCK_M}}       /* derived from {{DEVICE_NAME}}'s measured shared-mem budget */
#endif
#ifndef BLOCK_N
#define BLOCK_N {{BLOCK_N}}
#endif
#ifndef REPEATS
#define REPEATS 5
#endif
#ifndef WARMUP
#define WARMUP 2
#endif

#define QKV(b,i,d)   ((b)*N*D + (i)*D + (d))

static void fill_random(double *arr, int n, unsigned long *state)
{
    for (int i = 0; i < n; i++) {
        *state = (*state) * 6364136223846793005ULL + 1442695040888963407ULL;
        arr[i] = (double)(*state & 0x7FFFFFFFULL) / (double)0x7FFFFFFF - 1.0;
    }
}

static void softmax_row(double *row, int n)
{
    double m = -DBL_MAX;
    for (int i = 0; i < n; i++) if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - m); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

static double max_abs_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* Reference: same DNF, unmodified, CPU, sequential. */
static void moa_forward_dnf_ref(
    const double *Q, const double *K, const double *V,
    double *Out, double scale)
{
    double arow[N];
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                arow[ic] = 0.0;
                for (int j = 0; j < D; j++)
                    arow[ic] += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                arow[ic] *= scale;
            }
            softmax_row(arow, N);
            for (int id = 0; id < D; id++) {
                Out[QKV(b,ir,id)] = 0.0;
                for (int ic = 0; ic < N; ic++)
                    Out[QKV(b,ir,id)] += arow[ic] * V[QKV(b,ic,id)];
            }
        }
    }
}

/* ============================================================================
 * OpenACC realization, tiled -- BLOCK_M query rows per gang, sequential
 * loop over BLOCK_N-sized key/value tiles, online softmax. No atomics.
 * ============================================================================ */
void moa_forward_openacc_tiled(
    const double *Q, const double *K, const double *V,
    double *Out, double scale)
{
    int num_row_tiles = (N + BLOCK_M - 1) / BLOCK_M;

    /* gang = one (b, row_tile) pair -- each gang owns BLOCK_M query
     * rows EXCLUSIVELY, for the entire kernel. No other gang ever
     * touches these rows. */
    #pragma acc parallel loop collapse(2) \
        num_workers({{NUM_WORKERS}}) vector_length({{VECTOR_LENGTH}}) \
        present(Q, K, V, Out)
    for (int b = 0; b < B; b++) {
      for (int rt = 0; rt < num_row_tiles; rt++) {

        int row0 = rt * BLOCK_M;
        int rows_here = (row0 + BLOCK_M <= N) ? BLOCK_M : (N - row0);

        double q_tile[BLOCK_M][D];
        double acc_out[BLOCK_M][D];
        double m_i[BLOCK_M];
        double l_i[BLOCK_M];

        /* Load this gang's Q tile once; it is reused across every
         * key/value tile visited below. */
        #pragma acc loop worker
        for (int r = 0; r < BLOCK_M; r++) {
            #pragma acc loop vector
            for (int d = 0; d < D; d++) {
                q_tile[r][d] = (r < rows_here) ? Q[QKV(b, row0+r, d)] : 0.0;
                acc_out[r][d] = 0.0;
            }
        }
        #pragma acc loop worker
        for (int r = 0; r < BLOCK_M; r++) {
            m_i[r] = -DBL_MAX;
            l_i[r] = 0.0;
        }

        int num_col_tiles = (N + BLOCK_N - 1) / BLOCK_N;

        for (int ct = 0; ct < num_col_tiles; ct++) {
            int col0 = ct * BLOCK_N;
            int cols_here = (col0 + BLOCK_N <= N) ? BLOCK_N : (N - col0);

            double k_tile[BLOCK_N][D];
            double v_tile[BLOCK_N][D];
            double s_tile[BLOCK_M][BLOCK_N];

            #pragma acc loop worker
            for (int c = 0; c < BLOCK_N; c++) {
                #pragma acc loop vector
                for (int d = 0; d < D; d++) {
                    k_tile[c][d] = (c < cols_here) ? K[QKV(b, col0+c, d)] : 0.0;
                    v_tile[c][d] = (c < cols_here) ? V[QKV(b, col0+c, d)] : 0.0;
                }
            }

            /* Score tile: S[r][c] = scale * sum_d q_tile[r][d]*k_tile[c][d] */
            #pragma acc loop worker
            for (int r = 0; r < BLOCK_M; r++) {
                #pragma acc loop vector
                for (int c = 0; c < BLOCK_N; c++) {
                    double acc = 0.0;
                    for (int d = 0; d < D; d++)
                        acc += q_tile[r][d] * k_tile[c][d];
                    acc *= scale;
                    int valid = (r < rows_here) && (c < cols_here);
                    s_tile[r][c] = valid ? acc : -DBL_MAX;
                }
            }

            /* Online softmax update, one row at a time (each row
             * owned by exactly one worker -- no cross-row sharing). */
            #pragma acc loop worker
            for (int r = 0; r < BLOCK_M; r++) {
                double m_new = m_i[r];
                for (int c = 0; c < BLOCK_N; c++)
                    if (s_tile[r][c] > m_new) m_new = s_tile[r][c];

                double alpha = exp(m_i[r] - m_new);
                double l_new = l_i[r] * alpha;
                double p_row[BLOCK_N];
                for (int c = 0; c < BLOCK_N; c++) {
                    p_row[c] = (s_tile[r][c] > -DBL_MAX/2.0)
                                 ? exp(s_tile[r][c] - m_new) : 0.0;
                    l_new += p_row[c];
                }

                #pragma acc loop vector
                for (int d = 0; d < D; d++) {
                    double acc = acc_out[r][d] * alpha;
                    for (int c = 0; c < BLOCK_N; c++)
                        acc += p_row[c] * v_tile[c][d];
                    acc_out[r][d] = acc;
                }

                m_i[r] = m_new;
                l_i[r] = l_new;
            }
        }

        /* Final normalization and the ONLY write to Out this gang
         * ever performs -- no atomics, no other gang shares these
         * rows. */
        #pragma acc loop worker
        for (int r = 0; r < BLOCK_M; r++) {
            if (r >= rows_here) continue;
            #pragma acc loop vector
            for (int d = 0; d < D; d++)
                Out[QKV(b, row0+r, d)] = acc_out[r][d] / l_i[r];
        }
      }
    }
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void)
{
    const double scale = 1.0 / sqrt((double)D);
    const int nqkv = B * N * D;

    double *Q = malloc(nqkv * sizeof(double));
    double *K = malloc(nqkv * sizeof(double));
    double *V = malloc(nqkv * sizeof(double));

    unsigned long state = 42UL;
    fill_random(Q, nqkv, &state);
    fill_random(K, nqkv, &state);
    fill_random(V, nqkv, &state);

    double *Out_ref = calloc(nqkv, sizeof(double));
    double *Out_acc = calloc(nqkv, sizeof(double));

    moa_forward_dnf_ref(Q, K, V, Out_ref, scale);

    printf("============================================================\n");
    printf(" MoA Forward Attention -- OpenACC TILED, B=%d N=%d D=%d BLOCK_M=%d BLOCK_N=%d\n",
           B, N, D, BLOCK_M, BLOCK_N);
    printf("============================================================\n");

    #pragma acc data copyin(Q[0:nqkv], K[0:nqkv], V[0:nqkv]) \
                      copyout(Out_acc[0:nqkv])
    {
        moa_forward_openacc_tiled(Q, K, V, Out_acc, scale);
    }

    double err = max_abs_diff(Out_acc, Out_ref, nqkv);
    int ok = err < 1e-8;
    printf("Correctness vs CPU reference -- max|err|: Out=%.4e  [%s]\n",
           err, ok ? "PASS" : "FAIL");

    if (!ok) {
        printf("*** Correctness FAILED -- timing below is NOT trustworthy. ***\n");
        return 1;
    }

    #pragma acc data copyin(Q[0:nqkv], K[0:nqkv], V[0:nqkv]) \
                      copyout(Out_acc[0:nqkv])
    {
        for (int w = 0; w < WARMUP; w++)
            moa_forward_openacc_tiled(Q, K, V, Out_acc, scale);

        double t0 = now_sec();
        for (int r = 0; r < REPEATS; r++)
            moa_forward_openacc_tiled(Q, K, V, Out_acc, scale);
        double elapsed_ms = (now_sec() - t0) / REPEATS * 1e3;

        printf("MoA forward (OpenACC tiled)  B=%d  n=%-6d  time_ms=%.3f\n", B, N, elapsed_ms);
    }

    return 0;
}
