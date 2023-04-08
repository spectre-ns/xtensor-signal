#include <chrono>
#include <cstddef>
#include <random>
#include <string>
#include <benchmark/benchmark.h>
#include "xtensor-signal/lfilter.hpp"

namespace xt
{
    namespace signal
    {
        namespace benchmark_lfilter
        {
            void lfilter(benchmark::State& state)
            {
                //credit https://rosettacode.org/wiki/Apply_a_digital_filter_(direct_form_II_transposed)#C++
                //define the signal 
                xt::xtensor<double, 1> sig = xt::empty<double>({state.range(0)});

                //Constants for a Butterworth filter (order 3, low pass)
                xt::xtensor<float, 1> a = { 1.00000000};
                xt::xtensor<float, 1> b = {
                    -1.43784223e-04, -8.36125348e-05,  1.19173505e-04,  4.25496657e-04,
                    6.94633526e-04,  7.24177306e-04,  3.49521545e-04, -4.34584932e-04,
                    -1.39563577e-03, -2.09179948e-03, -2.03164747e-03, -9.23555283e-04,
                    1.09077195e-03,  3.35008833e-03,  4.82910680e-03,  4.53234580e-03,
                    1.99921968e-03, -2.29961235e-03, -6.90182084e-03, -9.75295479e-03,
                    -9.00084038e-03, -3.91590949e-03,  4.45649391e-03,  1.32770185e-02,
                    1.86909952e-02,  1.72542180e-02,  7.54398817e-03, -8.67689478e-03,
                    -2.63091689e-02, -3.80338221e-02, -3.64895345e-02, -1.68613380e-02,
                    2.10120036e-02,  7.18301706e-02,  1.25749185e-01,  1.70866876e-01,
                    1.96551032e-01,  1.96551032e-01,  1.70866876e-01,  1.25749185e-01,
                    7.18301706e-02,  2.10120036e-02, -1.68613380e-02, -3.64895345e-02,
                    -3.80338221e-02, -2.63091689e-02, -8.67689478e-03,  7.54398817e-03,
                    1.72542180e-02,  1.86909952e-02,  1.32770185e-02,  4.45649391e-03,
                    -3.91590949e-03, -9.00084038e-03, -9.75295479e-03, -6.90182084e-03,
                    -2.29961235e-03,  1.99921968e-03,  4.53234580e-03,  4.82910680e-03,
                    3.35008833e-03,  1.09077195e-03, -9.23555283e-04, -2.03164747e-03,
                    -2.09179948e-03, -1.39563577e-03, -4.34584932e-04,  3.49521545e-04,
                    7.24177306e-04,  6.94633526e-04,  4.25496657e-04,  1.19173505e-04,
                    -8.36125348e-05, -1.43784223e-04
                };

                auto filt = xt::signal::lfilter<double>();
                auto res = filt.set_coeffs(b,a);
                for (auto _ : state)
                {
                    auto z = filt(sig);
                    benchmark::DoNotOptimize(z);
                }
                state.SetComplexityN(state.range(0));
            }
            BENCHMARK(lfilter)->Range(2, 1048576)->Complexity();
        }
    }
}