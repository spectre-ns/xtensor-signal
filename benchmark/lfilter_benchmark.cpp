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
                xt::xtensor<float, 1> sig = xt::empty<float>({state.range(0)});

                //Constants for a Butterworth filter (order 3, low pass)
                xt::xtensor<float, 1> a = xt::zeros<float>({ 100 });
                xt::xtensor<float, 1> b = xt::zeros<float>({ 100 });

                auto filt = xt::signal::lfilter<float>();
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