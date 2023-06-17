#include "doctest/doctest.h"
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xview.hpp"
#include "xtensor-signal/detrend.hpp"

TEST_SUITE("detrend") {

  TEST_CASE("noise") {
   auto npoints = 1000;
   auto noise = xt::random::randn<double>({npoints});
   auto x = 3 + 2*xt::linspace<double>(0, 1, npoints) + noise;
   auto max = xt::amax(xt::signal::detrend(x) - noise)();
   REQUIRE_EQ(max, doctest::Approx(.6));
  }
}
