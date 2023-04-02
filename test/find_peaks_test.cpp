
#include "doctest/doctest.h"
#include "xtensor/xview.hpp"
#include "xtensor/xsort.hpp"
#include "xtensor/xio.hpp"
#include "xtensor-io/xnpz.hpp"

#include "xtensor-signal/find_peaks.hpp"

TEST_SUITE("find_peaks")
{
	TEST_CASE("FindSinglePeak")
	{
		//generate a gaussian curve
		//defines
		auto mu = 1;
		auto sigma = 2;
		auto start = -5;
		auto end = 5;
		auto numsamples = 10000;
		auto x = xt::linspace<double>(start, end, numsamples);
		auto y = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

		//we should have a single peak at x = 1
		auto peaks = xt::signal::find_peaks(y);

		//assert that we have one peak
		ASSERT_TRUE(peaks.shape(0) == 1);

		//assert that the value is x = 0
		ASSERT_NEAR(x(peaks(0)), 1, 1e-3);
	}

	TEST_CASE("RandNoise")
	{
		//run through some random noise and make sure the
		// algorithm doesn't dies on random noise.
		//defines
		auto start = -5;
		auto end = 5;
		auto numsamples = 10000;
		auto x = xt::linspace<double>(start, end, numsamples);
		auto y = xt::random::randn<double>(x.shape());

		auto peaks = xt::signal::find_peaks(y);
	}

	TEST_CASE("BiModal")
	{
		//generate a gaussian curve
		//defines
		auto mu = 1;
		auto sigma = 2;
		auto start = -5;
		auto end = 15;
		auto numsamples = 10000;
		auto x = xt::linspace<double>(start, end, numsamples);
		xt::xarray<double> y1 = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

		mu = 10;
		xt::xarray<double> y2 = (1 / (sigma * std::sqrt(2 * xt::numeric_constants<double>::PI))) * xt::exp(-.5 * xt::pow((x - mu), 2) / (std::pow(sigma, 2)));

		auto y = y1 + y2;

		//we should have a single peak at x = 1
		auto peaks = xt::signal::find_peaks(y);

		//assert that we have two peaks
		ASSERT_TRUE(peaks.shape(0) == 2);

		//assert that the value is x = 1
		auto res = x(peaks(0));
		ASSERT_NEAR(res, 1.0, 1e-3);

		//assert that the value is x = 10
		res = x(peaks(1));
		ASSERT_NEAR(res, 10.0, 1e-3);
	}
	TEST_CASE("ecg_widths_prominance")
	{
		xt::xarray<size_t> prominance = { 1 };
		xt::xarray<size_t> width = { 20 };
		xt::xarray<double> expectation = { 49, 691 };
		auto x = xt::load_npz<double>("test_data/ecg.npz", "data");
		x = xt::view(x, xt::range(17000, 18000));
		auto peaks = xt::signal::find_peaks(x, xt::xnone(), xt::xnone(), xt::xnone(), prominance, width);
		ASSERT_EQ(peaks(0), 49);
		ASSERT_EQ(peaks(1), 691);
	}
	TEST_CASE("select_by_property")
	{
		size_t width = 20 ;
		xt::xarray<size_t> widths = { 20, 21, 22, 10, 15 };
		auto keep = xt::signal::detail::select_by_property(widths, width);
		ASSERT_TRUE(keep(0));
		ASSERT_TRUE(keep(1));
		ASSERT_TRUE(keep(2));
		ASSERT_FALSE(keep(3));
		ASSERT_FALSE(keep(4));
	}

	TEST_CASE("ecg_distance")
	{
		size_t distance = 150;
		xt::xarray<size_t> expectation = { 65,  251,  431,  608,  779,  956, 1125, 1292, 1456, 1614, 1776, 1948 };
		auto x = xt::load_npz<double>("test_data/ecg.npz", "data");
		x = xt::view(x, xt::range(2000, 4000));
		auto peaks = xt::signal::find_peaks(x, xt::xnone(),xt::xnone(), distance);
		for (size_t i = 0; i < expectation.shape(0); i++)
		{
			ASSERT_EQ(peaks(i), expectation(i));
		}
	}
}